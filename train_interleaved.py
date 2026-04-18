"""
train_interleaved.py — Interleaved contrastive training for MATCHA.

Uses an InterleavedBatchLoader that cycles through data sources in
round-robin order, ensuring balanced exposure to each source within
every epoch. The loader is re-created each epoch so that within-source
shuffle order varies. Uses HuggingFace Accelerate for distributed training.
"""

import os
import json
import argparse
import logging
import random
from datetime import datetime
from types import SimpleNamespace

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    GPT2Model, GPT2Tokenizer, GPTNeoForCausalLM,
    AutoTokenizer, AutoModelForCausalLM,
    RobertaTokenizer, RobertaModel, AutoModelForMaskedLM,
)
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from huggingface_hub import login

from dataset.dataset_contrastive import ConDataset
from dataset.dataset_interleave import StreamingDataset, collate_fn, InterleavedBatchLoader
from model import ContrastiveModel

# Authenticate with HuggingFace Hub — set HF_TOKEN env var before running
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_random_seed(seed):
    """Set random seeds across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train a Contrastive Model")
    parser.add_argument('--config', default='configs/gpt2-small.yaml')
    parser.add_argument('--model-config', default='configs/gpt2small.json')
    parser.add_argument('--dataset', default='mixed_ft')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--restart', default=False)
    return parser.parse_args()


def setup_logging(output_path):
    """Create output directory and configure file-based logging."""
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_path, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


# ---------------------------------------------------------------------------
# Loss & training helpers
# ---------------------------------------------------------------------------

def modified_contrastive_loss(embedding_ref, embedding_pos, embedding_neg, margin=1.0):
    """Triplet margin loss using cosine similarity.

    Pushes positive pairs closer and negative pairs farther apart:
        loss = mean(relu(margin + neg_sim - pos_sim))
    """
    pos_sim = F.cosine_similarity(embedding_ref, embedding_pos, dim=-1)
    neg_sim = F.cosine_similarity(embedding_ref, embedding_neg, dim=-1)
    loss = torch.mean(F.relu(margin + neg_sim - pos_sim))
    return loss, torch.mean(pos_sim), torch.mean(neg_sim)


def save_checkpoint(model, optimizer, epoch, step, loss, output_path, accelerator, tag='best_model'):
    """Save model and optimizer state on the main process only."""
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        checkpoint_path = os.path.join(output_path, f'{tag}.pth')
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, checkpoint_path)


def train_step(model, accelerator, input_ref, input_pos, input_neg, optimizer, margin):
    """Run a single forward + backward pass with gradient accumulation."""
    with accelerator.accumulate(model):
        embedding_ref = model(input_ref)
        embedding_pos = model(input_pos)
        embedding_neg = model(input_neg)
        loss, pos_sim, neg_sim = modified_contrastive_loss(
            embedding_ref, embedding_pos, embedding_neg, margin
        )
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    return loss, pos_sim, neg_sim


def evaluate(model, val_loader, device, margin):
    """Run evaluation over the full validation set and return average metrics."""
    model.eval()
    total_val_loss, val_pos_sim, val_neg_sim = 0, 0, 0
    with torch.no_grad():
        for input_ref, input_pos, input_neg in tqdm(val_loader):
            input_ref = input_ref.to(device)
            input_pos = input_pos.to(device)
            input_neg = input_neg.to(device)
            embedding_ref = model(input_ref)
            embedding_pos = model(input_pos)
            embedding_neg = model(input_neg)
            val_loss, pos_sim, neg_sim = modified_contrastive_loss(
                embedding_ref, embedding_pos, embedding_neg, margin
            )
            total_val_loss += val_loss.item()
            val_pos_sim += pos_sim.item()
            val_neg_sim += neg_sim.item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_pos_sim = val_pos_sim / len(val_loader)
    avg_neg_sim = val_neg_sim / len(val_loader)
    return avg_val_loss, avg_pos_sim, avg_neg_sim


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(model, accelerator, train_dataset, val_loader, optimizer,
          num_epochs, start_epoch, start_step, best_loss,
          device, output_path, decay_rate, eval_steps, margin, config):
    """Full training loop with interleaved batch loading.

    A fresh InterleavedBatchLoader is created at the start of each epoch so
    that within-source shuffle order varies. Saves 'last_model' at every eval
    checkpoint, per-epoch checkpoints, and 'max_diff' when the
    (pos_sim - neg_sim) gap improves.
    """
    total_steps = 0
    max_diff = 0

    model = model.to(device)

    # Initial validation before training begins
    val_loss, val_pos_sim, val_neg_sim = evaluate(model, val_loader, device, margin)
    if accelerator.is_main_process:
        logging.info(f"Validation Loss: {val_loss:.4f}, Pos_Sim {val_pos_sim:.4f}, Neg_Sim: {val_neg_sim:.4f}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(model, optimizer, 0, 0, best_loss, output_path, accelerator, 'max_diff')

    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_train_loss, total_pos_sim, total_neg_sim = 0, 0, 0

        # Re-create the interleaved loader each epoch for fresh shuffle order
        train_loader = InterleavedBatchLoader(
            train_dataset,
            config['mixed_data'],
            batch_size=config['batch_size'],
            strategy='round_robin',
            shuffle_within=True,
        )
        train_loader = accelerator.prepare(train_loader)

        for step, batch in enumerate(tqdm(train_loader)):
            input_ref = batch['premise']
            input_pos = batch['correct']
            input_neg = batch['incorrect']

            # Skip steps already completed when resuming from a checkpoint
            if epoch == start_epoch and step <= start_step:
                continue

            input_ref = input_ref.to(device)
            input_pos = input_pos.to(device)
            input_neg = input_neg.to(device)

            loss, pos_sim, neg_sim = train_step(
                model, accelerator, input_ref, input_pos, input_neg, optimizer, margin
            )
            total_train_loss += loss.item()
            total_pos_sim += pos_sim.item()
            total_neg_sim += neg_sim.item()
            total_steps += 1

            # Periodic evaluation at fixed step intervals
            accelerator.wait_for_everyone()
            if eval_steps and total_steps % eval_steps == 0:
                avg_train_loss = total_train_loss / len(train_loader)
                avg_pos_sim = total_pos_sim / len(train_loader)
                avg_neg_sim = total_neg_sim / len(train_loader)
                val_loss, val_pos_sim, val_neg_sim = evaluate(model, val_loader, device, margin)

                if accelerator.is_main_process:
                    logging.info(f"Epoch {epoch+1}/{num_epochs} Step {total_steps}")
                    logging.info(f"Train Loss: {avg_train_loss:.4f} Pos_Sim: {avg_pos_sim:.4f}, Neg_Sim: {avg_neg_sim:.4f}")
                    logging.info(f"Validation Loss: {val_loss:.4f}, Pos_Sim: {val_pos_sim:.4f}, Neg_Sim: {val_neg_sim:.4f}")
                    save_checkpoint(model, optimizer, epoch, step, best_loss, output_path, accelerator, 'last_model')

                # Save checkpoint if max_diff improves
                if max_diff < val_pos_sim - val_neg_sim:
                    max_diff = val_pos_sim - val_neg_sim
                    save_checkpoint(model, optimizer, epoch, step, best_loss, output_path, accelerator, 'max_diff')
                    if accelerator.is_main_process:
                        logging.info(f"Model saved with max_diff: {max_diff:.4f} at Epoch {epoch+1} Step {step}")

                accelerator.wait_for_everyone()
                model.train()

            # Mid-epoch evaluation at the halfway point
            accelerator.wait_for_everyone()
            if step == len(train_loader) // 2:
                val_loss, val_pos_sim, val_neg_sim = evaluate(model, val_loader, device, margin)
                if accelerator.is_main_process:
                    logging.info(f"Epoch {epoch+1}/{num_epochs} Step {total_steps}")
                    logging.info(f"Validation Loss: {val_loss:.4f}, Pos_Sim: {val_pos_sim:.4f}, Neg_Sim: {val_neg_sim:.4f}")
                    save_checkpoint(model, optimizer, epoch, step, best_loss, output_path, accelerator, f'model_E{epoch+1}_half')
                accelerator.wait_for_everyone()
                model.train()

        accelerator.wait_for_everyone()

        # End-of-epoch metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_pos_sim = total_pos_sim / len(train_loader)
        avg_neg_sim = total_neg_sim / len(train_loader)

        val_loss, val_pos_sim, val_neg_sim = evaluate(model, val_loader, device, margin)

        if accelerator.is_main_process:
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            logging.info(f"Train Loss: {avg_train_loss:.4f}, Pos_Sim: {avg_pos_sim:.4f}, Neg_Sim: {avg_neg_sim:.4f}")
            logging.info(f"Validation Loss: {val_loss:.4f}, Pos_Sim: {val_pos_sim:.4f}, Neg_Sim: {val_neg_sim:.4f}")
            save_checkpoint(model, optimizer, epoch, step, best_loss, output_path, accelerator, f'model_E{epoch+1}')

        if max_diff < val_pos_sim - val_neg_sim:
            max_diff = val_pos_sim - val_neg_sim
            if accelerator.is_main_process:
                save_checkpoint(model, optimizer, epoch, step, best_loss, output_path, accelerator, 'max_diff')
                logging.info(f"Model saved with max_diff: {max_diff:.4f} at Epoch {epoch+1} Step {step}")

        # Decay the learning rate manually each epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] *= (1 - decay_rate)
        if accelerator.is_main_process:
            logging.info(f"Learning rate adjusted to: {optimizer.param_groups[0]['lr']}")

        accelerator.wait_for_everyone()
        model.train()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    print(vars(args))

    # Load training and model configs
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if args.checkpoint:
        with open(os.path.join(args.checkpoint, 'model_config.json'), 'r') as f:
            model_config = json.load(f)
    else:
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)

    # Set up output directory with timestamp (suffixed with _interleaved)
    time = datetime.now().strftime("%m-%d_%H-%M")
    dataset = args.dataset
    output_path = f"{config['output_path']}/{dataset}/{time}_interleaved"
    setup_logging(output_path)

    # Save configs for reproducibility
    yaml.dump(config, open(os.path.join(output_path, 'config.yaml'), 'w'))
    with open(os.path.join(output_path, 'model_config.json'), 'w') as f:
        json.dump(model_config, f)
        model_config = SimpleNamespace(**model_config)

    set_random_seed(config['seed'])
    config['learning_rate'] = float(config['learning_rate'])

    # Initialize Accelerator for distributed training
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    logging.info(f'Loading model on device: {device}')

    # -----------------------------------------------------------------------
    # Load backbone model and tokenizer based on config
    # -----------------------------------------------------------------------
    if 'gpt' in config['model_name']:
        tokenizer = GPT2Tokenizer.from_pretrained(config['tokenizer_name'])
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if 'gpt2' in config['model_name']:
            contxtl_model = GPT2Model.from_pretrained(config['model_name'])
        elif config['model_name'] == 'EleutherAI/gpt-neo-1.3B':
            contxtl_model = GPTNeoForCausalLM.from_pretrained(config['model_name'])

    elif 'Mistral' in config['model_name']:
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        contxtl_model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float32,
        )

    elif config['model_name'] == 'FacebookAI/roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(config['tokenizer_name'])
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        contxtl_model = RobertaModel.from_pretrained(config['model_name'])

    elif config['model_name'] == 'FacebookAI/xlm-roberta-large':
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
        tokenizer.pad_token = tokenizer.eos_token
        contxtl_model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

    # Wrap backbone in the contrastive head
    contrastive_model = ContrastiveModel(contxtl_model, model_config)
    optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=config['learning_rate'])

    # -----------------------------------------------------------------------
    # Optionally resume from checkpoint
    # -----------------------------------------------------------------------
    start_step = 0
    start_epoch = 0
    best_loss = float('inf')

    if args.checkpoint:
        model_state_path = os.path.join(args.checkpoint, 'last_model.pth')
        model_weights = torch.load(model_state_path, map_location=device)
        contrastive_model.load_state_dict(model_weights['model_state_dict'])
        optimizer.load_state_dict(model_weights['optimizer_state_dict'])
        if not args.restart:
            start_epoch = model_weights['epoch']
            start_step = model_weights['step']
            best_loss = model_weights['loss']
            logging.info(f"Starting from Epoch {start_epoch+1} Step {start_step}")
        logging.info(f"Model loaded from checkpoint: {args.checkpoint}")

    # -----------------------------------------------------------------------
    # Prepare datasets and data loaders
    # -----------------------------------------------------------------------
    train_dataset = StreamingDataset(
        data_path=config['mixed_data'],
        tokenizer=tokenizer,
        contexual_dim=config['contexual_dim'],
        max_samples=config['max_samples'],
    )
    val_dataset = ConDataset(
        config['dataset_path'], tokenizer, 'snli', 'test', config['contexual_dim']
    )

    print("Length of training dataset:", len(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=2, shuffle=False)

    # Only prepare model, optimizer, and val_loader here;
    # train_loader is created fresh each epoch inside train()
    contrastive_model, optimizer, val_loader = accelerator.prepare(
        contrastive_model, optimizer, val_loader
    )
    logging.info(f"Starting training with {len(val_loader)} validation steps")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    train(
        contrastive_model, accelerator, train_dataset, val_loader, optimizer,
        config['num_epochs'], start_epoch, start_step, best_loss,
        device, output_path, config['decay_rate'], config['eval_steps'], config['margin'],
        config,
    )
    logging.info("Training complete")
