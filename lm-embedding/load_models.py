"""Embedding model loading and batched text encoding."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    backend: str
    model_name: str | None = None
    pooling: str = "mean"
    trust_remote_code: bool = False
    gensim_name: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "word2vec": ModelSpec(
        key="word2vec",
        display_name="Word2Vec",
        backend="gensim",
        gensim_name="word2vec-google-news-300",
    ),
    "glove-6B.300d": ModelSpec(
        key="glove-6B.300d",
        display_name="GloVe",
        backend="gensim",
        gensim_name="glove-wiki-gigaword-300",
    ),
    "bert-mean": ModelSpec(
        key="bert-mean",
        display_name="BERT (base)",
        backend="transformer",
        model_name="bert-base-uncased",
        pooling="mean",
    ),
    "s-bert": ModelSpec(
        key="s-bert",
        display_name="MiniLM (S-BERT)",
        backend="sentence_transformer",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ),
    "mpnet": ModelSpec(
        key="mpnet",
        display_name="MPNet",
        backend="sentence_transformer",
        model_name="sentence-transformers/all-mpnet-base-v2",
    ),
    "snet-t5": ModelSpec(
        key="snet-t5",
        display_name="T5-Large (Sentence-T5)",
        backend="sentence_transformer",
        model_name="sentence-transformers/sentence-t5-large",
    ),
    "distilbert": ModelSpec(
        key="distilbert",
        display_name="DistilBERT-NLI",
        backend="sentence_transformer",
        model_name="sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    ),
    "gte-large-mean": ModelSpec(
        key="gte-large-mean",
        display_name="GTE-Large",
        backend="transformer",
        model_name="thenlper/gte-large",
        pooling="mean",
    ),
    "e5-large-v2": ModelSpec(
        key="e5-large-v2",
        display_name="E5-Large-v2",
        backend="sentence_transformer",
        model_name="intfloat/e5-large-v2",
    ),
    "multilingual-e5-large-instruct": ModelSpec(
        key="multilingual-e5-large-instruct",
        display_name="E5-Large (Multilingual)",
        backend="sentence_transformer",
        model_name="intfloat/multilingual-e5-large-instruct",
    ),
    "e5-mistral-7b-instruct": ModelSpec(
        key="e5-mistral-7b-instruct",
        display_name="E5-Mistral-7B",
        backend="sentence_transformer",
        model_name="intfloat/e5-mistral-7b-instruct",
    ),
    "bge-large": ModelSpec(
        key="bge-large",
        display_name="BGE-Large",
        backend="sentence_transformer",
        model_name="BAAI/bge-large-en-v1.5",
    ),
    "bilingual-embedding-large": ModelSpec(
        key="bilingual-embedding-large",
        display_name="Bilingual-embedding-large",
        backend="sentence_transformer",
        model_name="Lajavaness/bilingual-embedding-large",
        trust_remote_code=True,
    ),
    "jina-embeddings-v3": ModelSpec(
        key="jina-embeddings-v3",
        display_name="Jina-embeddings-v3",
        backend="sentence_transformer",
        model_name="jinaai/jina-embeddings-v3",
        trust_remote_code=True,
    ),
    "jasper": ModelSpec(
        key="jasper",
        display_name="Jasper",
        backend="sentence_transformer",
        model_name="NovaSearch/jasper_en_vision_language_v1",
        trust_remote_code=True,
    ),
    "linq-mistral": ModelSpec(
        key="linq-mistral",
        display_name="Linq-Embed-Mistral",
        backend="sentence_transformer",
        model_name="Linq-AI-Research/Linq-Embed-Mistral",
    ),
    "sfr-mistral": ModelSpec(
        key="sfr-mistral",
        display_name="SFR-Mistral",
        backend="sentence_transformer",
        model_name="Salesforce/SFR-Embedding-Mistral",
    ),
    "stella": ModelSpec(
        key="stella",
        display_name="Stella",
        backend="sentence_transformer",
        model_name="NovaSearch/stella_en_1.5B_v5",
        trust_remote_code=True,
    ),
    "speed-embedding-7b-instruct-mean": ModelSpec(
        key="speed-embedding-7b-instruct-mean",
        display_name="Speed-7B-Instruct",
        backend="transformer",
        model_name="Haon-Chen/speed-embedding-7b-instruct",
        pooling="mean",
    ),
    "mistral-7b-mean": ModelSpec(
        key="mistral-7b-mean",
        display_name="Mistral-7B",
        backend="causal_transformer",
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        pooling="mean",
    ),
    "llama-2-13b-mean": ModelSpec(
        key="llama-2-13b-mean",
        display_name="LLaMA-2-13B",
        backend="causal_transformer",
        model_name="meta-llama/Llama-2-13b-hf",
        pooling="mean",
    ),
    "llama-3-8b-mean": ModelSpec(
        key="llama-3-8b-mean",
        display_name="LLaMA-3-8B",
        backend="causal_transformer",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        pooling="mean",
    ),
}

PAPER_EMBEDDING_MODEL_KEYS = (
    "bert-mean",
    "bge-large",
    "bilingual-embedding-large",
    "distilbert",
    "multilingual-e5-large-instruct",
    "e5-large-v2",
    "e5-mistral-7b-instruct",
    "glove-6B.300d",
    "gte-large-mean",
    "jasper",
    "jina-embeddings-v3",
    "linq-mistral",
    "llama-2-13b-mean",
    "llama-3-8b-mean",
    "s-bert",
    "mistral-7b-mean",
    "mpnet",
    "sfr-mistral",
    "speed-embedding-7b-instruct-mean",
    "stella",
    "snet-t5",
    "word2vec",
)


def available_model_keys() -> list[str]:
    """Return model keys accepted by the CLI."""
    return sorted(MODEL_SPECS)


def resolve_model_specs(keys: Iterable[str]) -> list[ModelSpec]:
    """Resolve requested model keys."""
    specs = []
    for key in keys:
        if key == "snpmi":
            raise ValueError(
                "SNPMI is not available from precomputed files anymore. "
                "Add the SNPMI computation logic before enabling this baseline."
            )
        if key not in MODEL_SPECS:
            supported = ", ".join(available_model_keys())
            raise ValueError(f"Unsupported model '{key}'. Supported: {supported}")
        specs.append(MODEL_SPECS[key])
    return specs


def with_pooling(spec: ModelSpec, pooling: str | None) -> ModelSpec:
    """Apply a pooling override to transformer-backed models."""
    if not pooling or spec.backend not in {"transformer", "causal_transformer"}:
        return spec
    key = spec.key
    if key.endswith(("-mean", "-max", "-pooler")):
        key = key.rsplit("-", 1)[0]
    return replace(spec, key=f"{key}-{pooling}", pooling=pooling)


def embedding_details(spec: ModelSpec) -> str:
    """Return a stable detail key for output filenames and tables."""
    return spec.key


def display_name(spec_or_key: ModelSpec | str) -> str:
    """Return the human-readable model name."""
    if isinstance(spec_or_key, ModelSpec):
        return spec_or_key.display_name
    fallback = ModelSpec(spec_or_key, spec_or_key, "")
    return MODEL_SPECS.get(spec_or_key, fallback).display_name


class EmbeddingEncoder:
    """Lazy-loading embedding encoder for one model."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 256,
        hf_token: str | None = None,
    ) -> None:
        self.spec = spec
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.model = None
        self.tokenizer = None
        self.vector_size = None

    def encode(self, texts: Iterable[object]) -> np.ndarray:
        """Encode texts into a 2D numpy array."""
        self._load()
        normalized = [normalize_text(text) for text in texts]

        if self.spec.backend == "sentence_transformer":
            vectors = self.model.encode(
                normalized,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.asarray(vectors, dtype=np.float32)

        if self.spec.backend == "gensim":
            return self._encode_gensim(normalized)

        if self.spec.backend in {"transformer", "causal_transformer"}:
            return self._encode_transformer(normalized)

        raise ValueError(f"Unknown backend: {self.spec.backend}")

    def _load(self) -> None:
        if self.model is not None:
            return

        if self.spec.backend == "sentence_transformer":
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if self.spec.trust_remote_code:
                kwargs["trust_remote_code"] = True
            if self.hf_token:
                kwargs["token"] = self.hf_token
            self.model = SentenceTransformer(self.spec.model_name, **kwargs)
            if self.device:
                self.model.to(self.device)
            return

        if self.spec.backend == "gensim":
            import gensim.downloader as api

            self.model = api.load(self.spec.gensim_name)
            self.vector_size = int(self.model.vector_size)
            return

        if self.spec.backend in {"transformer", "causal_transformer"}:
            import torch
            from transformers import AutoModel, AutoTokenizer

            kwargs = {"trust_remote_code": self.spec.trust_remote_code}
            if self.hf_token:
                kwargs["token"] = self.hf_token
            self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name, **kwargs)
            added_pad_token = False
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    added_pad_token = True
            self.model = AutoModel.from_pretrained(self.spec.model_name, **kwargs)
            if added_pad_token:
                self.model.resize_token_embeddings(len(self.tokenizer))
            target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(target_device)
            self.model.eval()
            self.device = target_device
            return

        raise ValueError(f"Unknown backend: {self.spec.backend}")

    def _encode_gensim(self, texts: list[str]) -> np.ndarray:
        vectors = []
        assert self.vector_size is not None
        for text in texts:
            token_vectors = [
                self.model[token]
                for token in tokenize_static_embedding_text(text)
                if token in self.model
            ]
            if token_vectors:
                vectors.append(np.mean(token_vectors, axis=0))
            else:
                vectors.append(np.full(self.vector_size, np.nan, dtype=np.float32))
        return np.asarray(vectors, dtype=np.float32)

    def _encode_transformer(self, texts: list[str]) -> np.ndarray:
        import torch

        vectors = []
        assert self.model is not None
        assert self.tokenizer is not None

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)

            pooled = pool_transformer_output(output, encoded["attention_mask"], self.spec.pooling)
            vectors.append(pooled.detach().cpu().numpy())

        return np.concatenate(vectors, axis=0).astype(np.float32)


def pool_transformer_output(output, attention_mask, pooling: str):
    """Pool token embeddings from a transformer output."""
    if pooling == "pooler" and hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output

    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and getattr(output, "hidden_states", None):
        hidden = output.hidden_states[-1]
    if hidden is None:
        raise ValueError("Transformer output did not include hidden states")
    if pooling == "max":
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).bool()
        hidden = hidden.masked_fill(~mask, float("-inf"))
        return hidden.max(dim=1).values

    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    raise ValueError(f"Unsupported pooling '{pooling}'")


def tokenize_static_embedding_text(text: str) -> list[str]:
    """Tokenize text for static word-vector models."""
    cleaned = clean_text(text.lower())
    try:
        from nltk.tokenize import word_tokenize

        return word_tokenize(cleaned)
    except LookupError:
        return cleaned.split()


def clean_text(text: str) -> str:
    """Remove punctuation while preserving word characters and whitespace."""
    return re.sub(r"[^\w\s'-]", " ", text)


def normalize_text(value: object) -> str:
    """Normalize text input for embedding models."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except TypeError:
        pass
    return " ".join(str(value).split())
