"""Model specs, backend loading, and batched encoders for LM embedding experiments"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Iterable

import numpy as np

try:
    from nltk.tokenize import word_tokenize as nltk_word_tokenize
except ImportError:
    nltk_word_tokenize = None


@dataclass(frozen=True)
class ModelSpec:
    """Small registry row for one embedding backend"""

    key: str
    display_name: str
    backend: str
    model_name: str | None = None
    pooling: str = "mean"
    trust_remote_code: bool = False
    model_family: str = "encoder"  # other option we handle: causal
    gensim_name: str | None = None  # only req for word2vec & glove


MODEL_SPECS: dict[str, ModelSpec] = {
    "word2vec": ModelSpec(
        "word2vec", "Word2Vec", "gensim", gensim_name="word2vec-google-news-300"
    ),
    "glove": ModelSpec(
        "glove", "GloVe", "gensim", gensim_name="glove-wiki-gigaword-300"
    ),
    "bert": ModelSpec("bert", "BERT (base)", "transformer", "bert-base-uncased"),
    "s-bert": ModelSpec(
        "s-bert", "MiniLM (S-BERT)", "sentence-transformer", "all-MiniLM-L6-v2"
    ),
    "mpnet": ModelSpec("mpnet", "MPNet", "sentence-transformer", "all-mpnet-base-v2"),
    "distilbert": ModelSpec(
        "distilbert",
        "DistilBERT-NLI",
        "sentence-transformer",
        "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    ),
    "snet-t5": ModelSpec(
        "snet-t5",
        "T5-Large (Sentence-T5)",
        "sentence-transformer",
        "sentence-transformers/sentence-t5-large",
    ),
    "e5-large-v2": ModelSpec(
        "e5-large-v2", "E5-Large-v2", "sentence-transformer", "intfloat/e5-large-v2"
    ),
    "bge-large": ModelSpec(
        "bge-large", "BGE-Large", "sentence-transformer", "BAAI/bge-large-en-v1.5"
    ),
    "gte-large": ModelSpec(
        "gte-large", "GTE-Large", "transformer", "thenlper/gte-large"
    ),
    "mistral-7b": ModelSpec(
        "mistral-7b",
        "Mistral-7B",
        "transformer",
        "mistralai/Mistral-7B-Instruct-v0.2",
        model_family="causal",
    ),
    "llama-2-13b": ModelSpec(
        "llama-2-13b",
        "LLaMA-2-13B",
        "transformer",
        "meta-llama/Llama-2-13b-hf",
        model_family="causal",
    ),
    "llama-3.1-8B-Instruct": ModelSpec(
        "llama-3.1-8B-Instruct",
        "LLaMA-3.1-8B",
        "transformer",
        "meta-llama/Llama-3.1-8B-Instruct",
        model_family="causal",
    ),
    "e5-mistral-7b-instruct": ModelSpec(
        "e5-mistral-7b-instruct",
        "E5-Mistral-7B",
        "sentence-transformer",
        "intfloat/e5-mistral-7b-instruct",
    ),
    "speed-embedding-7b-instruct": ModelSpec(
        "speed-embedding-7b-instruct",
        "SpeedEmbed-7B-Instruct",
        "transformer",
        "Haon-Chen/speed-embedding-7b-instruct",
    ),
    "sfr-mistral": ModelSpec(
        "sfr-mistral",
        "SFR-Mistral",
        "sentence-transformer",
        "Salesforce/SFR-Embedding-Mistral",
    ),
    "ling-mistral": ModelSpec(
        "ling-mistral",
        "Linq-Embed-Mistral",
        "sentence-transformer",
        "Linq-AI-Research/Linq-Embed-Mistral",
    ),
    "multilingual-e5-large-instruct": ModelSpec(
        "multilingual-e5-large-instruct",
        "E5-Large (Multilingual)",
        "sentence-transformer",
        "intfloat/multilingual-e5-large-instruct",
    ),
    "jasper": ModelSpec(
        "jasper",
        "Jasper",
        "sentence-transformer",
        "NovaSearch/jasper_en_vision_language_v1",
        trust_remote_code=True,
    ),
    "stella": ModelSpec(
        "stella",
        "Stella",
        "sentence-transformer",
        "NovaSearch/stella_en_1.5B_v5",
        trust_remote_code=True,
    ),
    "bilingual-embedding-large": ModelSpec(
        "bilingual-embedding-large",
        "Bilingual-Embedding-Large",
        "sentence-transformer",
        "Lajavaness/bilingual-embedding-large",
        trust_remote_code=True,
    ),
    "jina-embeddings-v3": ModelSpec(
        "jina-embeddings-v3",
        "Jina-Embeddings-V3",
        "sentence-transformer",
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True,
    ),
}


def available_model_keys() -> list[str]:
    """Return model keys accepted by --models"""
    return list(MODEL_SPECS)


def resolve_model_specs(model_keys: Iterable[str]) -> list[ModelSpec]:
    """Resolve CLI model keys to registry specs"""
    specs = []
    for key in model_keys:
        if key not in MODEL_SPECS:
            known = ", ".join(sorted(MODEL_SPECS))
            raise ValueError(f"Unknown model '{key}', known models: {known}")
        specs.append(MODEL_SPECS[key])
    return specs


def embedding_details(spec: ModelSpec) -> str:
    """Return the filename/detail label used in saved score files"""
    if spec.key == "glove":
        return "glove-6B.300d"
    if spec.backend == "transformer":
        return f"{spec.key}-{spec.pooling}"
    return spec.key


def display_name(model_key_or_details: str) -> str:
    """Return the plot label for a model key or a saved detail label"""
    if model_key_or_details in MODEL_SPECS:
        return MODEL_SPECS[model_key_or_details].display_name
    reverse = {
        embedding_details(spec): spec.display_name for spec in MODEL_SPECS.values()
    }
    return reverse.get(model_key_or_details, model_key_or_details)


def tokenize_static_embedding_text(text: str) -> list[str]:
    """Match the original Word2Vec/GloVe preprocessing and NLTK tokenization"""
    if nltk_word_tokenize is None:
        raise RuntimeError(
            "Word2Vec/GloVe tokenization requires NLTK, install it with `pip install nltk`"
        )

    sentence = clean_text(text)
    try:
        return nltk_word_tokenize(sentence.lower())
    except LookupError as exc:
        raise RuntimeError(
            "Word2Vec/GloVe tokenization requires NLTK tokenizer data, "
            "Install it with `python -m nltk.downloader punkt punkt_tab`"
        ) from exc


def clean_text(text: str) -> str:
    """Keep static embedding punctuation cleanup in one place"""
    return re.sub(r"[^\w\s'-]", "", text)


def prepare_embedding_texts(texts: list[object]) -> list[str]:
    """Clean up batch sentences before embedding while preserving input alignment."""
    return [normalize_text(value) for value in texts]


class EmbeddingEncoder:
    """Thin wrapper around the different embedding backends used in the paper"""

    def __init__(
        self,
        spec: ModelSpec,
        device: str | None = None,
        max_length: int | None = None,
    ):
        """Load one encoder and keep its runtime options together"""
        self.spec = spec
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        """Create the backend object for the selected model spec"""
        # Keep heavy backend imports local so plotting and metadata paths stay light
        if self.spec.backend == "sentence-transformer":
            from sentence_transformers import SentenceTransformer

            kwargs = {"trust_remote_code": self.spec.trust_remote_code}
            token = os.getenv("HF_TOKEN")
            if token:
                kwargs["token"] = token
            self.model = SentenceTransformer(self.spec.model_name, **kwargs)
            if self.device:
                self.model = self.model.to(self.device)
            return

        if self.spec.backend == "gensim":
            import gensim.downloader as api

            self.model = api.load(self.spec.gensim_name)
            return

        if self.spec.backend == "transformer":
            self._load_transformer()
            return

        raise ValueError(f"Unsupported backend: {self.spec.backend}")

    def _load_transformer(self) -> None:
        """Load transformer models with the per-model settings used for scoring"""
        import torch
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoTokenizer,
            BertModel,
            BertTokenizer,
        )

        if self.spec.key == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.spec.model_name)
            self.model = BertModel.from_pretrained(self.spec.model_name)
            self._finish_transformer_load()
            return

        if self.spec.key == "gte-large":
            self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name)
            self.model = AutoModel.from_pretrained(self.spec.model_name)
            self._finish_transformer_load()
            return

        if self.spec.key == "speed-embedding-7b-instruct":
            self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name)
            model_kwargs = {"torch_dtype": torch.float16}
            if self.device is None:
                model_kwargs["device_map"] = "auto"
            self.model = AutoModel.from_pretrained(self.spec.model_name, **model_kwargs)
            self._finish_transformer_load()
            return

        if self.spec.model_family == "causal":
            tokenizer_kwargs = self._tokenizer_auth_kwargs(
                use_cached_token=self.spec.key in {"mistral-7b", "llama-2-13b"}
            )
            model_kwargs = self._model_auth_kwargs()
            model_kwargs["torch_dtype"] = torch.float16
            if self.device is None:
                model_kwargs["device_map"] = "auto"

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.spec.model_name, **tokenizer_kwargs
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.spec.model_name, **model_kwargs
            )
            self._finish_transformer_load()
            return

        tokenizer_kwargs = self._tokenizer_auth_kwargs()
        model_kwargs = self._model_auth_kwargs()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model_name, **tokenizer_kwargs
        )
        self.model = AutoModel.from_pretrained(self.spec.model_name, **model_kwargs)
        self._finish_transformer_load()

    def _tokenizer_auth_kwargs(
        self, use_cached_token: bool = False
    ) -> dict[str, object]:
        """Build tokenizer auth kwargs without changing model math"""
        kwargs: dict[str, object] = {}
        if self.spec.trust_remote_code:
            kwargs["trust_remote_code"] = True
        token = os.getenv("HF_TOKEN")
        if token:
            kwargs["token"] = token
        elif use_cached_token:
            kwargs["token"] = True
        return kwargs

    def _model_auth_kwargs(self) -> dict[str, object]:
        """Build model auth kwargs for hosted weights"""
        kwargs: dict[str, object] = {}
        if self.spec.trust_remote_code:
            kwargs["trust_remote_code"] = True
        token = os.getenv("HF_TOKEN")
        if token:
            kwargs["token"] = token
        return kwargs

    def _finish_transformer_load(self) -> None:
        """Finish tokenizer padding, optional device override, and eval mode"""
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.device:
            self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts with the selected backend and return one vector per input row"""
        # Whitespace-normalize while preserving row alignment with the source table.
        texts = prepare_embedding_texts(texts)
        if not texts:
            return np.asarray([], dtype=np.float32)

        if self.spec.backend == "sentence-transformer":
            return np.asarray(
                self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                ),
                dtype=np.float32,
            )

        if self.spec.backend == "gensim":
            vectors = [self._encode_static(text) for text in texts]
            return np.asarray(vectors, dtype=np.float32)

        if self.spec.backend == "transformer":
            return self._encode_transformer(texts, batch_size=batch_size)

        raise ValueError(f"Unsupported backend: {self.spec.backend}")

    def _encode_static(self, text: str) -> np.ndarray:
        """Encode one text with the Word2Vec/GloVe averaging path"""
        tokens = tokenize_static_embedding_text(text)
        # Aggregate the vectors if they exist in the corpus
        vectors = [self.model[token] for token in tokens if token in self.model]
        if not vectors:
            return np.full(self.model.vector_size, np.nan, dtype=np.float32)
        # Average
        return np.mean(vectors, axis=0, dtype=np.float32)

    def _encode_transformer(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode transformer models in batches while preserving input order"""
        import torch

        outputs = []
        device = (
            self.device
            or getattr(self.model, "device", None)
            or next(self.model.parameters()).device
        )
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                # Leave max_length unset by default, use tokenizer truncation behavior
                **(
                    {"max_length": self.max_length}
                    if self.max_length is not None
                    else {}
                ),
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                model_output = self.model(
                    **inputs, output_hidden_states=True, return_dict=True
                )

            pooled = pool_transformer_output(
                model_output, inputs["attention_mask"], self.spec.pooling
            )
            outputs.append(pooled.detach().cpu().float().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


def pool_transformer_output(model_output, attention_mask, pooling: str):
    """Apply the requested transformer pooling mode"""
    if getattr(model_output, "hidden_states", None) is not None:
        token_embeddings = model_output.hidden_states[-1]
    else:
        token_embeddings = model_output.last_hidden_state

    if pooling == "mean":
        return mean_pool(token_embeddings, attention_mask)
    if pooling == "max":
        return max_pool(token_embeddings, attention_mask)
    if pooling == "pooler" and hasattr(model_output, "pooler_output"):
        return model_output.pooler_output
    raise ValueError(f"Invalid pooling type {pooling}")


def mean_pool(token_embeddings, attention_mask):
    """Average non-padding token embeddings"""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def max_pool(token_embeddings, attention_mask):
    """Take a max over non-padding token embeddings"""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).bool()
    masked_hidden = token_embeddings.masked_fill(~mask, float("-inf"))
    return masked_hidden.max(dim=1).values


def normalize_text(text: object) -> str:
    """Collapse missing values and repeated whitespace into plain text"""
    if text is None:
        return ""
    try:
        if np.isnan(text):  # type: ignore[arg-type]
            return ""
    except TypeError:
        pass
    return re.sub(r"\s+", " ", str(text)).strip()
