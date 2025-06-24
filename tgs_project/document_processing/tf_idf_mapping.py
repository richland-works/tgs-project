import os
import pickle
from pathlib import Path
from typing import Iterable, Union, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("dotenv not installed, skipping environment variable loading.")

class TfidfModel:
    """
    Lightweight TF-IDF wrapper that
    • fits / loads transparently
    • persists to *.pkl (fast) or *.parquet (optional)
    • exposes `weight(word)` for Word2Vec weighting
    """

    def __init__(
        self,
        model_path: Union[str, Path] = os.getenv("TFIDF_MODEL_PATH", "tfidf_model.pkl"),
        use_parquet: bool = False,
        **sk_kwargs,
    ):
        self.model_path = Path(model_path)
        self.use_parquet = use_parquet
        self.vec: TfidfVectorizer = TfidfVectorizer(**sk_kwargs)
        self.idf: np.ndarray = np.array([])
        self.vocab: dict[str, int] = {}
        if self.model_path.exists():
            self._load()

    def fit(self, corpus: Iterable[str]) -> "TfidfModel":
        """Fit on corpus and persist to disk."""
        X = self.vec.fit_transform(corpus)
        self.idf = self.vec.idf_
        self.vocab = self.vec.vocabulary_
        self._save()
        return self

    def transform(self, docs: Iterable[str]):
        """Return TF-IDF sparse matrix (lazy)."""
        self._ensure_ready()
        return self.vec.transform(docs)

    def weight(self, word: str) -> float:
        """
        IDF weight for a single token.
        0.0 ⇢ OOV or stop-word filtered.
        """
        self._ensure_ready()
        idx = self.vocab.get(word.lower(), None)  # lowercase to match analyser
        return float(self.idf[idx]) if idx is not None else 0.0

    def _ensure_ready(self):
        if self.idf is None or self.vocab is None:
            raise RuntimeError("Model not trained / loaded.")

    def _save(self):
        if self.use_parquet:
            import pandas as pd

            pd.DataFrame(
                {"token": list(self.vocab.keys()), "idf": self.idf}
            ).to_parquet(self.model_path.with_suffix(".parquet"), index=False)
        with open(self.model_path, "wb") as f:
            pickle.dump((self.vec, self.idf, self.vocab), f)

    def _load(self):
        with open(self.model_path, "rb") as f:
            self.vec, self.idf, self.vocab = pickle.load(f)

