import os
import pickle
from pathlib import Path
from typing import Iterable, Union, List, Iterator
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tgs_project.logger import logger

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
        remove_stop_words: bool = True,
        stop_words: set[str] = set(stopwords.words("english")),
        n_workers: int = -1,
        **sk_kwargs,
    ):
        self.model_path = Path(model_path)
        self.use_parquet = use_parquet
        self.vec: TfidfVectorizer = TfidfVectorizer(**sk_kwargs)
        self.idf: np.ndarray = np.array([])
        self.vocab: dict[str, int] = {}
        self.remove_stop_words = remove_stop_words
        self.stop_words = stop_words
        self.n_workers = n_workers if n_workers != -1 else os.cpu_count()
        if self.model_path.exists():
            self._load()

    def _process_doc(self, doc, remove_stop_words, stop_words)-> str:
        tokens = word_tokenize(doc)
        if remove_stop_words:
            return " ".join([word.lower() for word in tokens if word.lower() not in stop_words])
        else:
            return " ".join([token.lower() for token in tokens])
    

    def preprocess(
        self,
        documents: List[str] | Iterable[str],
        len_of_documents: int | None = None # Used for reporting progress
    ) -> List[str]:
        if not isinstance(documents, Iterable):
            logger.info(f"Preprocessing {len(documents)} documents...")
        else:
            logger.info("Preprocessing documents...")
        if len_of_documents is None:
            raise ValueError("len_of_documents must be provided for progress reporting.")

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            processed_docs = list(tqdm(
                executor.map(
                        self._process_doc,
                        documents,
                        [self.remove_stop_words] * len_of_documents,
                        [self.stop_words] * len_of_documents
                    ),
                total=len_of_documents,
                desc="Tokenizing documents"
            ))
        return processed_docs

    def fit(self, corpus: List[str]|Iterable[str], len_of_documents: int | None = None) -> "TfidfModel":
        """Fit on corpus and persist to disk."""
        if not isinstance(corpus, Iterable):
            logger.info(f"Fitting TF-IDF on {len(corpus)} documents...")
        else:
            logger.info("Fitting TF-IDF on documents...")
        if len_of_documents is None:
            raise ValueError("len_of_documents must be provided for progress reporting.")
        corpus = self.preprocess(corpus, len_of_documents=len_of_documents)
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

