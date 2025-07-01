import os
import pickle
from pathlib import Path
from typing import Iterable, Union, List, Callable
from itertools import repeat
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
import multiprocessing

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("dotenv not installed, skipping environment variable loading.")

N_CPU_CORES = os.cpu_count() if os.cpu_count() is not None else 1
if N_CPU_CORES is None:
    raise ValueError("Could not determine the number of CPU cores. Please set N_CPU_CORES manually.")

def _process_doc_to_str(doc, remove_stop_words, stop_words)-> str:
    tokens = word_tokenize(doc)
    if remove_stop_words:
        return " ".join([word.lower() for word in tokens if word.lower() not in stop_words])
    else:
        return doc

def preprocess(
    documents: List[str],
    len_of_documents: int | None = None,
    remove_stop_words: bool = True,
    stop_words: set[str] = set(stopwords.words("english")),
    doc_processing_function: Callable = _process_doc_to_str,
    n_workers: int = N_CPU_CORES,
    chunksize: int = 100
) -> list[str]:
    """
    Preprocess a list of documents in parallel.
    Args:
        documents: List of documents to preprocess.
        len_of_documents: Total number of documents for progress reporting.
        remove_stop_words: Whether to remove stop words.
        stop_words: Set of stop words to remove.
        doc_processing_function: Function to process each document.
        n_workers: Number of parallel workers to use.
    Returns:
        List of preprocessed documents.
    """
    if len_of_documents is None:
        raise ValueError("len_of_documents must be provided.")
    logger.info(f"Preprocessing {len_of_documents:,.0f} documents...")
    if isinstance(documents, str):
        logger.warning("documents is a string, converting to list of one document.")
        documents = [documents]
    if not isinstance(documents, list):
        raise ValueError("documents must be a materialized list of strings for parallel processing.")
    
    spawn_safe = multiprocessing.current_process().name == "MainProcess"

    if spawn_safe and n_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                return list(tqdm(
                    ex.map(
                        doc_processing_function,
                        documents,
                        repeat(remove_stop_words),
                        repeat(stop_words),
                        chunksize=chunksize
                    ),
                    total=len_of_documents,
                    desc=f"Tokenizing documents ({n_workers} workers)",))
        except Exception as e:
            logger.warning(f"Pool failed ({e}); falling back to serial.")

    # serial fallback
    return [doc_processing_function(d, remove_stop_words, stop_words)
            for d in tqdm(documents,
                            total=len_of_documents,
                            desc="Tokenizing documents (fallback)")]


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
        chunksize: int = 100,
        **sk_kwargs,
    ):
        self.model_path = Path(model_path)
        self.use_parquet = use_parquet
        self.vec: TfidfVectorizer = TfidfVectorizer(**sk_kwargs)
        self.idf: np.ndarray = np.array([])
        self.vocab: dict[str, int] = {}
        self.remove_stop_words = remove_stop_words
        self.stop_words = stop_words
        clean_n_workers = n_workers if n_workers != -1 else os.cpu_count()
        if clean_n_workers is None:
            raise ValueError("Could not determine the number of CPU cores. Please set n_workers manually.")
        self.n_workers = clean_n_workers
        self.chunksize = chunksize
        if self.model_path.exists():
            self._load()

    

    def fit(self, corpus: List[str]) -> "TfidfModel":
        """Fit on corpus and persist to disk."""
        if isinstance(corpus, Iterable):
            docs = list(corpus)
        len_of_documents = len(corpus)

        corpus = preprocess(
            corpus,
            len_of_documents=len_of_documents,
            n_workers=self.n_workers,
            remove_stop_words=self.remove_stop_words,
            chunksize=self.chunksize,
            stop_words=self.stop_words,
        )
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

