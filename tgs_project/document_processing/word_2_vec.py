# Set up nltk
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

import os
from itertools import repeat
import platform
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from typing import Union, List, Iterable, Callable
from nltk.corpus import stopwords
import numpy as np
from tgs_project.pipeline.pipeline import Stage
from tgs_project.document_processing.tf_idf_mapping import TfidfModel
from tqdm import tqdm
import multiprocessing
import logging

logger = logging.getLogger(__name__)

N_CPU_CORES = os.cpu_count() if os.cpu_count() is not None else 1
if N_CPU_CORES is None:
    raise ValueError("Could not determine the number of CPU cores. Please set N_CPU_CORES manually.")

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    logger.warning("dotenv not installed, skipping environment variable loading.")

def _tokenize_doc(doc, remove_stop_words, stop_words) -> list[str]:
    tokens = word_tokenize(doc)
    if remove_stop_words:
        return [word.lower() for word in tokens if word.lower() not in stop_words]
    else:
        return [token.lower() for token in tokens]

def preprocess(
    documents: List[str],
    len_of_documents: int | None = None,
    remove_stop_words: bool = True,
    stop_words: set[str] = set(stopwords.words("english")),
    doc_processing_function: Callable = _tokenize_doc,
    n_workers: int = N_CPU_CORES
) -> list[list[str]]:
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
        logger.info(f"Using {n_workers} workers for parallel processing.")
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                return list(tqdm(
                    ex.map(doc_processing_function,
                            documents,
                            repeat(remove_stop_words),
                            repeat(stop_words)),
                    total=len_of_documents,
                    desc="Tokenizing documents"))
        except Exception as e:
            logger.warning(f"Pool failed ({e}); falling back to serial.")

    # serial fallback
    return [doc_processing_function(d, remove_stop_words, stop_words)
            for d in tqdm(documents,
                            total=len_of_documents,
                            desc="Tokenizing documents (fallback)")]


class DocumentEmbedding:
    def __init__(
        self,
        additional_stop_words: set[str] = set(),
        embedding_model: str = "word2vec",
        embedding_size: int = 100,
        window_size: int = 5,
        min_count: int = 1,
        n_workers: int = -1,
        remove_stop_words: bool = True,
        model_path: str = os.getenv("WORD_2_VEC_MODEL_PATH", "word2vec.model"),
        tf_idf_model: Union[TfidfModel, None] = None,
        weighted_vectors: bool = False
    ):
        self.stop_words = set(stopwords.words("english")) | additional_stop_words
        self.embedding_model = embedding_model
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.min_count = min_count
        self.remove_stop_words = remove_stop_words
        self.tf_idf_model = tf_idf_model
        self.weighted_vectors = weighted_vectors

        if self.weighted_vectors and not self.tf_idf_model:
            msg = (
                "If weighted_vectors is True, tf_idf_model must be provided. "
                "Please provide a TfidfModel instance to the DocumentEmbedding constructor."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Get the number of workers
        # This is for typing purposes
        clean_n_workers: int|None = n_workers if n_workers != -1 else os.cpu_count()
        if clean_n_workers is None:
            msg = (
                "Could not determine the number of CPU cores. "
                "Please set n_workers to a positive integer."
            )
            logger.error(msg)
            raise ValueError(msg)
        self.n_workers: int = clean_n_workers

        self.model_path = model_path

    def train_word2vec(
        self,
        documents: List[str] = [],
        len_of_documents: int | None = None # Used for reporting progress
    ) -> None:
        """Train a Word2Vec model on the processed documents."""
        if self.embedding_model != "word2vec":
            msg = "Currently, only Word2Vec is supported."
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(documents, list):
            logger.warning("documents is not a list, converting to list.")
            documents = list(documents)
        if not documents:
            msg = "Documents must be provided to train the model."
            logger.error(msg)
            raise ValueError(msg)
        if len_of_documents is None:
            msg = "len_of_documents must be provided if documents are provided."
            logger.error(msg)
            raise ValueError(msg)
        # If we pickle the object, keep the length of documents
        if len_of_documents:
            self.len_of_documents = len_of_documents
        # Train the Word2Vec model
        processed_docs = preprocess(documents, len_of_documents=len_of_documents)
        self.model = Word2Vec(sentences=processed_docs, vector_size=self.embedding_size, window=self.window_size, min_count=self.min_count, workers=self.n_workers)

    def train_or_load_word2vec(
        self,
        documents: List[str] = [],
        len_of_documents: int | None = None # Used for reporting progress
    ) -> None:
        """Train a Word2Vec model on the processed documents."""
        if not os.path.exists(self.model_path):
            if self.embedding_model != "word2vec":
                msg = "Currently, only Word2Vec is supported."
                logger.error(msg)
                raise ValueError(msg)
            if not documents:
                msg = "Documents must be provided to train the model."
                logger.error(msg)
                raise ValueError(msg)
            if not isinstance(documents, list):
                logger.warning("documents is not a list, converting to list.")
                documents = list(documents)
            if len_of_documents is None:
                msg = "len_of_documents must be provided if documents are provided."
                logger.error(msg)
                raise ValueError(msg)
            # Train the Word2Vec model
            processed_docs = preprocess(documents, len_of_documents=len_of_documents)
            self.model = Word2Vec(sentences=processed_docs, vector_size=self.embedding_size, window=self.window_size, min_count=self.min_count, workers=self.n_workers)
        else:
            self.model = Word2Vec.load(self.model_path)

    def _save(self) -> None:
        """Save the trained model to the specified path."""
        if self.model is None:
            msg = "Model not trained. Call train_word2vec() first."
            logger.error(msg)
            raise ValueError(msg)
        self.model.save(self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def embed(self, text: str) -> list[float]:
        """Embed a single text using the trained model."""
        if self.model is None:
            msg = "Model not trained. Call train_word2vec() first."
            logger.error(msg)
            raise ValueError(msg)
        tokens = word_tokenize(text)
        if self.remove_stop_words:
            tokens = [word.lower() for word in tokens if word.lower() not in self.stop_words]
        else:
            tokens = [token.lower() for token in tokens]
        if self.weighted_vectors and self.tf_idf_model:
            vectors, weights = zip(*[
                (self.model.wv[word], self.tf_idf_model.weight(word))
                for word in tokens if word in self.model.wv
            ])
        else:
            vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
            weights = [1.0] * len(vectors)
        if not vectors:
            return np.zeros(self.model.vector_size).astype(float).tolist()
        vectors = np.stack(vectors)
        weights = np.array(weights)
        return np.average(vectors, axis=0, weights=weights).astype(float).tolist()

class DocumentWeightedEmbeddingStage(DocumentEmbedding,Stage):
    def __init__(
        self,
        model_path: str = os.getenv("WORD_2_VEC_PATH", "word2vec.model"),
        additional_stop_words: set[str] = set(),
        embedding_size: int = 100,
        window_size: int = 5,
        min_count: int = 1,
        workers: int = -1,
        remove_stop_words: bool = True,
    ):
        if model_path is None:
            msg = "model_path must be provided."
            logger.error(msg)
            raise ValueError(msg)
        self.model_path = model_path
        self.weighted_vectors = True
        self.additional_stop_words = additional_stop_words
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.min_count = min_count
        self.workers = workers
        self.remove_stop_words = remove_stop_words
    def fn(self, data: Iterable[str]) -> Iterable[list[float]]:
        """Apply the embedding to each document in the data."""
        for item in data:
            yield self.embed(item)