# Set up nltk
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from tgs_project.logger import logger
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from typing import Union, List, Iterable
from nltk.corpus import stopwords
import numpy as np
from tgs_project.pipeline.pipeline import Stage
from tgs_project.document_processing.tf_idf_mapping import TfidfModel
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("dotenv not installed, skipping environment variable loading.")

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
            raise ValueError("If weighted_vectors is True, tf_idf_model must be provided.")

        # Get the number of workers
        # This is for typing purposes
        clean_n_workers: int|None = n_workers if n_workers != -1 else os.cpu_count()
        if clean_n_workers is None:
            raise ValueError("n_workers must be greater than 0 or set to -1 to use all available CPU cores.")
        self.n_workers: int = clean_n_workers

        self.model_path = model_path

    def _process_doc(self, doc, remove_stop_words, stop_words):
        tokens = word_tokenize(doc)
        if remove_stop_words:
            return [word.lower() for word in tokens if word.lower() not in stop_words]
        else:
            return [token.lower() for token in tokens]

    def preprocess(
        self,
        documents: List[str] | Iterable[str],
        len_of_documents: int | None = None # Used for reporting progress
    ) -> List[List[str]]:
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

    def train_or_load_word2vec(
        self,
        documents: List[str]| Iterable[str] | None = None,
        len_of_documents: int | None = None # Used for reporting progress
    ) -> None:
        """Train a Word2Vec model on the processed documents."""
        if self.embedding_model != "word2vec":
            raise ValueError("Currently, only Word2Vec is supported.")
        if not len_of_documents:
            if documents is None:
                raise ValueError("len_of_documents must be provided if documents are not provided.")
            
        # Train the Word2Vec model
        if not os.path.exists(self.model_path):
            if documents is None:
                raise ValueError("Documents must be provided to train the model.")
            processed_docs = self.preprocess(documents, len_of_documents=len_of_documents)
            self.model = Word2Vec(sentences=processed_docs, vector_size=self.embedding_size, window=self.window_size, min_count=self.min_count, workers=self.n_workers)
            self.model.save(self.model_path)
        else:
            self.model = Word2Vec.load(self.model_path)
    
    def embed(self, text: str) -> list[float]:
        """Embed a single text using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_word2vec() first.")
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
            raise ValueError("model_path must be provided.")
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