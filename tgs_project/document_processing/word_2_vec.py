# Set up nltk
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

import os

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from typing import Union, List, Iterable
from nltk.corpus import stopwords
import numpy as np
from tgs_project.pipeline.pipeline import Stage
from tgs_project.document_processing.tf_idf_mapping import TfidfModel

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("dotenv not installed, skipping environment variable loading.")

class DocumentEmbedding:
    def __init__(
        self,
        documents: list[str],
        additional_stop_words: set[str] = set(),
        embedding_model: str = "word2vec",
        embedding_size: int = 100,
        window_size: int = 5,
        min_count: int = 1,
        workers: int = -1,
        remove_stop_words: bool = True,
        model_path: str = os.getenv("WORD_2_VEC_PATH", "word2vec.model"),
        tf_idf_model: Union[TfidfModel, None] = None,
        weighted_vectors: bool = False
    ):
        self.documents = documents
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
        n_workers = workers if workers > 0 else os.cpu_count()
        self.workers = n_workers if n_workers else 1

        self.model_path = model_path
        self.model = self.train_word2vec(self.preprocess())

    def preprocess(self):
        """Preprocess the documents by tokenizing and removing stop words."""
        processed_docs = []
        for doc in self.documents:
            tokens = word_tokenize(doc)
            if self.remove_stop_words:
                # Remove stop words
                filtered = [word.lower() for word in tokens if word.lower() not in self.stop_words]
            else:
                filtered = [token.lower() for token in tokens]
            processed_docs.append(filtered)
        return processed_docs

    def train_word2vec(self, processed_docs: list[list[str]]) -> Word2Vec:
        """Train a Word2Vec model on the processed documents."""
        if self.embedding_model != "word2vec":
            raise ValueError("Currently, only Word2Vec is supported.")
        # Train the Word2Vec model
        if not os.path.exists(self.model_path):
            model = Word2Vec(sentences=processed_docs, vector_size=self.embedding_size, window=self.window_size, min_count=self.min_count, workers=self.workers)
            model.save(self.model_path)
        else:
            model = Word2Vec.load(self.model_path)
        return model
    
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