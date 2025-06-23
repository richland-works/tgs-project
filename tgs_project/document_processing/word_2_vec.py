# Set up nltk
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

import os
import json

from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords
import numpy as np

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
        remove_stop_words: bool = False,
        model_path: str = os.getenv("WORD_2_VEC_PATH", "word2vec.model")
    ):
        self.documents = documents
        self.stop_words = set(stopwords.words("english")) | additional_stop_words
        self.embedding_model = embedding_model
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.min_count = min_count
        self.remove_stop_words = remove_stop_words

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
                filtered = [word for word in tokens if word.lower() not in self.stop_words]
            else:
                filtered = tokens
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
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        if not vectors:
            return np.zeros(model.vector_size).astype(float) # type: ignore
        return np.mean(vectors, axis=0).astype(float)