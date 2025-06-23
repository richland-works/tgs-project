import unittest
import spacy
from datasets import load_dataset

class TestNEROnRealArticle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spacy.require_gpu() # type: ignore
        cls.nlp = spacy.load("en_core_web_trf")
        cls.dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1]")

    def test_entities_present(self):
        article:str = self.dataset[0]["article"] # type: ignore
        doc = self.nlp(article)
        ents = [(ent.text, ent.label_) for ent in doc.ents]

        # Ensure at least a few known entity types are found
        entity_labels = {label for _, label in ents}
        self.assertTrue(any(label in entity_labels for label in {"PERSON", "ORG", "GPE", "DATE"}), 
                        "Expected at least one key entity label (PERSON, ORG, GPE, DATE)")
        