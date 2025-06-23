import unittest
import spacy

class TestNERExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spacy.require_gpu()
        cls.nlp = spacy.load("en_core_web_trf")

    def test_named_entity_extraction(self):
        """Test named entity extraction using spaCy's transformer model.
        Just a simple test to ensure the model is loaded and can process text.
        """
        text = "Barack Obama was born in Hawaii and served as President of the United States."
        doc = self.nlp(text)
        ents = [(ent.text, ent.label_) for ent in doc.ents]

        self.assertIn(("Barack Obama", "PERSON"), ents)
        self.assertIn(("Hawaii", "GPE"), ents)
        self.assertIn(("the United States", "GPE"), ents)

