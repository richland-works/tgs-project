from tgs_project.document_processing.ner_extraction import ArticleEntityProcessor
from tgs_project.utils import load_cnn_dailymail

dataset = load_cnn_dailymail()
articles = list(dataset["train"])

ner_processor = ArticleEntityProcessor()
ner_processor.process(articles)