from tgs_project.document_processing.resolve_coreferences import CorefResolver
from tgs_project.utils import load_cnn_dailymail

dataset = load_cnn_dailymail()
articles = list(dataset["train"])

coref_resolver = CorefResolver()
coref_resolver.process(articles)