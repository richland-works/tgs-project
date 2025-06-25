from tgs_project.document_processing.tf_idf_mapping import TfidfModel
import os

# Load the dataset
from tgs_project.utils import read_jsonl, get_jsonl_n_count
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("dotenv not installed, skipping environment variable loading.")

coref_results_path = os.getenv("COREF_RESULTS_PATH", "coref_results.jsonl")
coref_results = (
    _['article']
    for _ in read_jsonl(
        coref_results_path
    )
)

# Initialize the TF-IDF mapping
tfidf_mapping = TfidfModel(n_workers=4)

# Fit the model on the dataset
tfidf_mapping.fit(coref_results, len_of_documents=get_jsonl_n_count(coref_results_path))

# Save the model after fitting
tfidf_mapping._save()

