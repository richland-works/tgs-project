import os
from tgs_project.document_processing.word_2_vec import DocumentEmbedding
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
len_of_coref_results = get_jsonl_n_count(coref_results_path)
word2vec_model = DocumentEmbedding(
)
word2vec_model.train_or_load_word2vec(
    coref_results,
    len_of_documents=len_of_coref_results
)

