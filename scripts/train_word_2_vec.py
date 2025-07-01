import os
from tgs_project.document_processing.word_2_vec import DocumentEmbedding
from tgs_project.utils import read_jsonl, get_jsonl_n_count
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("dotenv not installed, skipping environment variable loading.")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    
    coref_results_path = os.getenv("COREF_RESULTS_PATH", "coref_results.jsonl")
    coref_results = [
        _['article']
        for _ in read_jsonl(
            coref_results_path
        )
    ]

    # Experiment to ensure evaluation
    word2vec_model = DocumentEmbedding(
        n_workers=4,
    )
    word2vec_model.train_or_load_word2vec(
        coref_results,
        len_of_documents=len(coref_results)
    )

