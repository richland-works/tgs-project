from tgs_project.document_processing.sentence_processing import SentenceTokenizerPipeline
from tgs_project.pipeline.pipeline import DataBatch
from tgs_project.utils import read_jsonl, get_jsonl_n_count
import os
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("dotenv not installed, skipping environment variable loading.")


def main():
    # Load the dataset
    dataset = read_jsonl(os.getenv("COREF_RESULTS_PATH", "coref_results.jsonl"))
    # Get the number of lines in the dataset
    dataset_count = get_jsonl_n_count(os.getenv("COREF_RESULTS_PATH", "coref_results.jsonl"))
    print(f"Loaded {dataset_count} articles...")

    processed_ids = set()
    for idx, item in enumerate(
        read_jsonl(
            os.getenv("SENTENCE_RESULTS_PATH", "tokenized_sentences.jsonl"),
            create=True
        )
    ):
        processed_ids.add(item["id"])

    # Use the DataBatch class to wrap the dataset
    # and filter out already processed items
    dataset = DataBatch(
        data = dataset,
        count=dataset_count - len(processed_ids),
        processed_ids=processed_ids
    )
    print(f"Already processed {len(processed_ids)} articles... skipping them.")

    # Initialize the pipeline
    pipeline = SentenceTokenizerPipeline(
        n_workers=1,
        batch_size=int(os.getenv("BATCH_SIZE", 128)),
        flush_interval=int(os.getenv("FLUSH_INTERVAL", 512)),
        results_path=os.getenv("SENTENCE_RESULTS_PATH", "tokenized_sentences.jsonl"),
    )

    # Process the dataset
    dataset = dataset | pipeline
    print(f"Processed {len(list(dataset))} articles.")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()

