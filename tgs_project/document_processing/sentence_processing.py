import multiprocessing as mp
from tgs_project.utils import write_jsonl, read_jsonl
from tgs_project.pipeline.pipeline import Parallel, Stage, DataBatch
import os
from tqdm import tqdm

# ── spaCy worker init (called once per process) ───────────────────────────────
def _init_worker():
    global nlp
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

# ── pipeline stages ───────────────────────────────────────────────────────────
def _ensure_dict(x):
    return {"resolved_text":x} if isinstance(x,str) else x

def _tokenize(rec):
    global nlp
    d=_ensure_dict(rec)
    d["doc"]=nlp(d["resolved_text"]) # type: ignore
    return d
tokenize = Stage(_tokenize)

def _sentences(rec):
    rec["sentences"] = [sent.text for sent in rec["doc"].sents]
    return rec
count_sentences = Stage(_sentences)

def _drop_doc(rec):
    rec.pop("doc", None)
    return rec
drop_doc = Stage(_drop_doc)



class SentenceTokenizerPipeline(Stage):
    """
    A pipeline stage that tokenizes text into sentences using spaCy's sentencizer.

    This class is designed for use in a composable data processing pipeline and supports
    both serial and parallel execution via Python's multiprocessing. Each input record
    is expected to contain an "article" field (or be a raw string), and the pipeline adds
    a "sentences" field containing the list of sentence spans.

    Attributes:
        n_workers (int): Number of parallel worker processes to use. If set to 1,
            processing runs serially. If set to 0 or -1, uses all available CPU cores.
        results_path (str): Path to the output JSONL file for storing processed results.
        batch_size (int): Number of records to process per batch.
        flush_interval (int): Number of processed records to buffer before writing to disk.
        chunksize (int): Chunk size used when distributing data to workers in parallel mode.

    Example usage:
        >>> from sentence_pipeline import SentenceTokenizerPipeline
        >>> data = [{"id": "001", "article": "This is a test. Another sentence."}]
        >>> pipeline = SentenceTokenizerPipeline(n_workers=4)
        >>> list(data | pipeline)

    Running as a standalone script:
        This module optionally defines a `main()` function guarded by
        `if __name__ == "__main__"` to allow safe execution in multiprocessing
        contexts (especially on Windows). This ensures that the pipeline can be run
        directly from the command line without triggering recursive subprocess execution.

        Example:
            if __name__ == "__main__":
                import multiprocessing as mp
                mp.freeze_support()
                main()

    Note:
        Each parallel worker initializes its own spaCy `nlp` pipeline using a
        global variable and a process initializer to avoid pickling issues.
    """
    def __init__(
        self,
        n_workers: int = 1,
        results_path: str = "sentence_tokenized_docs.jsonl",
        batch_size: int = 256,
        flush_interval: int = 512,
        chunksize: int = 8,
    ):
        """
        Initialize the SentenceTokenizerPipeline.
        Args:
            n_workers (int): Number of worker processes to use. If None, uses the number of CPU cores.
        """
        self.n_workers = n_workers if n_workers > 0 else mp.cpu_count()
        self.results_path = results_path
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.chunksize = chunksize

        # Set up the pipeline
        self.pipeline = (
            tokenize |
            count_sentences |
            drop_doc
        )

        # If we have more than one worker, we need to parallelize the pipeline
        # Less than 1 means use all available cores
        # If n_workers is 1, we don't need to parallelize
        # If > 1, we need to parallelize
        if self.n_workers != 1: 
            self.pipeline = Parallel(
                self.pipeline,
                n_workers=self.n_workers,
                chunksize=self.chunksize,
                initializer=_init_worker,
            )
        else:
            _init_worker()  # Initialize the worker for the main process

    def __ror__(self, texts):
        """
        Processes a list of article records to extract sentences using a spaCy-based pipeline.
        
        Args:
            texts (Iterable[dict]): A list or iterable of dictionaries, each containing at least
                an "id" and an "article" field. Records already present in the results file will
                be skipped based on their "id".

        Side Effects:
            Writes processed records to the specified JSONL file in `self.results_path`.
            Each processed record will include a "sentences" field with a list of sentence strings.

        Returns:
            Generator[dict, None, None]: A generator of processed article dictionaries, streamed
            from the output JSONL file. Each dictionary includes the original fields and a new
            "sentences" field.
        
        Notes:
            This function is designed for batch processing and will flush to disk every
            `flush_interval` records to avoid high memory usage. The file is appended to if it exists,
            and previously processed IDs are automatically skipped.
        """
        # Check if results already exist
        if not isinstance(texts, DataBatch):
            raise ValueError("Input must be a DataBatch object.")
        print(f"Processing {texts.count} articles...")
        n_texts = texts.count if texts.count is not None else 0
        n_batches = n_texts // self.batch_size

        with open(self.results_path, "a") as f:
            # Process the articles in batches
            results = []
            for batch in tqdm(texts.batched(self.batch_size), total=n_batches, desc="Processing"):
                results.extend(batch | self.pipeline)
                if len(results) >= self.flush_interval:
                    write_jsonl(f, results)
                    results = []
            if results:
                # Write the remaining results to the file
                write_jsonl(f, results)
        return read_jsonl(self.results_path) # This will return the results as a list of dictionaries

