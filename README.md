# tgs-project
A Basic NLP project
## Project Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Install Poetry

```bash
pip install poetry
poetry install
```
## Solr Setup (Docker)
### In a Windows Environment
1. Enable WSL if you're on Windows:
```bash
wsl --install
```
2. Follow [this guide](https://gist.github.com/dehsilvadeveloper/c3bdf0f4cdcc5c177e2fe9be671820c7) to install Docker.
3. PUll the latest Solr image
```bash
docker pull solr:latest
```
4. Create a persistent Solr data volume:
```bash
docker volume create solrdata
```
5. Start Solr
```bash
docker run -p 8983:8983 -v solrdata:/var/solr solr:latest
```
Solr should now be accessible at [http://localhost:8983](http://localhost:8983)

## Environmental Variables
There are several environmental variables that are used to assist in accessing OpenAI LLMs and the solr instance. Please make sure the following are setup:
```bash
OPENAI_KEY=<your key>
DATASET_NAME='cnn_dailymail'
DATASET_CONFIG_NAME='3.0.0'
SOLR_URL='http://localhost:8983/solr'
```

# ArticleEntityProcessor

`ArticleEntityProcessor` is a Python class for efficiently extracting named entities from a dataset of articles using spaCy's transformer-based pipeline. It supports GPU acceleration, incremental flushing to disk, and skipping already-processed articles based on prior results.

## Features

- GPU support with `spacy.require_gpu()`
- Batch processing for performance
- Automatic skipping of already-processed articles
- Flushes results incrementally to avoid memory overload
- Reads/writes using `read_jsonl` and `write_jsonl` utilities
- Configurable via environment variables (`NER_RESULTS_PATH`)

## Requirements

- Python 3.8+
- `spacy`
- `tqdm`
- `python-dotenv` (optional)
- A spaCy transformer model, e.g., `en_core_web_trf`
```bash
pip install spacy tqdm python-dotenv
python -m spacy download en_core_web_trf
```

## Environment Variables
NER_RESULTS_PATH=ner_output.jsonl

## Usage
```python
from tgs_project.ner_processor import ArticleEntityProcessor
from datasets import load_dataset  # or your own dataset

articles = load_dataset("cnn_dailymail", "3.0.0")["train"]

processor = ArticleEntityProcessor()
processor.process(articles)
```
## Input Format
```python
{
    "id": "unique-id",
    "article": "Full article text...",
    "highlights": "Summary highlights..."
}
```
## Output Format
The output is written in JSON lines (.jsonl) format
```python
{
  "id": "unique-id",
  "entities": [["Marie Curie", "PERSON"], ["Paris", "GPE"]],
  "highlights": "Nobel-winning physicist...",
  "article": "Marie Curie lived in Paris and..."
}
```
# CorefResolver

`CorefResolver` is a utility class that performs coreference resolution on a dataset of articles using the `fastcoref` spaCy component. It resolves pronouns and other references to improve downstream NLP tasks like summarization or entity extraction.

## Features

- GPU support for high-throughput processing
- Coreference resolution via `fastcoref` and spaCy
- Incremental JSONL output with configurable flush interval
- Skips already-processed articles based on output file
- Reads environment variables via `dotenv` if available

## Requirements

- Python 3.8+
- `spacy`
- `fastcoref`
- `tqdm`
- `python-dotenv` (optional)
- Custom utilities: `write_jsonl`, `read_jsonl`

Install requirements:

```bash
pip install spacy tqdm fastcoref python-dotenv
python -m spacy download en_core_web_trf
```
## Environment Variables
```bash
COREF_RESULTS_PATH=coref_output.jsonl
```
## Usage
```python
from tgs_project.coref_resolver import CorefResolver
from tgs_project.utils import load_cnn_dailymail

dataset = load_cnn_dailymail()
articles = dataset["train"]

resolver = CorefResolver()
resolver.process(articles)
```
## Input Format
```python
{
    "id": "unique-article-id",
    "article": "Text of the article...",
    "highlights": "Summary of the article..."
}
```
### Ouput Format
The output is written in JSON lines (.jsonl) format
```python
{
  "id": "unique-article-id",
  "resolved_text": "Text with resolved coreferences...",
  "article": "Original article text...",
  "highlights": "Summary of the article..."
}
```
# SentenceTokenizerPipeline

This module defines a sentence tokenization pipeline for processing large sets of documents in parallel using [spaCy](https://spacy.io/), with multiprocessing and batch writing to disk in JSONL format.

IMPORTANT WARNING:

Due to how Python's `multiprocessing` works, this module should not be executed directly unless wrapped in an `if __name__ == "__main__":` block.

Failure to do so may result in:
- Pickling errors
- Recursive subprocess spawning (especially on Windows)
- Unpredictable behavior when using multiprocessing pools

To safely run the pipeline:

    def main():
        ...

    if __name__ == "__main__":
        import multiprocessing as mp
        mp.freeze_support()
        main()


## Features

- Tokenizes input text into sentences using spaCy's sentencizer
- Parallel processing via Python `multiprocessing`
- Efficient batch handling and buffered disk writes
- Skips already-processed data using `id` field tracking
- Composable with other pipeline stages

## Installation

Install the required dependencies:

```bash
pip install spacy tqdm
python -m spacy download en_core_web_sm
```

You also need your custom modules accessible:

- `tgs_project.pipeline.pipeline`
- `tgs_project.utils`

Ensure they're in your PYTHONPATH or part of your package.

## Usage

```python
from sentence_pipeline import SentenceTokenizerPipeline
from tgs_project.pipeline.pipeline import DataBatch

data = [{"id": "001", "article": "This is a test. Another sentence."}]
batch = DataBatch(data, count=len(data))

# Use the default of 1 worker, otherwise, 
# it needs to be encapsulated in a "main"
pipeline = SentenceTokenizerPipeline()
results = list(batch | pipeline)
```

## Command-Line Execution

```python
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
```

## Parameters

| Parameter       | Type   | Description                                                                 |
|----------------|--------|-----------------------------------------------------------------------------|
| `n_workers`     | int    | Number of processes (0 or -1 uses all cores)                                |
| `results_path`  | str    | Path to store output JSONL                                                  |
| `batch_size`    | int    | Number of records per batch                                                 |
| `flush_interval`| int    | Write to disk after this many processed records                             |
| `chunksize`     | int    | Chunk size passed to the multiprocessing pool                               |

## Output Format

Each output record is a JSON object:

```json
{
  "id": "001",
  "article": "This is a test. Another sentence.",
  "sentences": ["This is a test.", "Another sentence."]
}
```

## Notes

- Only `article` and `id` fields are required for input
- Each parallel worker initializes its own spaCy pipeline to avoid pickling issues
- Processing is streamed to file, not held in memory
