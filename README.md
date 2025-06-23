# TGS Project

This repository contains tooling for text processing and modeling using spaCy, Solr, and NLP techniques.

## Demo
Here is a [URL](https://novel-talented-silkworm.ngrok-free.app) that you can use in order to view the results from the NLP processing. Please ask for the credentials to be able to access this.

# LLM Integration and Document Preparation Summary

All of this work was performed in anticipation of integrating LLMs for document retrieval, data analysis, and general query handling. The data was prepared in multiple formats to allow the LLM to examine different aspects of the content.

For instance, named entity recognition (NER), including temporal markers like dates, was extracted to enable temporal trend analysis — understanding who or what was discussed over time.

A total of **287,113 articles** were downloaded from a publicly available dataset. The processing pipeline included:

1. **Overall structural analysis** of the corpus  
2. **spaCy NER tagging** to assist with rapid identity resolution  
3. **FastCoref application** to resolve coreferences, improving clarity for topic segmentation  
4. **Sentence tokenization** on coreferenced text using spaCy, anticipating vectorization. This helps ensure pronouns and references are resolved before chunking or applying simpler models like Word2Vec.

While higher-order models (e.g., OpenAI's `text-embedding-3`) don't require coref resolution, it improves performance for traditional or cost-constrained vectorizers.

**Stop words were intentionally retained.** In modern transformer-based models, these words carry syntactic and semantic weight, unlike earlier NLP pipelines where they were excluded due to limited model capacity.

The dataset includes two sets of embeddings:

- **OpenAI Embeddings** – Raw text, chunk-level  
- **Word2Vec Embeddings** – Coreferenced text, sentence-level (stop words removed)

These can be used for **HNSW search in Solr** to efficiently retrieve semantically similar documents.

If time permits, I intend to complete a chatbot layer to showcase the utility of this structured pipeline.

## Project Structure

```
genderated_data/
    The default output directory from the various stages of the pipeline
notebooks/                  # Jupyter notebooks for exploration and prototyping
    document_eda.ipynb      # EDA notebook on documents
    tests.ipynb             # Notebook for testing components

scripts/                    # Standalone executable scripts
    tokenize_by_sentence.py # Tokenizes input documents by sentence

tests/                      # Python test scripts
    test_colbert_install.py
    test_get_data.py
    test_gpu_participation.py
    test_ner_article.py
    test_spacy_install.py

tgs_project/                # Main source code
├── backend/
│   └── initialize_solr.py  # Solr initialization utility
│   └── solr.py             # Solr utility functions
├── document_processing/
│   └── ner_extraction.py         # Named Entity Recognition logic
│   └── resolve_coreferences.py   # Coreference resolution
│   └── sentence_processing.py    # Sentence splitting pipeline
│   └── topic_modeling.py         # Topic modeling logic
│   └── word_2_vec.py             # Word2Vec training
├── etl/                   # Placeholder for ETL processes
├── frontend/              # Placeholder for UI or frontend code
├── pipeline/
│   └── pipeline.py        # Parallel/staged pipeline class
│   └── utils.py           # General utilities

.env                        # User provided environment variable definitions
.gitignore                  # Git ignore file
poetry.lock                 # Poetry lock file
pyproject.toml              # Poetry config with dependencies
README.md                   # This file
```

## Setup

Install dependencies:

```bash
poetry install
```

## Scripts

### Solr Initialization ( Indexing )
A script is provided to assist in loading (index) data into solr.
```bash
python -m scripts.initialize_solr
```

### tokenize_by_sentence
This script is used to do sentence tokenization. Due to how Windows handles the multiprocessing, this needed to be placed in its own script file.

```bash
python -m scripts.tokenize_by_sentence
```

## Notes

- All NLP components are modular and designed to work within a staged processing pipeline.
- `tgs_project.pipeline` provides support for multiprocessing and batch processing of data.

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

## Helper Script
Located in the scripts directory is a script to help with initializing a solr instance for querying. One would need to edit this file as needed for their particular needs; however, this provides an example and starting script to index data.
```python
python -m scripts.initialize_solr
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
