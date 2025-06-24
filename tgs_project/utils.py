import os
import json
from typing import Union
from io import TextIOBase
from datasets import load_dataset
from typing import Any
import time
from typing import Iterator, Any
import spacy
from nltk.corpus import stopwords

# Load blank English model — super fast
spacy_nlp_for_tokenizing = spacy.blank("en")
DEFAULT_MERGE_SET = set([
    "'s",
    "’s",
    "n't",
    "’nt",
    "’t",
    "'t",
    "'ll",
    "’ll",
    "'re",
    "’re",
    "'ve",
    "’ve",
    "'d",
    "’d",
    "'m",
    "’m"
])
stopwords_set = set(stopwords.words("english"))


try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("dotenv not installed, skipping environment variable loading.")

def load_cnn_dailymail()-> Any:
    if not os.getenv("DATASET_NAME"):
        print("DATASET_NAME environment variable not set, using default 'cnn_dailymail'")
    if not os.getenv("DATASET_CONFIG_NAME"):
        print("DATASET_CONFIG_NAME environment variable not set, using default '3.0.0'")
    dataset_name = os.getenv("DATASET_NAME", "cnn_dailymail")
    dataset_config_name = os.getenv("DATASET_CONFIG_NAME", "3.0.0")
    # Load 1% of the training data in the dataset
    # for a quick test
    dataset: Any = load_dataset(dataset_name, dataset_config_name)
    return dataset

def de_dupe_jsonl(file_name: str) -> None:
    """
    De-duplicates a JSONL file based on the 'id' field.
    """
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            processed = [json.loads(line) for line in f]
    else:
        raise FileNotFoundError(f"File {file_name} not found. Create a blank file at least.")
    
    de_duped = []
    de_duped_ids = set()
    for item in processed:
        if item["id"] not in de_duped_ids:
            de_duped.append(item)
            de_duped_ids.add(item["id"])

    output_file = f"{file_name}_de_duped.jsonl"
    with open(output_file, "w") as f:
        for item in de_duped:
            f.write(json.dumps(item) + "\n")

def write_jsonl(
    file_name: Union[str, TextIOBase],
    data: list,
    mode: str='a'
) -> None:
    """
    Write a list of dictionaries to a JSONL file.
    """
    close_file = False
    if isinstance(file_name, str):
        close_file = True
        f = open(file_name, mode)
    elif not isinstance(file_name, TextIOBase):
        raise ValueError("file_name must be a string or a TextIO object.")
    else:
        f = file_name
    for item in data:
        f.write(json.dumps(item) + "\n")
    
    # Ensure the file is flushed and synced to disk
    # This is important for large files to ensure data integrity
    # and to avoid data loss in case of a crash.
    if close_file:
        f.close()
    else:
        f.flush()
        os.fsync(f.fileno())
    print(f"Written {len(data)} items to {f.name}") # type: ignore
    

def read_jsonl(
    file_name: str,
    create: bool = False
) -> Iterator[dict[str, Any]]:
    """
    Read a JSONL file and return a list of dictionaries.
    """
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            for line in f:
                yield json.loads(line)
    else:
        if create:
            with open(file_name, "w") as f:
                pass
        else:
            raise FileNotFoundError(f"File {file_name} not found. Create a blank file at least.")
        
def get_jsonl_n_count(file_name: str) -> int:
    """
    Get the number of lines in a JSONL file.
    """
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return sum(1 for _ in f.readlines())
    else:
        raise FileNotFoundError(f"File {file_name} not found.")

def read_and_follow_jsonl(path: str, delay: float = 0.5) -> Iterator[dict]:
    """Yield existing JSONL records, then follow new ones as they're appended."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

        while True:
            line = f.readline()
            if line:
                yield json.loads(line)
            else:
                time.sleep(delay)

def tokenize_merge_phrases(
    text: str,
    merge_tokens: set[str] = DEFAULT_MERGE_SET,
    remove_stop_words: bool = False,
    remove_punct: bool = True,
    merge_punct: bool = False
) -> list[str]:
    """
    Tokenize the text and merge specified tokens.
    Args:
        text (str): The input text to process.
        merge_tokens (set[str]): A set of tokens to merge.
    """
    if remove_punct and merge_punct:
        raise ValueError("remove_punct and merge_punct cannot be both True.")
    tokens = []
    doc = spacy_nlp_for_tokenizing(text)
    i = 0
    while i < len(doc):
        token = doc[i]
        if (
            (
                token.text in merge_tokens or
                merge_punct and token.is_punct
            ) and tokens
        ):
            tokens[-1] = tokens[-1] + token.text
        else:
            if token.is_punct and remove_punct:
                pass
            if remove_stop_words and token.text.lower() in stopwords_set:
                pass
            else:
                tokens.append(token.text)
        i += 1
    return tokens

def context_windows(text: str, ner_token_list: list[str], window: int = 5) -> Iterator[str]:
    """
    Yield context windows around tokens in the text that are in the ner_token_list.
    Args:
        text (str): The input text to process.
        ner_token_list (list[str]): A list of named entity tokens to look for.
        window (int): The size of the context window to yield.
    """
    tokens = tokenize_merge_phrases(text, set(["'s", "’s"]))
    ner_token_set = set([_.lower() for _ in ner_token_list])
    for i, token in enumerate(tokens):
        if token.lower() in ner_token_set:
            context = tokens[max(0, i - window): i + window + 1]
            yield " ".join(context)
