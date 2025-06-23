import os
import json
from typing import Union
from io import TextIOBase
from datasets import load_dataset
from typing import Any
import time
from typing import Iterator, Any, Dict


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