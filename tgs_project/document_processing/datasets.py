from typing import Any
import os

from datasets import load_dataset
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
