import unittest
import os
from datasets import load_dataset # type: ignore
from dotenv import load_dotenv
from typing import Any

load_dotenv(override=True)

class TestCNNDataLoading(unittest.TestCase):
    def test_load_cnn_dailymail(self):
        """Test loading the CNN/DailyMail dataset."""
        if not os.getenv("DATASET_NAME"):
            print("DATASET_NAME environment variable not set, using default 'cnn_dailymail'")
        if not os.getenv("DATASET_CONFIG_NAME"):
            print("DATASET_CONFIG_NAME environment variable not set, using default '3.0.0'")
        dataset_name = os.getenv("DATASET_NAME", "cnn_dailymail")
        dataset_config_name = os.getenv("DATASET_CONFIG_NAME", "3.0.0")
        # Load 1% of the training data in the dataset
        # for a quick test
        try:
            dataset: Any = load_dataset(dataset_name, dataset_config_name, split="train[:1%]")
            self.assertGreater(len(dataset), 0, "Dataset should not be empty")
            self.assertIn("article", dataset.column_names, "Missing 'article' column")
            self.assertIn("highlights", dataset.column_names, "Missing 'highlights' column")
        except Exception as e:
            self.fail(f"Failed to load CNN/DailyMail dataset: {e}")
