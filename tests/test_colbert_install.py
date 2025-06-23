import warnings

# Suppress PyTorch deprecation warning early—before importing transformers
warnings.filterwarnings(
    "ignore",
    message="torch.utils._pytree._register_pytree_node is deprecated.*",
    category=UserWarning,
    module="transformers"
)

import unittest
from colbert.infra.run import Run, RunConfig

class TestColBERTSetup(unittest.TestCase):
    def test_colbert_run_context(self):
        try:
            runconfig = RunConfig(experiment="test")
            with Run().context(runconfig):
                self.assertTrue(True, "ColBERT context initialized successfully")
        except Exception as e:
            self.fail(f"ColBERT failed to initialize: {e}")
