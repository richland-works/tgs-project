import unittest
import torch

class TestGPUAvailability(unittest.TestCase):
    def test_gpu_available(self):
        self.assertTrue(torch.cuda.is_available(), "No GPU found. ColBERTv2 will be very slow without one.")
