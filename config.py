# config.py
"""
Shared configuration constants for fine-tuning Qwen3-8B on mental therapy dataset.
Contains values that need to be consistent across different scripts.
"""

# Model information
BASE_MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SEQ_LENGTH = 2048

# Paths
OUTPUT_DIR = "./qwen3-8b-therapy-finetuned"
TEST_SET_PATH = "therapy_test_set"

# Dataset information
DATASET_NAME = "vibhorag101/phr_mental_therapy_dataset"