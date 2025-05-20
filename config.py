# config.py
"""
Shared configuration constants for fine-tuning Qwen3-8B on mental therapy dataset.
Contains values that need to be consistent across different scripts.
"""

# Model information
BASE_MODEL_NAME = "unsloth/Qwen3-8B"
MAX_SEQ_LENGTH = 2048

# Dataset information
DATASET_NAME = "vibhorag101/phr_mental_therapy_dataset"

# Paths
OUTPUT_DIR = "./mentalmate-qwen3-8b-finetuned"
TEST_SET_PATH = OUTPUT_DIR + "/test-set"

# Random seed for reproductivity
RANDOM_SEED = 42