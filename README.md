# Fine-tuning Qwen3-8B for Mental Therapy Conversations

This repository contains code for fine-tuning the Qwen3-8B model on the [Mental Therapy Dataset](https://huggingface.co/datasets/vibhorag101/phr_mental_therapy_dataset) using Hugging Face's SFT Trainer and Unsloth for acceleration. Training metrics are tracked using Weights & Biases (WandB).

## Repository Structure

- `train.py` - Main script for model fine-tuning with WandB integration
- `data_processor.py` - Data processing utilities for dataset preparation
- `config.py` - Shared configuration values used across scripts
- `requirements.txt` - Required dependencies
- `run.sh` - Bash script to run the training pipeline

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/user/qwen3-therapy-finetuning.git
   cd qwen3-therapy-finetuning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up a Weights & Biases account:
   - Sign up at [wandb.ai](https://wandb.ai) if you don't have an account
   - Run `wandb login` in your terminal and enter your API key
   - Your API key can be found in your WandB account settings

4. Make sure you have enough VRAM for fine-tuning (at least 16GB is recommended).

## Dataset

The dataset used is the [PHR Mental Therapy Dataset](https://huggingface.co/datasets/vibhorag101/phr_mental_therapy_dataset), which contains multi-turn therapy conversations.

- Each conversation consists of multiple turns between a user (patient) and an assistant (therapist)
- The script automatically processes the dataset format, which includes system instructions
- The dataset is split into train, validation, and test sets (with 2000 samples reserved for testing)

### Data Processing

The `data_processor.py` module handles:
- Loading the dataset from Hugging Face
- Parsing the conversation format and converting to chatml format
- Splitting the dataset into train, validation, and test sets
- Logging example conversations to WandB
- Saving the test set for future use

## Configuration

Shared configuration values are stored in `config.py`. Script-specific settings are defined within the individual scripts. The main configuration values include:

```python
# From config.py
BASE_MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./qwen3-8b-therapy-finetuned"
TEST_SET_PATH = "therapy_test_set"
DATASET_NAME = "vibhorag101/phr_mental_therapy_dataset"
```

## Usage

To fine-tune the model, simply run:

```bash
python train.py
```

Alternatively, use the provided shell script:

```bash
bash run.sh
```

The training script:
- Processes the dataset into the correct format
- Applies LoRA fine-tuning with rank 16
- Uses 4-bit quantization to minimize memory usage
- Saves both the LoRA adapter and merged model
- Logs metrics and artifacts to Weights & Biases

## Monitoring Training Progress

### Weights & Biases Dashboard

The training script logs metrics and artifacts to Weights & Biases. To view the training and validation losses, as well as other metrics:

1. After starting training, you'll see a link in the console output like:
   ```
   View run at: https://wandb.ai/username/qwen3-therapy/runs/run_id
   ```

2. Click on the link to access your WandB dashboard, where you can see:
   - Real-time training and validation loss curves
   - GPU/CPU usage and memory statistics
   - Model parameters and gradients
   - Example conversations from the dataset
   - Logged model artifacts

3. The dashboard allows you to:
   - Compare different runs
   - Create custom visualizations
   - Share results with team members
   - Analyze system performance during training

### WandB Features Used in This Project

- **Run Tracking**: Automatic tracking of training and validation metrics
- **Model Watching**: Monitoring parameter and gradient distributions
- **Artifact Logging**: Storing model checkpoints as artifacts
- **Dataset Visualization**: Example conversations displayed for reference
- **System Monitoring**: Tracking hardware usage during training

## Implementation Details

- **Model**: Qwen3-8B from Alibaba Cloud
- **Quantization**: 4-bit quantization to reduce memory usage
- **Optimization**: LoRA fine-tuning with rank 16
- **Training Framework**: TRL's SFT Trainer with Unsloth acceleration
- **Thinking Mode**: Disabled as per requirements
- **Hyperparameters**:
  - Learning rate: 2e-4
  - Batch size: 2 (per device)
  - Gradient accumulation steps: 4
  - Epochs: 3
  - Warmup ratio: 0.1
  - Optimizer: AdamW 8-bit
  - Weight decay: 0.01
  - Max sequence length: 2048

## Output

The training script saves:
1. **LoRA adapter** - Small file containing just the fine-tuned parameters
2. **Merged model** - Complete model with fine-tuned parameters merged into the base model

These files are saved in run-specific directories:
```
qwen3-8b-therapy-finetuned/
└── qwen3-therapy-YYYYMMDD_HHMMSS/
    ├── model/
    │   ├── lora_adapter/  # LoRA parameters only
    │   └── merged/        # Complete fine-tuned model
    └── test_set/          # Saved test set
```

Additionally, all artifacts are logged to Weights & Biases for easy access and version tracking.

## License

This project code is provided for educational purposes. Use of the Qwen3-8B model is subject to its original license.
