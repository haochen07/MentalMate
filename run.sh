#!/bin/bash
# run.sh - Simple script for running the training process

echo "========================================="
echo "Qwen3-8B MentalMate Model Fine-tuning"
echo "========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python not found. Please install Python 3.8+ before continuing."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if wandb is logged in
if ! wandb status &> /dev/null; then
    echo "Please log in to Weights & Biases:"
    wandb login
fi

# Run fine-tuning
echo "Starting fine-tuning process..."
python train.py

# Deactivate virtual environment
deactivate

echo "Training completed."