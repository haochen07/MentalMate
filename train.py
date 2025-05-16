# train.py
#!/usr/bin/env python
"""
Main training script for fine-tuning Qwen3-8B on the mental therapy dataset.
Uses Weights & Biases for experiment tracking.
"""

import os
import logging
import torch
import random
import numpy as np
from datetime import datetime
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
import wandb

# Import shared configuration
from config import BASE_MODEL_NAME, MAX_SEQ_LENGTH, OUTPUT_DIR, TEST_SET_PATH

# Import data processing utilities
from data_processor import load_and_process_dataset, save_test_set, log_example_conversations

# Constants used multiple times in this file
RANDOM_SEED = 42
LORA_RANK = 16
LORA_ALPHA = 16
PROJECT_NAME = "qwen3-therapy"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finetune.log")
    ]
)
logger = logging.getLogger("qwen3-finetune")

def set_seed(seed):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_wandb(config):
    """Initialize Weights & Biases tracking"""
    run_name = f"qwen3-therapy-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=PROJECT_NAME,
        config=config,
        name=run_name,
    )
    logger.info(f"WandB initialized with run name: {run_name}")
    return run_name

def get_available_device():
    """Returns the best available device for training"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            gpu_props = torch.cuda.get_device_properties(i)
            memory_mb = gpu_props.total_memory / (1024 * 1024)
            logger.info(f"GPU {i}: {gpu_props.name}, {memory_mb:.0f}MB memory")
        
        return "cuda"
    elif torch.backends.mps.is_available():
        logger.info("CUDA not available, using Apple MPS (Metal)")
        return "mps"
    else:
        logger.warning("No GPU found, using CPU. Training will be very slow!")
        return "cpu"

def count_model_parameters(model):
    """Count and log the number of parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Model has {trainable_params:,} trainable parameters ({trainable_params/total_params*100:.2f}%)")
    
    # Log to wandb
    wandb.log({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": trainable_params/total_params*100
    })
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": trainable_params/total_params*100
    }

def create_experiment_dir(base_dir, run_name):
    """Create a unique directory for this experiment"""
    exp_dir = os.path.join(base_dir, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir

def log_artifacts(model_dir):
    """Log model artifacts to wandb"""
    # Log the model as an artifact
    model_artifact = wandb.Artifact(
        name=f"qwen3-therapy-model",
        type="model",
        description="Fine-tuned Qwen3-8B model for mental therapy"
    )
    model_artifact.add_dir(model_dir)
    wandb.log_artifact(model_artifact)

def main():
    # Record start time
    start_time = datetime.now()
    logger.info(f"Starting fine-tuning at {start_time}")
    
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Initialize wandb with config
    wandb_config = {
        "model_name": BASE_MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "max_seq_length": MAX_SEQ_LENGTH,
        "random_seed": RANDOM_SEED,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "grad_accumulation_steps": 4,
        "epochs": 3,
        "load_in_4bit": True,
        "use_gradient_checkpointing": True,
    }
    run_name = init_wandb(wandb_config)
    
    # Get available device
    device = get_available_device()
    wandb.log({"device": device})
    
    # Create experiment directory
    exp_dir = create_experiment_dir(OUTPUT_DIR, run_name)
    
    # Load and process dataset
    processed_datasets = load_and_process_dataset(random_seed=RANDOM_SEED)
    
    # Log example conversations
    log_example_conversations(processed_datasets["test"])
    
    # Save the test set
    test_set_path = os.path.join(exp_dir, "test_set")
    save_test_set(processed_datasets["test"], test_set_path)
    
    # Load the Qwen3-8B model with Unsloth
    logger.info(f"Loading {BASE_MODEL_NAME} model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect the best dtype
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    # Count model parameters
    count_model_parameters(model)
    
    # Target modules for LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Enable wandb to watch model parameters and gradients
    wandb.watch(model, log="all", log_freq=10)
    
    # Configure model for LoRA fine-tuning
    logger.info(f"Applying LoRA configuration (rank={LORA_RANK})...")
    model = FastModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=target_modules,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,  # Optimized in Unsloth
        bias="none",  # Optimized in Unsloth
        use_gradient_checkpointing=True,
        random_state=RANDOM_SEED,
    )
    
    # Create output path for model checkpoints
    model_output_dir = os.path.join(exp_dir, "model")
    
    # Configure SFT training arguments
    logger.info("Configuring training arguments...")
    training_args = SFTConfig(
        output_dir=model_output_dir,
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_ratio=0.1,
        # Use the appropriate precision based on GPU capabilities
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=RANDOM_SEED,
        report_to="wandb",  # Use wandb instead of tensorboard
        load_best_model_at_end=True,
        packing=False,
        dataset_num_proc=4,
    )
    
    # Initialize the SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        args=training_args,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    
    # Log training stats
    wandb.log({
        "final_loss": trainer_stats.training_loss,
        "training_steps": trainer_stats.global_step
    })
    
    # After training, prepare model for inference
    logger.info("Preparing model for inference...")
    FastModel.for_inference(model)
    
    # Save the model (LoRA adapter only)
    adapter_path = os.path.join(model_output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"LoRA adapter saved to: {adapter_path}")
    
    # Save the merged model
    merged_path = os.path.join(model_output_dir, "merged")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    logger.info(f"Merged model saved to: {merged_path}")
    
    # Log artifacts to wandb
    log_artifacts(model_output_dir)
    
    # Calculate and log training time
    end_time = datetime.now()
    training_duration = end_time - start_time
    logger.info(f"Training completed at {end_time}")
    logger.info(f"Total training time: {training_duration}")
    
    # Log final stats to wandb
    wandb.log({
        "training_duration_seconds": training_duration.total_seconds(),
        "training_end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Finish wandb run
    wandb.finish()
    
    return model_output_dir

if __name__ == "__main__":
    try:
        output_dir = main()
        logger.info(f"Training completed successfully. Model saved to {output_dir}")
    except Exception as e:
        # Log exception to wandb
        if wandb.run is not None:
            wandb.log({"error": str(e)})
            wandb.finish(exit_code=1)
        logger.exception(f"Error during training: {e}")
        raise