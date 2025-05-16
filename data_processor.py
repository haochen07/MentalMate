# data_processor.py
"""
Data processing utilities for fine-tuning Qwen3-8B on mental therapy dataset.
"""
import logging
import re
import random
from datasets import load_dataset
import wandb

# Import shared configuration
from config import DATASET_NAME, TEST_SET_PATH

# Set up logging
logger = logging.getLogger("qwen3-finetune.data")

def parse_conversation_format(text):
    """
    Parse the PHR mental therapy dataset format and convert to chatml format.
    
    Input format example:
    <s>[INST] <<SYS>>System instruction<</SYS>> User message [/INST] Assistant response </s>
    
    Output format:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]}
    """
    # Remove <s> and </s> tags
    text = re.sub(r'<s>|</s>', '', text)
    
    # Extract system instruction
    sys_match = re.search(r'\[INST\]\s*<<SYS>>(.*?)<</SYS>>', text, re.DOTALL)
    system_instruction = sys_match.group(1).strip() if sys_match else ""
    
    # Remove system instruction from text for further processing
    if sys_match:
        text = text.replace(sys_match.group(0), "[INST]")
    
    # Split into turns by [INST] and [/INST]
    turns = re.split(r'\[INST\]|\[/INST\]', text)
    turns = [turn.strip() for turn in turns if turn.strip()]
    
    # Prepare messages in Qwen3's chatml format
    messages = []
    
    # Add system message if it exists
    if system_instruction:
        messages.append({
            "role": "system",
            "content": system_instruction
        })
    
    # Add user and assistant messages alternately
    for i in range(0, len(turns), 2):
        if i < len(turns):
            messages.append({
                "role": "user",
                "content": turns[i]
            })
        
        if i + 1 < len(turns):
            messages.append({
                "role": "assistant",
                "content": turns[i + 1]
            })
    
    return {"messages": messages}

def load_and_process_dataset(random_seed=42, num_proc=4):
    """
    Load and process the dataset.
    
    Args:
        random_seed: Random seed for reproducibility
        num_proc: Number of processes for parallel processing
        
    Returns:
        Dictionary with train, validation, and test splits
    """
    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    
    # Apply the formatting function
    logger.info("Processing dataset...")
    processed_dataset = dataset["train"].map(
        lambda example: parse_conversation_format(example["text"]),
        remove_columns=["text"],
        num_proc=num_proc
    )
    
    # Split into train, validation, and test sets
    logger.info("Splitting dataset into train, validation, and test sets...")
    processed_dataset = processed_dataset.shuffle(seed=random_seed)
    
    # Save examples for test set (2000 max)
    test_size = min(2000, len(processed_dataset))
    
    # Use 10% of remaining data for validation (max 2000)
    remaining = len(processed_dataset) - test_size
    val_size = min(int(0.1 * remaining), 2000)
    
    # Create the splits
    splits = processed_dataset.train_test_split(
        test_size=test_size, 
        shuffle=True, 
        seed=random_seed
    )
    train_val = splits["train"].train_test_split(
        test_size=val_size, 
        shuffle=True, 
        seed=random_seed
    )
    
    result = {
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": splits["test"]
    }
    
    # Log dataset split information
    logger.info(f"Dataset splits created:")
    logger.info(f"  Training set: {len(result['train'])} examples")
    logger.info(f"  Validation set: {len(result['validation'])} examples")
    logger.info(f"  Test set: {len(result['test'])} examples")
    
    # Log dataset sizes to wandb
    if wandb.run is not None:
        wandb.log({
            "train_size": len(result["train"]),
            "val_size": len(result["validation"]),
            "test_size": len(result["test"])
        })
    
    return result

def save_test_set(test_dataset, output_path):
    """
    Save the test set to disk
    
    Args:
        test_dataset: The test dataset to save
        output_path: Path where to save the test set
    """
    test_dataset.save_to_disk(output_path)
    logger.info(f"Test set saved to '{output_path}'")

def log_example_conversations(dataset, num_examples=3):
    """
    Log a few example conversations to wandb for reference
    
    Args:
        dataset: Dataset containing conversations
        num_examples: Number of examples to log
    """
    if wandb.run is None:
        return
        
    examples = []
    
    for i, example in enumerate(dataset[:num_examples]):
        messages = example["messages"]
        conversation = []
        
        for msg in messages:
            conversation.append(f"{msg['role'].upper()}: {msg['content'][:100]}...")
        
        examples.append("<br>".join(conversation))
    
    # Log to wandb as a Table
    example_table = wandb.Table(columns=["Example Conversation"])
    for example in examples:
        example_table.add_data(example)
    
    wandb.log({"example_conversations": example_table})