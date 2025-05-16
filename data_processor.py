# data_processor.py
"""
Data processing utilities for fine-tuning Qwen3-8B on mental therapy dataset.
"""
import logging
import re
import wandb

# Import shared configuration
from config import RANDOM_SEED

# Set up logging
logger = logging.getLogger("qwen3-finetune.data")

def parse_conversation_format(examples):
    """
    Parse the PHR mental therapy dataset format and convert to chatml format.
    
    Input format example:
    <s>[INST] <<SYS>>System instruction<</SYS>> User message [/INST] Assistant response </s>
    
    Output format:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]}
    """
    text = examples["text"]
    messages = []

    # Extract and append system prompt if exists
    system_prompt_match = re.search(r'<<SYS>>(.*?)<</SYS>', text, re.DOTALL)
    if system_prompt_match:
        system_prompt = system_prompt_match.group(1).strip()
        messages.append({"role": "system", "content": system_prompt})
        text = text.replace(system_prompt_match.group(0), '')

    # Regex to capture user-assistant pairs
    turns = re.findall(r'\[INST\](.*?)\[/INST\](.*?)(?:<\/s>|$)', text, re.DOTALL)

    for user_text, assistant_text in turns:
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()

        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
    
    return {"messages": messages}

def split_and_save_test_set(dataset, test_set_path):
    logger.info("Splitting dataset into train, validation, and test sets...")
    splits = dataset.train_test_split(
        test_size = 2000,
        shuffle = True, 
        seed = RANDOM_SEED
    )
    save_test_set(splits["test"], test_set_path)
    return splits["train"]

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