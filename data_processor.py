# data_processor.py
"""
Data processing utilities for fine-tuning Qwen3-8B on mental therapy dataset.
"""
import logging
import re
import wandb

# Import shared configuration
from config import RANDOM_SEED, TEST_SET_PATH

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

def split_and_save_test_set(dataset):
    splits = dataset.train_test_split(
        test_size = 2000,
        shuffle = True, 
        seed = RANDOM_SEED
    )

    train_eval_set = splits["train"]
    test_set = splits["test"]

    test_set.save_to_disk(TEST_SET_PATH)
    logger.info(f"Test set saved to '{TEST_SET_PATH}'")
    return train_eval_set
