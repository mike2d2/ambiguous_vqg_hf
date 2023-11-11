

def remove_examples_by_qid(from_dataset, remove_dataset):
    print(f"Removing {len(remove_dataset)} items from {from_dataset.builder_name}...")
    matching_examples = from_dataset.filter(lambda example: str(example["question_id"]) not in remove_dataset["question_id"])
    
    print(f'Dataset now of size: {len(matching_examples)}')
    return matching_examples