

def remove_examples_by_qid(from_dataset, remove_dataset):
    
    for row in from_dataset:
        matching_examples = remove_dataset.filter(lambda example: str(example["question_id"]) == str(row["question_id"]))

        if len(matching_examples) > 0:

            pass