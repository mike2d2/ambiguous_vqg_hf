import os
from tqdm import tqdm
from datasets import load_dataset, Dataset, load_from_disk
from transformers import ViltConfig, AutoTokenizer
from PIL import Image

class ViltFinetuneDataLoader(object):
    def __init__(self, config) -> None:
        self.config = config
        pass

    @staticmethod
    def from_saved_or_load(split='train',config=None, verbose=False, force_reload=False, save_dir='saved_datasets/', truncate_len=None):
        print(f"Split is {split}")
        if not force_reload:
            if os.path.exists(save_dir):
                print('loading dataset from disk...')
                reloaded_encoded_dataset = load_from_disk(save_dir)
                print(f'loaded dataset from disk with {len(reloaded_encoded_dataset)} examples')
                return reloaded_encoded_dataset
            else:
                print(f'Cannot find saved datasets directory at: {save_dir}')
        loader = ViltFinetuneDataLoader(config)

        if truncate_len is None:
            dataset = load_dataset("HuggingFaceM4/VQAv2", split=f'{split}')
        else:
            dataset = load_dataset("HuggingFaceM4/VQAv2", split=f'{split}[0:{truncate_len}]')
        print('------------------------')
        print(f'loaded huggingface dataset with {len(dataset)} examples')
        # load ambiguity dataset
        # jsonObj = pd.read_json(path_or_buf=file_path, lines=True)
        ambiguous_vqa_dataset = load_dataset('json', data_files=f'datasets/combined_dev_test_from_dataset.json', split='train')
        
        # find first example
        # q_id_check = ambiguous_vqa_dataset[0]['Input.question_id']
        # for q in tqdm(dataset):
        #     if q_id_check == q['question_id']:
        #         print('found')

        # Remove examples from ambiguous dataset
        dataset = loader.isolate_examples_by_qid(dataset, ambiguous_vqa_dataset, remove=True)
        if verbose:
            print(dataset[0])
            image = Image.open(dataset[0]['image'].filename)

        dataset = dataset.map(loader.add_labels)

        dataset.save_to_disk(save_dir)
        return dataset
    
    # vocab selection found here: https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/Using_ViLT_for_image_text_retrieval.ipynb#scrollTo=4qC5r7ZEUAcr
    def get_score(self, count: int) -> float:
        return min(1.0, count / 3)

    def add_labels(self, example):
        annotation = example
        answers = annotation['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer["answer"]
            answer_count[answer_] = answer_count.get(answer_, 0) + 1
        labels = []
        scores = []
        for answer in answer_count:
            if answer not in list(self.config.label2id.keys()):
                continue
            labels.append(self.config.label2id[answer])
            score = self.get_score(answer_count[answer])
            scores.append(score)
        annotation['labels'] = labels
        annotation['scores'] = scores
        return example

    def isolate_examples_by_qid(self, from_dataset, remove_dataset, remove):
        print(f"Removing {len(remove_dataset)} items from {from_dataset.builder_name}...")
        # matching_examples = from_dataset.filter(lambda example: example["question_id"] not in remove_dataset["Input.question_id"])
        q_ids_dict = {key: None for key in remove_dataset["Input.question_id"]}
        if remove:
            matching_examples = from_dataset.filter(lambda example: example["question_id"] not in q_ids_dict)
        else:
            matching_examples = from_dataset.filter(lambda example: example["question_id"] in q_ids_dict)
        print(f'Dataset now of size: {len(matching_examples)}')
        return matching_examples
