import torch
from PIL import Image
import random
random.seed(10)

class VQGDataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, dataset, config):
        self.questions = dataset['question']
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        question = self.questions[idx]
        answers = [answer['answer'] for answer in item['answers'] if answer['answer_confidence'] == 'yes' or answer['answer_confidence'] == 'maybe']
        if len(answers) == 0:
            # Choose another random idx
            idx = random.randint(0, len(self.dataset)-1)
            return self.__getitem__(idx)
        answer = random.choice(answers)
        return image, question, answer
