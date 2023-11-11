import torch
from PIL import Image

class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, questions, annotations, processor, config):
        self.questions = questions
        self.annotations = annotations
        self.processor = processor

        # config steals vocab from another pretrained on vqa vilt model
        self.config = config

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        question = self.questions[idx]
        # image = Image.open(id_to_filename[annotation['image_id']])
        image = annotation['image']
        text = question

        if(image.mode != 'RGB'):
            print('gray')

        image = image.convert('RGB')
        # elif len(image.shape)==3:
        #     print('Color(RGB)')
        # else:
        #     print('others')


        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k,v in encoding.items():
          encoding[k] = v.squeeze()
        # add labels
        labels = annotation['labels']
        scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(self.config.id2label))
        for label, score in zip(labels, scores):
              targets[label] = score
        encoding["labels"] = targets

        return encoding