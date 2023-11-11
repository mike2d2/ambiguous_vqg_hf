import itertools
import torch
import os
import numpy as np
import pandas as pd    
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from utils import remove_examples_by_qid
from vqa_dataset import VQADataset
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering, DefaultDataCollator
from PIL import Image
from tqdm.auto import tqdm
# os.environ['TRANSFORMERS_CACHE'] = "/scratch2/madefran/hf_cache"
saved_dataset_dir = 'saved_datasets/'
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
verbose = True

dataset = load_dataset("HuggingFaceM4/VQAv2", split='train[0:1000]')
print('loaded hugginface dataset')
# load ambiguity dataset
# jsonObj = pd.read_json(path_or_buf=file_path, lines=True)
ambiguous_vqa_dataset = load_dataset('json', data_files=f'ambiguous_vqa/data/cleaned_data.jsonl', split='train')

dataset = remove_examples_by_qid(dataset, ambiguous_vqa_dataset)

# uses vilt config to check vocab
config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

if verbose:
    print(dataset[0])
    image = Image.open(dataset[0]['image'].filename)


# vocab selection found here: https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/Using_ViLT_for_image_text_retrieval.ipynb#scrollTo=4qC5r7ZEUAcr
def get_score(count: int) -> float:
    return min(1.0, count / 3)


def add_labels(example):
    annotation = example
    answers = annotation['answers']
    answer_count = {}
    for answer in answers:
        answer_ = answer["answer"]
        answer_count[answer_] = answer_count.get(answer_, 0) + 1
    labels = []
    scores = []
    for answer in answer_count:
        if answer not in list(config.label2id.keys()):
            continue
        labels.append(config.label2id[answer])
        score = get_score(answer_count[answer])
        scores.append(score)
    annotation['labels'] = labels
    annotation['scores'] = scores
    return example


dataset = dataset.map(add_labels)

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

# Torch Dataset

dataset.save_to_disk(saved_dataset_dir)
dataset = VQADataset(questions=dataset['question'],
                     dataset=dataset,
                     processor=processor,
                     config=config)
if verbose:
    print(processor.decode(dataset[0]['input_ids']))

#labels = torch.nonzero(dataset[0]['labels']).squeeze().tolist()
# print([config.id2label[label] for label in labels])

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 id2label=config.id2label,
                                                 label2id=config.label2id)

model.to(device)

# combines features into concatenations
# data_collator = DefaultDataCollator()

def collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  pixel_values = [item['pixel_values'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  token_type_ids = [item['token_type_ids'] for item in batch]
  labels = [item['labels'] for item in batch]

  # create padded pixel values and corresponding pixel mask
  encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

  # create new batch
  batch = {}
  batch['input_ids'] = torch.stack(input_ids)
  batch['attention_mask'] = torch.stack(attention_mask)
  batch['token_type_ids'] = torch.stack(token_type_ids)
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = torch.stack(labels)

  return batch

train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)

# Visualize batch and decode random examples
if verbose:
    batch = next(iter(train_dataloader))
    for k,v in batch.items():
        print(k, v.shape)

    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    batch_idx = 1

    unnormalized_image = (batch["pixel_values"][batch_idx].numpy() * np.array(image_mean)[:, None, None]) + np.array(image_std)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    im = Image.fromarray(unnormalized_image)
    im.show()

    print(processor.decode(batch["input_ids"][batch_idx]))

    labels = torch.nonzero(batch['labels'][batch_idx]).tolist()[0]
    print([config.id2label[label] for label in labels])

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(50):  # loop over the dataset multiple times
    print(f"Epoch: {epoch}")
    for batch in tqdm(train_dataloader):
            # get the inputs;
            batch = {k:v.to(device) for k,v in batch.items()}

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            print("Loss:", loss.item())
            loss.backward()
            optimizer.step()

pass
