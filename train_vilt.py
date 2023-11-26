import itertools
import torch
import os
import numpy as np
import pandas as pd    
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from data.vilt_finetune_data_loader import ViltFinetuneDataLoader
from data.vqa_dataset import VQADataset
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering, DefaultDataCollator
from PIL import Image
from tqdm.auto import tqdm

saved_dataset_dir = 'saved_datasets/'
saved_model_dir = 'saved_models/run_1/epoch_9'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
verbose = False


# uses vilt config to check vocab
config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
dataset = ViltFinetuneDataLoader.from_saved_or_load(config=config, verbose=verbose, force_reload=False, save_dir=saved_dataset_dir)
print(f'loaded dataset with {len(dataset)} examples')
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
dataset = VQADataset(questions=dataset['question'],
                     dataset=dataset,
                     processor=processor,
                     config=config)
if verbose:
    print(processor.decode(dataset[0]['input_ids']))

#labels = torch.nonzero(dataset[0]['labels']).squeeze().tolist()
# print([config.id2label[label] for label in labels])

# model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
#                                                  id2label=config.id2label,
#                                                  label2id=config.label2id)
model = ViltForQuestionAnswering.from_pretrained(saved_model_dir, local_files_only=True,
                                                 id2label=config.id2label,
                                                 label2id=config.label2id)

# device_ids = [0, 1]
# model = DataParallel(model, device_ids=device_ids)
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

train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=30, shuffle=True)

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
losses = []
for epoch in range(50): 
    print(f"Epoch: {epoch}")
    for batch in tqdm(train_dataloader):
            # get the inputs;
            batch = {k:v.to(device) for k,v in batch.items()}

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss

            if verbose:
                print("Loss:", loss.item())
                losses.append(loss.item())
            loss.backward()
            optimizer.step()

    print(str(losses[-1]))
    model.save_pretrained(f'saved_models/epoch_{str(epoch)}')
pass