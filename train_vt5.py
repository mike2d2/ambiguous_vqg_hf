import itertools
import torch
import os
import numpy as np
import pandas as pd
from models.vt5 import VT5
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from data.vilt_finetune_data_loader import ViltFinetuneDataLoader
from data.vqg_dataset import VQGDataset
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering, DefaultDataCollator
from PIL import Image
from tqdm.auto import tqdm
from nltk.translate.bleu_score import corpus_bleu
import wandb
import torch.nn as nn

enable_wandb = True
checkpoint = 'checkpoints/0.pt'
if enable_wandb:
    wandb.login()
    run_name = 'real1'
    wandb.init(project = 'vqg', name = run_name)

saved_dataset_dir = 'saved_datasets/'

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
dataset = VQGDataset(dataset, config)

# Do we need this for data parallel?
#vl_processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
#t5_processor = T5TokenizerFast.from_pretrained('t5-base')

if checkpoint == None:
    model = VT5()
else:
    model = torch.load(checkpoint)
    if isinstance(model, nn.DataParallel):
        model = model.module

device_ids = [0]
model = DataParallel(model, device_ids=device_ids)
model.to(device)

def collate_fn(x):
    images = [item[0] for item in x]
    questions = [item[1] for item in x]
    answers = [item[2] for item in x]
    return images, questions, answers
train_dataloader = DataLoader(dataset,batch_size=80, collate_fn = collate_fn, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
for epoch in range(50): 
    print(f"Epoch: {epoch}")
    for images, questions, answers in tqdm(train_dataloader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images, questions, answers, sample=True)
            loss = outputs['loss']
            text = outputs['text']
            
            loss.backward()
            optimizer.step()
             
            metrics = {'train_ce': loss.item()}
            if enable_wandb:
                wandb.log(metrics)
    # TODO: Compute bleu on dev set
    weights = [(1, 0, 0, 0),
                (1./2., 1./2., 0, 0),
                (1./3., 1./3., 1./3., 0),
                (1./4., 1./4., 1./4., 1./4.)]
    torch.save(model,f'checkpoints/{epoch}.pt')
