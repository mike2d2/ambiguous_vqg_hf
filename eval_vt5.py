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
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering, DefaultDataCollator, AutoTokenizer
from PIL import Image
from tqdm.auto import tqdm
from nltk.translate.bleu_score import corpus_bleu
import wandb
import torch.nn as nn
import re
import random
from utils import get_nps
random.seed(101)

model_path = './checkpoints/0.pt'
saved_dataset_dir = 'saved_dataset_val/'
add_disjunctive_constraints=True

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
verbose = False

# uses vilt config to check vocab

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
dataset = ViltFinetuneDataLoader.from_saved_or_load(
    split='validation',
    config=config,
    verbose=verbose, 
    force_reload=False,
    save_dir=saved_dataset_dir)
print(f'loaded dataset with {len(dataset)} examples')
dataset = VQGDataset(dataset, config)

# Do we need this for data parallel?
#vl_processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
#t5_processor = T5TokenizerFast.from_pretrained('t5-base')

# We load in the model this way because we added the generate fucntion
model = VT5()
checkpoint = torch.load(model_path)
if isinstance(checkpoint, nn.DataParallel):
    checkpoint = checkpoint.module
model.load_state_dict(checkpoint.state_dict())

model.to(device)

def collate_fn(x):
    images = [item[0] for item in x]
    questions = [item[1] for item in x]
    answers = [item[2] for item in x]
    return images, questions, answers
train_dataloader = DataLoader(dataset,batch_size=128, collate_fn = collate_fn, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.eval()
with torch.no_grad():
    pred = []
    gt = []
    idx = 0
    for images, questions, answers in tqdm(train_dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()
        if add_disjunctive_constraints:
            constraints = []
            for question in questions:
                constraints+=[get_nps(question)]
        outputs = model.generate(images, answers, force_toks = constraints)
        texts = outputs['text']
        pred.extend(texts)
        gt.extend(questions)
        
        idx+=1
        #if idx == 1:
    #    break

tokenizer = AutoTokenizer.from_pretrained('t5-base')

def tokenize(sentence):
    sentence = re.sub("<.*?>", "", sentence)
    sentence = tokenizer.tokenize(sentence)
    return sentence
print(pred[:10], gt[:10])
references = []
candidates = []
for candidate, reference in zip(pred, gt):
    candidate_tok = tokenize(candidate)
    reference_tok = tokenize(reference)
    print(reference_tok, candidate_tok)
    references.append(reference_tok)
    candidates.append(candidate_tok)

weights = [(1, 0, 0, 0),
            (1./2., 1./2., 0, 0),
            (1./3., 1./3., 1./3., 0),
            (1./4., 1./4., 1./4., 1./4.)]

mean_bleu_score = corpus_bleu(references, candidates, weights)
for i in range(4):
    print(f"BLEU-{i+1}: {mean_bleu_score[i]:.2f}")
