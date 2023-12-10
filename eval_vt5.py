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
from rouge import Rouge
from cider.pyciderevalcap.cider.cider import Cider
import sys
import wandb
import torch.nn as nn
import re
import random
from utils import get_nps
random.seed(101)

model_path = './checkpoints/real2-5.pt'
saved_dataset_dir = 'saved_datasets_ambiguous/'
add_disjunctive_constraints=True
save_images = True
save_text = True

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
verbose = False

# uses vilt config to check vocab

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
dataset = ViltFinetuneDataLoader.from_saved_or_load(
    split='train',
    config=config,
    verbose=verbose, 
    force_reload=False,
    save_dir=saved_dataset_dir,
    remove=False)

dataset = ViltFinetuneDataLoader.from_saved_or_load(
    split='val',
    config=config,
    verbose=verbose, 
    force_reload=False,
    save_dir=saved_dataset_dir,
    remove=False)

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
eval_dataloader = DataLoader(dataset,batch_size=128, collate_fn = collate_fn, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.eval()
with torch.no_grad():
    pred = []
    gt = []
    idx = 0
    for images, questions, answers in tqdm(eval_dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()
        if add_disjunctive_constraints:
            constraints = []
            for question in questions:
                constraints+=[get_nps(question)]
        outputs = model.generate(images, answers, force_toks = constraints)
        #outputs = model.generate(images, answers, force_toks=None)
        texts = outputs['text']
        pred.extend(texts)
        gt.extend(questions)
        if save_images:
            for idx, image in enumerate(images):
                image.save(f'examples/{idx}.jpg')
        if save_text:
            f = open('./text.txt', 'w')
            for idx in range(len(questions)):
                question = questions[idx]
                answer = answers[idx]
                gen = texts[idx]
                out = f'{question}\n{answer}\n{gen}\n'
                print(out, file = f)
            f.close()
        idx+=1

print(pred[:10], gt[:10])
tokenizer = AutoTokenizer.from_pretrained('t5-base')

def tokenize(sentence):
    sentence = re.sub("<.*?>", "", sentence)
    sentence = tokenizer.tokenize(sentence)
    return sentence
print(pred[:10], gt[:10])
references_superlist_tok = []
references_tok = []
candidates_tok = []
candidates_superlist_tok = []
references = []
candidates = []
for candidate, reference in zip(pred, gt):
    candidate_tok = tokenize(candidate)
    reference_tok = tokenize(reference)
    print(reference_tok, candidate_tok)
    references_superlist_tok.append([reference_tok])
    references_tok.append(reference_tok)
    candidates_superlist_tok.append([candidate_tok])
    candidates_tok.append(candidate_tok)
    references.append(reference)
    candidates.append(candidate)

with open('generations.txt','w') as f:
    for gen in pred:
        print(gen, file = f)

cider_scorer = Cider()
golds, preds = references, candidates
fake_golds = {i: [g] for i, g in enumerate(golds)}
fake_preds = [{"caption": [p], "image_id": i} for i, p in enumerate(preds)]
cider_score, __ = cider_scorer.compute_score(fake_golds, fake_preds)

weights = [(1, 0, 0, 0),
            (1./2., 1./2., 0, 0),
            (1./3., 1./3., 1./3., 0),
            (1./4., 1./4., 1./4., 1./4.)]
rouge = Rouge()
mean_bleu_score = corpus_bleu(references_superlist_tok, candidates_tok, weights)
rouge_score_dict = rouge.get_scores(references, candidates, avg=True)
rouge_avg_score = np.mean(rouge_score_dict['rouge-l']['f'])
for i in range(4):
    print(f"BLEU-{i+1}: {mean_bleu_score[i]:.2f}")
print(f"ROUGE: {rouge_avg_score:.2f}")

print(f"CIDE-r: {cider_score:.2f}")
