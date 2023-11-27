from collections import defaultdict
import json
import torch
from tqdm import tqdm
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering
from data.kmeans_dataset import KmeansDataset

from data.vilt_finetune_data_loader import ViltFinetuneDataLoader
from data.vqa_dataset import VQADataset
from torch.utils.data import DataLoader

from datasets import Dataset


import itertools
import torch
import os
import numpy as np
import pandas as pd
from models.kmeans_cluster import KMeansCluster
from models.vt5 import VT5
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from data.vilt_finetune_data_loader import ViltFinetuneDataLoader
from data.vqg_dataset import VQGDataset
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering, DefaultDataCollator, AutoTokenizer
from PIL import Image
from tqdm.auto import tqdm
from scipy.optimize import linear_sum_assignment
from nltk.translate.bleu_score import corpus_bleu
from models.vt5_kmeans import VT5Kmeans
import wandb
import torch.nn as nn
import re
import random
from utils import get_nps
random.seed(101)

model_path = './checkpoints/real2-4.pt'
saved_dataset_dir = 'saved_datasets/test_data'
add_disjunctive_constraints=True

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
verbose = False

test_annotations = json.load(open("datasets/test_annotations.json"))['annotations']
test_questions = json.load(open("datasets/test_questions.json"))['questions']

# get the clusters from annotations 
def get_annotator_clusters(questions, annotations): 
    anns_by_qid = defaultdict(list)
    for quest, ann in zip(questions, annotations):

        qid, i = quest['question_id'].split("_")
        anns_by_qid[qid].append((quest, ann))

    clusters_by_qid = {}
    for qid, list_of_qas in anns_by_qid.items():
        clusters = defaultdict(list)
        for quest, ann in list_of_qas:
            rewritten = quest['new_question']
            answer = ann['answers'][0]['answer']
            answer_id = ann['answers'][0]['mturk_id']
            cluster_dict = {"answer": answer, "id": answer_id} 
            clusters[rewritten].append(cluster_dict)
        clusters_by_qid[qid] = clusters
    return clusters_by_qid

ann_clusters = get_annotator_clusters(test_questions, test_annotations)



# uses vilt config to check vocab

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
dataset = ViltFinetuneDataLoader.from_saved_or_load(
    split='train',
    config=config,
    verbose=verbose, 
    force_reload=False,
    save_dir=saved_dataset_dir)
print(f'loaded dataset with {len(dataset)} examples')

# for each item in this dataset create a new dataset with just one question and every answer and image
answers_for_qid = []

for example in dataset:
    qid = str(example["question_id"])  # Assuming 'id' contains the question ID in your dataset

    # Get answers for the current question ID from ann_clusters dictionary
    answers = []
    if qid in ann_clusters:
        for cluster_dict in ann_clusters[qid].values():
            for ans_dict in cluster_dict:
                answers.append(ans_dict["answer"])

    # Create a new dataset with answers for the current question ID
    answers_dataset = Dataset.from_dict(
        {
            "qid": [qid] * len(answers), 
            "question": [example["question"]] * len(answers),
            "image": [example["image"]] * len(answers), 
            "answer": answers
        })
    
    answers_for_qid.append(answers_dataset)

# 'answers_for_qid' will contain a list of datasets, each dataset containing answers for a specific question ID



# Do we need this for data parallel?
#vl_processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
#t5_processor = T5TokenizerFast.from_pretrained('t5-base')

# We load in the model this way because we added the generate fucntion
model = VT5Kmeans()
checkpoint = torch.load(model_path)
if isinstance(checkpoint, nn.DataParallel):
    checkpoint = checkpoint.module
model.load_state_dict(checkpoint.state_dict())

model.to(device)
model.eval()

def collate_fn(x):
    images = [item[0] for item in x]
    questions = [item[1] for item in x]
    answers = [item[2] for item in x]
    return images, questions, answers

def cluster_vectors(vectors, cluster_method='optics', do_pca=False, pca_comp_max=2, pred_data = None):
    if pred_data is None: 
        vidxs = [x[1] for x in vectors]
        vectors = np.vstack([x[0] for x in vectors]).reshape(len(vectors), -1)
        if do_pca:
            from sklearn.decomposition import PCA
            comp = min(vectors.shape[0], pca_comp_max)
            pca = PCA(n_components=comp)
            vectors = pca.fit_transform(vectors)

        # if cluster_method == "optics":
        #     clust = OPTICS(min_samples=2, metric='euclidean')
        # elif cluster_method == "mean_shift":
        #     bw = estimate_bandwidth(vectors, quantile=0.5)
        #     if bw < 0.0001:
        #         bw = 0.0001
        #     clust = MeanShift(bandwidth=bw)
       # elif cluster_method == "kmeans": 
        clust = KMeansCluster(penalty_factor=52.0)
        # elif cluster_method == "gmm": 
        #     clust = GMMCluster(use_aic=True)
        # elif cluster_method == "bayes_gmm": 
        #     clust = BayesianGMMCluster()

        # else:
        #     raise AssertionError("Unknown cluster method")
        clust.fit(vectors)
        clusters = defaultdict(list)
        max_label = max(clust.labels_) + 1
        for i, vidx in enumerate(vidxs):
            label = clust.labels_[i]
            if label == -1:
                label = max_label
            clusters[label].append(vidx)
    else:
        pass
        # clusters = get_clusters_from_pred_data(pred_data)
    return clusters 

# get the clusters from predictions 
def get_prediction_clusters(questions, annotations, qid_to_vectors, cluster_method='optics', do_pca=False, pca_comp_max=2, pred_data=None):
    anns_by_qid = defaultdict(list)
    for quest, ann in zip(questions, annotations):
        qid, i = quest['question_id'].split("_")
        anns_by_qid[qid].append((quest, ann))

    vectors_by_qid = defaultdict(list)
    answers_by_qid = defaultdict(list)

    # print(anns_by_qid)
    for qid, list_of_qas in anns_by_qid.items():
        image_id = list_of_qas[0][0]['image_id']
        for quest, ann in list_of_qas:
            qid, i = quest['question_id'].split("_")
            # path = pathlib.Path(save_dir).joinpath(f"{image_id}_{qid}_{i}_0.pt")
            # vector = torch.load(path, map_location=torch.device('cpu')).detach().numpy()
            if qid in qid_to_vectors:
                vector = qid_to_vectors[qid][int(i)].detach().numpy()
                answer = ann['answers'][0]['answer']
                answer_id = ann['answers'][0]['mturk_id']
                vectors_by_qid[qid].append((vector, int(i)))
                answers_by_qid[qid].append({"answer": answer, "id": answer_id})
            else:
                pass

    clusters_by_qid = {}
    for qid, vectors in vectors_by_qid.items():
        # pdb.set_trace()
        # pred_data_at_qid = [x for x in pred_data if x['question_id'].split("_")[0] == qid]
        # pdb.set_trace() 
        clusters = cluster_vectors(vectors, cluster_method=cluster_method, do_pca=do_pca, pca_comp_max=pca_comp_max) #, pred_data = pred_data_at_qid)

        clusters = {k: [answers_by_qid[qid][idx] for idx in v ] for k, v in clusters.items()}
        clusters_by_qid[qid] = clusters

    return clusters_by_qid

def preprocess(cluster_data):
    if type(cluster_data) in [dict, defaultdict]:
        # dealing with predicted clusters or preprocessed clusters
        return cluster_data.values()
    return cluster_data

def safe_divide(num, denom): 
    try: 
        return num/denom
    except ZeroDivisionError:
        return 0

def f1_helper(group1, group2): 
    """
    Helper function to compute the F1 score
    between two groups 
    """
    ids1 = set([x['id'] for x in group1])
    ids2 = set([x['id'] for x in group2])
    # treat 1 as pred, 2 as true (symmetric so doesn't matter)
    tp = len(ids1 & ids2)
    fp = len(ids1 - ids2) 
    fn = len(ids2 - ids1)

    precision = safe_divide(tp, tp+fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1

def f1_score(groups1, groups2):
    """
    Compute the f1 score between two sets of groups.
    First, compute the F1 score between each of the 
    possible set combinations, then use the 
    Hungarian algorithm to find the maximum assignment,
    i.e. the best alignment between groups in the two sets.
    """
    # what do if groups are different length? 
    # just compute quadriatic, take max  
    p_scores = np.zeros((len(groups1), len(groups2)))
    r_scores = np.zeros((len(groups1), len(groups2)))
    f1_scores = np.zeros((len(groups1), len(groups2)))
    for i, group1 in enumerate(groups1): 
        for j, group2 in enumerate(groups2):  
            p, r, f1 = f1_helper(group1, group2) 
            p_scores[i,j] = p
            r_scores[i,j] = r
            f1_scores[i,j] = f1
    cost_matrix = np.ones_like(f1_scores) * np.max(f1_scores) - f1_scores
    f1_assignment = linear_sum_assignment(cost_matrix) 
    best_f1_scores = f1_scores[f1_assignment]
    best_p_scores = p_scores[f1_assignment]
    best_r_scores = r_scores[f1_assignment]
    return np.mean(best_f1_scores, axis=0), np.mean(best_p_scores, axis=0), np.mean(best_r_scores, axis=0), f1_assignment


def get_scores(clusters_by_qid_a, clusters_by_qid_b):
    f1_scores = []
    p_scores = []
    r_scores = []
    cluster_lens_a = []
    cluster_lens_b = []
    for qid in clusters_by_qid_a.keys():
        cluster_a = preprocess(clusters_by_qid_a[qid])
        cluster_b = preprocess(clusters_by_qid_b[qid])
        f1, p, r, _ = f1_score(cluster_a, cluster_b)
        f1_scores.append(f1)
        p_scores.append(p)
        r_scores.append(r)
        cluster_lens_a.append(len(cluster_a))
        cluster_lens_b.append(len(cluster_b))
    return np.mean(f1_scores), np.mean(p_scores), np.mean(r_scores), np.mean(cluster_lens_a), np.mean(cluster_lens_b)


############ START EVAL ###############

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

qid_to_vilt_hidden_vec = {}
qid_to_t5_hidden_vec = {}

scores = {}
f1s = []
rs= []
ps = []

for dataset in answers_for_qid:
    # loop through each list of answers and grab all output vectors, and run kmeans on them
    qid = dataset[0]['qid']
    dataset = KmeansDataset(dataset)

    train_dataloader = DataLoader(dataset,batch_size=1, collate_fn = collate_fn, shuffle=True)

    vilt_hidden_vectors = []
    t5_hidden_vectors = []

    with torch.no_grad():
        pred = []
        gt = []
        idx = 0
        for images, questions, answers in tqdm(train_dataloader):
            # zero the parameter gradients
            optimizer.zero_grad()
            # if add_disjunctive_constraints:
            #     constraints = []
            #     for question in questions:
            #         constraints+=[get_nps(question)]
            # outputs = model.generate(images, answers, force_toks = constraints)

            outputs = model.generate(images, answers, force_toks=None)
            vilt_hidden_vectors.append(outputs['vilt_out_hidden'])
            t5_hidden_vectors.append(outputs['t5_out_hidden'])
            
            texts = outputs['text']
            pred.extend(texts)
            gt.extend(questions)
            
            idx+=1
            if idx == 10:
                break

    qid_to_vilt_hidden_vec[qid] = vilt_hidden_vectors
    qid_to_t5_hidden_vec[qid] = t5_hidden_vectors

    pred_clusters = get_prediction_clusters(test_questions, 
                                    test_annotations, 
                                    # str(checkpoint_dir), 
                                    qid_to_vectors=qid_to_vilt_hidden_vec,
                                    cluster_method='kmeans', 
                                    do_pca=True,
                                    pca_comp_max=5,
                                    pred_data = None)


    out_scores = (f1_pred_to_ann, 
        p_pred_to_ann, 
        r_pred_to_ann,
        cluster_lens_pred, __) = get_scores(pred_clusters, ann_clusters)
    
    scores[qid] = out_scores
    f1s.append(f1_pred_to_ann)
    ps.append(p_pred_to_ann)
    rs.append(r_pred_to_ann)

print('avg f1: ', sum(f1s)/len(f1s))
print('avg p: ', sum(ps)/len(ps))
print('avg r: ', sum(rs)/len(rs))





    # run kmeans on the vectors


    # print(pred[:10], gt[:10])
    # tokenizer = AutoTokenizer.from_pretrained('t5-base')

    # def tokenize(sentence):
    #     sentence = re.sub("<.*?>", "", sentence)
    #     sentence = tokenizer.tokenize(sentence)
    #     return sentence
    # print(pred[:10], gt[:10])
    # references = []
    # candidates = []
    # for candidate, reference in zip(pred, gt):
    #     candidate_tok = tokenize(candidate)
    #     reference_tok = tokenize(reference)
    #     print(reference_tok, candidate_tok)
    #     references.append([reference_tok])
    #     candidates.append(candidate_tok)

    # weights = [(1, 0, 0, 0),
    #             (1./2., 1./2., 0, 0),
    #             (1./3., 1./3., 1./3., 0),
    #             (1./4., 1./4., 1./4., 1./4.)]

    # mean_bleu_score = corpus_bleu(references, candidates, weights)
    # for i in range(4):
    #     print(f"BLEU-{i+1}: {mean_bleu_score[i]:.2f}")
