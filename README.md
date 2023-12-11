# ambiguous_vqg_hf

# Installation

Clone the repo, create an environment with conda or venv and run

```pip install -r requirements.txt```

also need to install spacy library

```python -m spacy download en_core_web_sm```

# Preprocessing

Data preprocessing happens automatically when training is run. Data downloads are handled by huggingface. Data preprocessing is performed in data/vilt_finetune_data_loader.py. Filtering the dataset to remove examples from the authors test dataset is also performed here and may take some time.

# Training

To train the ViLT model run 

```python train_vilt.py```

The dataset setup will pull the VQAv2 from huggingface and other models, this may take a while. Also "verbose" on line 21 can be set to True to see some output examples of VQA. 

To train the full VQG model with VT5 and frozen ViLT run

```python train_vt5.py```

Set force_toks to True to generate with constraints and False to generate without constraints.

# Evaluation

Evaluating the ViLT model on VQA task can be performed by running 

```python eval_vilt.py```

Evaluating the VT5 model can be performed by running 

```python eval_vt5.py```

Kmeans clustering can be performed on the test dataset by running

```python kmeans_eval.py```

which will run the test examples through the model, extract the data and then perform clustering. Scores will be saved in cluster_scores.json for the outputs selected. There are two choices for output, the output logits of Vilt or the output of the VT5 encoder. To switch between the two, change the qid_to_vectors argument of the get_prediction_clusters function  on line 354 to be qid_to_vilt_hidden_vec to get vilt cluster scores and change to qid_to_t5_hidden_vec to get VT5 cluster scores. 

# Pretrained model

We have uploaded weights for a pretrained VQG model which can be found on google drive by downloading the file real2-4.pt and placing in checkpoints/ directory, which must be created for the eval code to run:

```https://drive.google.com/file/d/1NWnVXvx12M4rAhU8MVhiLPurqkOKd_G8/view?usp=drive_link```
