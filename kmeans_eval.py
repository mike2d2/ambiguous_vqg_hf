import torch
from tqdm import tqdm
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering

from data.vilt_finetune_data_loader import ViltFinetuneDataLoader
from data.vqa_dataset import VQADataset
from torch.utils.data import DataLoader


saved_dataset_dir = 'saved_datasets/test_data/'
saved_model_dir = 'saved_models/run_1/epoch_9'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
verbose = False

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
dataset = ViltFinetuneDataLoader.from_saved_or_load(
    config=config,
    verbose=verbose, 
    force_reload=False, 
    save_dir=saved_dataset_dir, 
    truncate_len=None,
    remove=False)

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
dataset = VQADataset(questions=dataset['question'],
                     dataset=dataset,
                     processor=processor,
                     config=config)

model = ViltForQuestionAnswering.from_pretrained(saved_model_dir, local_files_only=True,
                                                 id2label=config.id2label,
                                                 label2id=config.label2id)

model.to(device)
model.eval()