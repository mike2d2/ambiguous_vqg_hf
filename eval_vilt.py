import torch
from tqdm import tqdm
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering

from data.vilt_finetune_data_loader import ViltFinetuneDataLoader
from data.vqa_dataset import VQADataset
from torch.utils.data import DataLoader


saved_dataset_dir = 'saved_datasets/'
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
    truncate_len=None)

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

example = dataset[0]
print(example.keys())
print(processor.decode(example['input_ids']))

example = {k: v.unsqueeze(0).to(device) for k,v in example.items()}

# forward pass
outputs = model(**example)

logits = outputs.logits
predicted_classes = torch.sigmoid(logits)

probs, classes = torch.topk(predicted_classes, 5)

for prob, class_idx in zip(probs.squeeze().tolist(), classes.squeeze().tolist()):
  print(prob, model.config.id2label[class_idx])

correct = 0
with torch.no_grad():
    for batch in tqdm(train_dataloader):

        inputs = batch['input_ids'].to(device)  # Assuming 'input_ids' is a key in your dataset
        labels = batch['labels'].to(device)

        # forward pass
        outputs = model(**batch)
        _, predicted = torch.max(outputs.logits, 1)
        _, label_ids = torch.max(labels, 1)
        
        correct += (predicted == label_ids).sum().item()

accuracy = correct / len(dataset)








