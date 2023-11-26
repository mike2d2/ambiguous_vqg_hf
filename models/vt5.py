import requests
from PIL import Image 
import transformers
from transformers import ViltModel, ViltProcessor, T5ForConditionalGeneration, T5TokenizerFast
import torch
import torch.nn as nn

class VT5(nn.Module):
    def __init__(self, freeze_vlm = True):
        super().__init__()

        vilt_hf_path = 'dandelin/vilt-b32-finetuned-vqa'
        t5_hf_path = 't5-base'

        self.vilt_processor = ViltProcessor.from_pretrained(vilt_hf_path)
        self.vilt_model= ViltModel.from_pretrained(vilt_hf_path) # Replace this eventually
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_hf_path)
        self.t5_processor = T5TokenizerFast.from_pretrained(t5_hf_path)

        self.embedding_proj = nn.Linear(self.vilt_model.config.hidden_size, self.t5_model.config.d_model)

        # Freeze vilt
        if freeze_vlm:
            for param in self.vilt_model.parameters():
                param.requires_grad = False

    def forward(self, images, questions, answers, sample = False):
        device = self.vilt_model.device
        x = self.vilt_processor(images=images, text=answers, padding='max_length', return_tensors='pt') 
        x = {k:v.to(device) for k,v in x.items()}
        x = self.vilt_model(**x).last_hidden_state

        # Project outputs into right dimension
        x = self.embedding_proj(x)

        targets = self.t5_processor(questions, padding='longest', return_tensors='pt')['input_ids'].to(device)
        targets[targets == self.t5_processor.pad_token_id] = -100 # Remove padding from loss

        # Generation
        outputs = self.t5_model(inputs_embeds = x,  labels=targets)
        loss = outputs.loss # Span BCE Loss
        logits = outputs.logits
        
        text = None
        if sample:
            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            text = self.t5_processor.batch_decode(sequences=preds, skip_special_tokens=True)

        return {'loss':loss, 'logits':logits, 'text':text}
       

if __name__ == '__main__':
    device = torch.device(0)
    model = VT5().to(device)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    question = "How many cats are there?"
    answer = '2'

    output = model(image, question, answer, sample=True)
    print(output['loss'], output['text'])
