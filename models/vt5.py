import requests
from PIL import Image 
import transformers
from transformers import ViltModel, ViltProcessor, T5ForConditionalGeneration, T5TokenizerFast
import torch
import torch.nn as nn
import torch.nn.functional as F

class VT5(nn.Module):
    def __init__(self, vilt_chkpt = None, freeze_vlm = True):
        super().__init__()

        vilt_hf_path = 'dandelin/vilt-b32-finetuned-vqa'
        t5_hf_path = 't5-base'

        self.vilt_processor = ViltProcessor.from_pretrained(vilt_hf_path)
        self.vilt_model= ViltModel.from_pretrained(vilt_hf_path) # Replace this eventually
        if vilt_chkpt != None:
            ft_model = torch.load(vilt_chkpt)
            if isinstance(ft_model, nn.DataParallel):
                state_dict = ft_model.module.state_dict()
            else:
                state_dict = ft_model.state_dict()
            self.vilt_model.load_state_dict(state_dict)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_hf_path)
        self.t5_processor = T5TokenizerFast.from_pretrained(t5_hf_path)
        self.embedding_proj = nn.Linear(self.vilt_model.config.hidden_size, self.t5_model.config.d_model)
        
        # Freeze vilt
        if freeze_vlm:
            for param in self.vilt_model.parameters():
                param.requires_grad = False

        # Generation params
        self.beam_size=5

    def forward(self, images, questions, answers, sample = False):
        device = self.vilt_model.device
        x = self.vilt_processor(images=images, text=answers, padding='max_length', return_tensors='pt') 
        x = {k:v.to(device) for k,v in x.items()}
        x = self.vilt_model(**x).last_hidden_state

        # Project outputs into right dimension
        x = self.embedding_proj(x)

        targets = self.t5_processor(questions, padding=True, max_length=64, return_tensors='pt')['input_ids'].to(device)
        targets[targets == self.t5_processor.pad_token_id] = -100 # Remove padding from loss
        print(targets.shape)

        # Generation
        outputs = self.t5_model(inputs_embeds = x,  labels=targets)
        loss = outputs.loss # Span BCE Loss
        logits = outputs.logits
        
        text = None
        if sample:
            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            text = self.t5_processor.batch_decode(sequences=preds, skip_special_tokens=True)

        return {'loss':loss, 'logits':logits, 'text':text}
    
    def generate(self, images, answers, force_toks):
        device = self.vilt_model.device
        x = self.vilt_processor(images=images, text=answers, padding='max_length', return_tensors='pt') 
        x = {k:v.to(device) for k,v in x.items()}
        x = self.vilt_model(**x).last_hidden_state

        # Project outputs into right dimension
        x = self.embedding_proj(x)

        # Generate
        if force_toks is not None:
            force_words_ids = []
            for tok_seq in force_toks:
                if len(tok_seq) > 0:
                    tokenized = self.t5_processor(tok_seq, padding=False, add_special_tokens=False)
                    input_ids = [[y]  for x in tokenized['input_ids'] for y in x]
                    # pdb.set_trace()
                    force_words_ids.append(input_ids)
                else:
                    force_words_ids.append(None)
        else:
            force_words_ids = None
        if force_words_ids is not None and len(force_words_ids) == 0:
            force_words_ids = None
        if force_words_ids is not None: 
            pred_toks, pred_lens = [], []
            for i in range(x.shape[0]): 
                seq_slice = x[i,:,:].unsqueeze(0)
                try:
                    if force_words_ids[i] is None:
                        force_slice = None
                    else:
                        force_slice = [force_words_ids[i]]
                except IndexError:
                    pdb.set_trace()
                try:
                    pred_dict = self.t5_model.generate(inputs_embeds = seq_slice, 
                                                            num_beams=self.beam_size, 
                                                            force_words_ids = force_slice,
                                                            output_hidden_states=True,
                                                            return_dict_in_generate=True)
                except ValueError:
                    # In the rare case that a word gets split into subwords and there are repeated subwords
                    # we will just ignore it    
                    pred_dict = self.t5_model.generate(inputs_embeds = seq_slice, 
                                                            num_beams=self.beam_size, 
                                                            force_words_ids = None,
                                                            output_hidden_states=True,
                                                            return_dict_in_generate=True)
                pred_slice = pred_dict['sequences']
                pred_lens.append(pred_slice.shape[1])
                pred_toks.append(pred_slice) 
            max_len = max(pred_lens)
            for i, pred in enumerate(pred_toks): 
                curr_len = pred.shape[1]
                pred = torch.nn.functional.pad(pred, (0, max_len - curr_len), value=self.t5_processor.pad_token_id)
                pred_toks[i] = pred
            pred_toks = torch.cat(pred_toks, dim=0).squeeze(1)
            # pdb.set_trace() 

        else:
            pred_dict = self.t5_model.generate(inputs_embeds = x, 
                                                num_beams=self.beam_size, 
                                                output_hidden_states=True,
                                                return_dict_in_generate=True)
            pred_toks = pred_dict['sequences']

        text = self.t5_processor.batch_decode(pred_toks, skip_special_tokens=True)
        return {'tokens':pred_toks, 'text':text}


if __name__ == '__main__':
    device = torch.device(0)
    model = VT5().to(device)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    question = "How many cats are there? There are four dogs."
    answer = '2'

    output = model(image, question, answer, sample=True)
    print(output['loss'], output['text'], output['logits'].shape)

    output = model.generate(image, answer)
    print(output)
