import copy
import json
from pathlib import Path
import torch
from transformers.models.blenderbot.configuration_blenderbot import BlenderbotConfig
from transformers import BlenderbotTokenizer, BlenderbotSmallTokenizer
from modeling_blenderbot_cont_sum import BlenderbotForConditionalGeneration


tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
special_tokens_dict = {'additional_special_tokens': ['<keywords>', '<keyword_sep>', '</keywords>', '<gk>', '</gk>', '<persona>', '</persona>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

config = BlenderbotConfig.from_json_file("config_medium.json")
config.vocab_size = config.vocab_size + 7
model = BlenderbotForConditionalGeneration(config, return_ppl=True).to(f"cuda:0")
pytorch_total_params = sum(p.numel() for p in model.parameters())

weights_path = "blenderbot_pals.pth.tar"

if Path(weights_path).exists():
    checkpoint = torch.load(weights_path, map_location=torch.device(f"cuda:0"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print("------------------- loaded -----------------------")

encoder_seq_len = 96
decoder_seq_len = 32

def process_history(uttr, response, gk, dialog_act, sent):
    input_ids = []
    attention_mask = []
    
    cur_input_ids = tokenizer([uttr])["input_ids"][0]
    if cur_input_ids[-1] != 2:
        cur_input_ids.append(2)
        
    keywords_str = f"<gk> {gk} </gk>"
    keywords_ids = tokenizer([keywords_str])["input_ids"][0]
    input_ids = keywords_ids + cur_input_ids[-(encoder_seq_len - len(keywords_ids)):]
    
    attention_mask = [1 for _ in input_ids]
    
    label_ids = tokenizer([response])["input_ids"][0]
    if label_ids[-1] != 2:
        label_ids.append(2)
    
    inputs = {"input_ids": torch.LongTensor([input_ids[-encoder_seq_len:]]).to(f"cuda:0"),
              "attention_mask": torch.LongTensor([attention_mask[-encoder_seq_len:]]).to(f"cuda:0"),
              "dialog_acts": torch.LongTensor([int(dialog_act)]).to(f"cuda:0"),
              "sentiments": torch.LongTensor([int(sent)]).to(f"cuda:0"),
              "labels": torch.LongTensor([label_ids[:decoder_seq_len]]).to(f"cuda:0")
              }
    
    return inputs


with open("wow_da_sent_test.json", 'r') as fl:
    data_valid = json.load(fl)

ppl_list = []
for n, (uttr, response, da, sent, gk) in enumerate(data_valid):
    #inputs = process_history(uttr, response, gk, 4, 3)
    inputs = process_history(uttr, response, gk, int(da) - 1, sent)
    ppl = model(**inputs)
    ppl_list.append(ppl.item())

print("perplexity", round(sum(ppl_list), 2), round(sum(ppl_list) / len(ppl_list), 2))
