import copy
import json
import torch
from transformers.models.blenderbot.configuration_blenderbot import BlenderbotConfig
from transformers import BlenderbotTokenizer, BlenderbotSmallTokenizer
from modeling_blenderbot import BlenderbotForConditionalGeneration


model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model.return_ppl = True

model.to(f"cuda:0")

with open("wow_da_sent_test.json", 'r') as fl:
    data_valid = json.load(fl)

dialog_acts_dict = {}
sentiment_dict = {}

ppl_list = []
for n, (uttr, response, da, sent, gk) in enumerate(data_valid):
    uttr = f"{gk} </s> {uttr}"
    encoding = tokenizer([uttr])
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    input_ids = input_ids[:58]
    attention_mask = attention_mask[:58]
    encoding = tokenizer([response])
    label_ids = encoding["input_ids"][0]
    
    inputs = {"input_ids": torch.LongTensor([input_ids]).to(f"cuda:0"),
              "attention_mask": torch.LongTensor([attention_mask]).to(f"cuda:0"),
              "labels": torch.LongTensor([label_ids]).to(f"cuda:0")}
    ppl = model(**inputs)
    ppl_list.append(ppl.item())

print("perplexity", round(sum(ppl_list), 4), round(sum(ppl_list) / len(ppl_list), 4))
