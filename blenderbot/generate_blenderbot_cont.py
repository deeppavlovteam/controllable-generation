import argparse
import copy
import json
import torch
from pathlib import Path
from transformers.models.blenderbot.configuration_blenderbot import BlenderbotConfig
from transformers import BlenderbotTokenizer, BlenderbotSmallTokenizer
from modeling_blenderbot_cont_sum import BlenderbotForConditionalGeneration


parser = argparse.ArgumentParser()
parser.add_argument("-d", action="store", dest="device")
parser.add_argument("-out", action="store", dest="outfile")
args = parser.parse_args()

tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
special_tokens_dict = {'additional_special_tokens': ['<keywords>', '<keyword_sep>', '</keywords>', '<gk>', '</gk>', '<persona>', '</persona>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

config = BlenderbotConfig.from_json_file("config_medium.json")
config.vocab_size = config.vocab_size + 7
model = BlenderbotForConditionalGeneration(config).to(f"cuda:{args.device}")

weights_path = "blenderbot_pals.pth.tar"

if Path(weights_path).exists():
    checkpoint = torch.load(weights_path, map_location=torch.device(f"cuda:{args.device}"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print("------------------- loaded -----------------------")

encoder_seq_len = 96
decoder_seq_len = 32

def process_history(cur_history, dialog_act, sent):
    input_ids = []
    attention_mask = []
    
    cur_uttr_hist = [elem for elem in cur_history]
    for j in range(len(cur_uttr_hist)):
        if j > 0:
            input_ids.append(1)
        cur_utt = copy.deepcopy(cur_uttr_hist[j])
        cur_input_ids = tokenizer([cur_utt])["input_ids"][0]
        if cur_input_ids[-1] != 2:
            cur_input_ids.append(2)
        input_ids += cur_input_ids
    
    attention_mask = [1 for _ in input_ids]
    
    inputs = {"input_ids": torch.LongTensor([input_ids[-encoder_seq_len:]]).to(f"cuda:{args.device}"),
              "attention_mask": torch.LongTensor([attention_mask[-encoder_seq_len:]]).to(f"cuda:{args.device}"),
              "dialog_acts": torch.LongTensor([int(dialog_act)]).to(f"cuda:{args.device}"),
              "sentiments": torch.LongTensor([int(sent)]).to(f"cuda:{args.device}")
              }
    
    return inputs

with open("daily_dialog_da_sent_val.tsv", 'r') as fl:
    lines = fl.readlines()

processed_samples = []

for n, line in enumerate(lines):
    line_split = line.strip().split("\t")
    history = line_split[0].split(" EOS ")
    gold_response = line_split[1]
    da = int(line_split[2]) - 1
    sent = int(line_split[3]) - 1
    inputs = process_history(history, 4, sent)
    reply_ids = model.generate(**inputs)
    response = tokenizer.batch_decode(reply_ids)[0]
    processed_samples.append([history, gold_response, response, da, sent])


with open(args.outfile, 'w') as out:
    json.dump(processed_samples, out, indent=2)

print("finished")
