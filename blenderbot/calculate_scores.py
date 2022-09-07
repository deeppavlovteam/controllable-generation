import argparse
import json
import time
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from deeppavlov import configs, build_model
from deeppavlov.core.commands.utils import parse_config


parser = argparse.ArgumentParser()
parser.add_argument("-in", action="store", dest="infile")
args = parser.parse_args()

config_da = parse_config('dialog_acts_hist.json')
config_da["chainer"]["pipe"][3]["device"] = "cuda:0"
model_da = build_model(config_da)

config_sent = parse_config('sentiment_hist.json')
config_sent["chainer"]["pipe"][3]["device"] = "cuda:0"
model_sent = build_model(config_sent)

batch_size = 30

sentiment_label_dict = {0: 0, -1: 2, 1: 1}
pred_dialog_acts = []
gold_dialog_acts = []
pred_sentiment = []
gold_sentiment = []


fl = open(args.infile, 'r')
data = json.load(fl)
num_batches = len(data) // batch_size + int(len(data) % batch_size > 0)
for j in range(num_batches):
    utt1, utt2, da_batch, sent_batch = [], [], [], []
    cur_chunk = data[j*batch_size:(j+1)*batch_size]
    for history, gold_response, response, da, sent in cur_chunk:
        utt1.append(history[-1])
        for symb in ["</s>", "<s>", "__start__", "__end__", "  ", "  "]:
            response = response.replace(symb, " ")
        utt2.append(response)
        da_batch.append(da)
        sent_batch.append(sent)
    conf_batch, pred_batch = model_da(utt1, utt2)
    for pred, da in zip(pred_batch, da_batch):
        gold_dialog_acts.append(int(da))
        pred_dialog_acts.append(int(pred) - 1)


fl = open(args.infile, 'r')
data = json.load(fl)
num_batches = len(data) // batch_size + int(len(data) % batch_size > 0)
for j in range(num_batches):
    utt1, utt2, da_batch, sent_batch = [], [], [], []
    cur_chunk = data[j*batch_size:(j+1)*batch_size]
    for history, gold_response, response, da, sent in cur_chunk:
        utt1.append(history[-1])
        for symb in ["</s>", "<s>", "__start__", "__end__", "  ", "  "]:
            response = response.replace(symb, " ")
        utt2.append(response)
        da_batch.append(da)
        sent_batch.append(sent)
    conf_batch, pred_batch = model_sent(utt1, utt2)
    for pred, sent in zip(pred_batch, sent_batch):
        gold_sentiment.append(int(sent))
        pred_sentiment.append(sentiment_label_dict[int(pred)])
            

print("dialog_acts balanced accuracy", balanced_accuracy_score(gold_dialog_acts, pred_dialog_acts))
print("dialog_acts accuracy", accuracy_score(gold_dialog_acts, pred_dialog_acts))
print("sentiment balanced accuracy", balanced_accuracy_score(gold_sentiment, pred_sentiment))
print("sentiment accuracy", accuracy_score(gold_sentiment, pred_sentiment))
