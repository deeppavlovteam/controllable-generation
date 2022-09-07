import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("-in", action="store", dest="infile")
args = parser.parse_args()

correct_i, correct_q, total_i, total_q = 0, 0, 0, 0

with open(args.infile, 'r') as fl:
    data = json.load(fl)

for history, gold_response, response, da, sent in data:
    if da == 0:
        if "?" not in response.lower():
            correct_i += 1
        total_i += 1
    if da == 1:
        if "?" in response.lower():
            correct_q += 1
        total_q += 1

accuracy = (correct_i + correct_q) / (total_i + total_q)

print(round(accuracy, 4))

