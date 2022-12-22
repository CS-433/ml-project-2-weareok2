import numpy as np
import pickle, sys
import torch
from _transf import *


Lmax = 60
outputF = sys.argv[1]
model = sys.argv[2]


device = torch.device('cpu')

with open("vocab_full.pkl", "rb") as f:
    vocab = pickle.load(f)
Ldict = len(vocab)+1

f = "../twitter-datasets/test_data.txt"

twittDatasetTest = TwittDataset("test", vocab, Ldict, f, Lmax=Lmax)
dataloaderTest = torch.utils.data.DataLoader(
    twittDatasetTest,
    batch_size=1000,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    num_workers=2)
    
transfModel = torch.load(model, map_location=device)

ys = []
transfModel.train(False)
with torch.no_grad():
    for n, data in enumerate(dataloaderTest):
        print(n/len(dataloaderTest))
        xs, _ = data
        xs = xs.to(device)
        prediction = transfModel(xs).squeeze().to('cpu')
        prediction = (prediction>0)*2-1
        ys += prediction.tolist()


output = "Id,Prediction\n"
for i,y in enumerate(ys):
    output += "{},{}\n".format(i+1, y)
    
with open(outputF, 'wt') as f:
    f.write(output)
    
