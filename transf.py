import numpy as np
import pickle, sys
import torch
from _transf import *

H = int(sys.argv[1])
nLayer = int(sys.argv[2])
nHead = int(sys.argv[3])

createNewDataset = True
fullDataset = True  # False to use the reduced dataset
Lmax = 60

lr = float(sys.argv[4])*10**-4
Nmini = int(sys.argv[5])
NminiTest = 2000
iterMax = 15

outputF = sys.argv[6]


device = torch.device('cuda')

with open("vocab_full.pkl", "rb") as f:
    vocab = pickle.load(f)
Ldict = len(vocab)+1

            
if createNewDataset:
    fP = "../twitter-datasets/train_pos_full.txt" if fullDataset else "../twitter-datasets/train_pos.txt"
    fN = "../twitter-datasets/train_neg_full.txt" if fullDataset else "../twitter-datasets/train_neg.txt"
    twittDatasetTr = TwittDataset("train", vocab, Ldict, fP, fN, Lmax)
    twittDatasetTest = TwittDataset("eval", vocab, Ldict, fP, fN, Lmax)
    dataloaderTrain = torch.utils.data.DataLoader(
        twittDatasetTr,
        batch_size=Nmini,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=3)
    dataloaderTest = torch.utils.data.DataLoader(
        twittDatasetTest,
        batch_size=NminiTest,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=3)
    torch.save(dataloaderTrain, "dataloaderTrain.pt")
    torch.save(dataloaderTest, "dataloaderTest.pt")
else:
    dataloaderTrain = torch.load("dataloaderTrain.pt")
    dataloaderTest = torch.load("dataloaderTest.pt")

    
transfModel = TransfModel(H, Ldict, Lmax, nLayer, nHead, device)
transfModel = transfModel.to(device)
optimiser = torch.optim.Adam(transfModel.parameters(), lr=lr)


for t in range(iterMax):
    transfModel.train(True)
    for n, data in enumerate(dataloaderTrain):
        xs, ys = data
        xs = xs.to(device)
        ys = ys.to(device)
        prediction = transfModel(xs)
        loss = TransfModel.loss(prediction, ys)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        
        if not n%(len(dataloaderTrain)//5):
            with torch.no_grad():
                transfModel.train(False)
                losses, accs = [], []
                acc = TransfModel.accuracy(prediction, ys)
                for data in dataloaderTest:
                    xs, ys = data
                    xs = xs.to(device)
                    ys = ys.to(device)
                    prediction = transfModel(xs)
                    losses.append(TransfModel.loss(prediction, ys).item())
                    accs.append(TransfModel.accuracy(prediction, ys))
            transfModel.train(True)
            
            print("{:.1f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(t+n/len(dataloaderTrain), loss.item(),
                                                                  acc, np.mean(losses), np.mean(accs)))
    if t>1 and not t%3:
        torch.save(transfModel, outputF+"-ep{}".format(t))

torch.save(transfModel, outputF)
