import numpy as np
import torch


class TwittDataset(torch.utils.data.Dataset):
    """
    Class to deal with the twitter dataset. Takes the raw files and transforms them into ready-to-use inputs for the model.
    """
    def __init__(self, mode, vocab, Ldict, positiveExampleFile="", negativeExampleFile="", Lmax=60):
        """
        mode: str in ["train", "eval", "test"]
        vocab: dict, dictionary of the tokens of the text ; as obtained by build_vocab.sh and cut_vocab.sh
        Ldict: int, nbr of possible tokens to consider ; len(vocab) plus special tokens
        positiveExampleFile: str, positive-tweet file of the dataset, or test dataset
        negativeExampleFile: str, negative-tweet file of the dataset
        Lmax: int, cutoff token-length of a sequence
        """
        assert mode in ["train", "eval", "test"]
        super().__init__()
        self.Ldict = Ldict
        self.Lmax = Lmax
        self.buildDatasets(vocab, mode, positiveExampleFile, negativeExampleFile)
        
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return (self.xs[idx], self.ys[idx])
        
    def buildEncodings(file, vocab, Ldict, Lmax, testData=False):
        """
        file: str, file containing the dataset
        Ldict: int, nbr of possible tokens to consider ; len(vocab) plus special tokens
        Lmax: int, cutoff token-length of a sequence
        paddingId: int, id used for padding to Lmax ; must be positive, not in the vocabulary
        testData: bool, True if used for predicting (the input file has a different format)
        """            
        xs = []
        Ls = []
        with open(file) as f:
            for line in f:
                if testData:
                    iComma = line.find(',')
                    line = line[iComma+1:]
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [0]+[t+1 for t in tokens if t >= 0]
                Ls.append(len(tokens))
                xs.append(tokens)
        print("The longuest sequence is {} token long".format(max(Ls)))
        print("The dataset contains {} examples".format(len(xs)))
        print("###")
        
        paddingId = Ldict
        for i,x in enumerate(xs):
            if len(x)<Lmax:
                xs[i] = x+[paddingId]*(Lmax-len(x))
            else:
                xs[i] = x[:Lmax]

        return torch.tensor(xs)
    
    def buildDatasets(self, vocab, mode, fileP="", fileM=""):
        """
        Transforms a text dataset to numbered tokens ready-to-use for the model.
        
        vocab: dict, dictionary of the tokens of the text ; as obtained by build_vocab.sh and cut_vocab.sh
        mode: str in ["train", "eval", "test"]
        fileP: str, positive-tweet file of the dataset, or test dataset
        fileN: str, negative-tweet file of the dataset
        """
        if mode=="test":
            self.xs = TwittDataset.buildEncodings(fileP, vocab, self.Ldict, self.Lmax, True)
            self.ys = torch.zeros(self.xs.shape)
        else:
            xsP = TwittDataset.buildEncodings(fileP, vocab, self.Ldict, self.Lmax)
            xsM = TwittDataset.buildEncodings(fileM, vocab, self.Ldict, self.Lmax)

            NP, NM = len(xsP), len(xsM)
            NtestP, NtestM = NP//20, NM//20

            if mode=="train":
                self.xs = torch.vstack([xsP[NtestP:], xsM[NtestM:]])
                self.ys = torch.zeros((NP-NtestP+NM-NtestM, 1))
                self.ys[:NP-NtestP] = 1
            elif mode=="eval":
                self.xs = torch.vstack([xsP[:NtestP], xsM[:NtestM]])
                self.ys = torch.zeros((NtestP+NtestM, 1))
                self.ys[:NtestP] = 1


class TransfModel(torch.nn.Module):
    """
    Class of the model for classifying tweets.
    """
    def __init__(self, H, Ldict, L, nLayer=2, nHead=5, device='cpu'):
        """
        H: int, feature dimension
        Ldict: int, length of the dictionnary
        L: int, length of the sequences
        nLayer: int, number of attention layers
        nHead: int, number of head per attention layer
        """
        super().__init__()
        self.paddingId = Ldict
        self.encoding = torch.nn.Embedding(Ldict+1, H, self.paddingId)
        self.positionalEncoding = torch.nn.Embedding(L, H)
        self.positionalEncoding_positions = torch.arange(L, device=device).unsqueeze(0)
        attLayer = torch.nn.TransformerEncoderLayer(H, nHead, 2*H, batch_first=True)
        self.transfEncoder = torch.nn.TransformerEncoder(attLayer, nLayer)
        self.output = torch.nn.Linear(H, 1)
        
    def forward(self, x):
        """
        x: tensor of dimension (N batch, L, H)
        """
        x = self.encoding(x)+self.positionalEncoding(self.positionalEncoding_positions)
        x = self.transfEncoder(x)
        return self.output(x[:,0,:])
    
    def loss(x, y):
        """
        Binary cross-entropy loss.
        
        x: tensor of dimension (N batch, 1), logit
        y: tensor of dimension (N batch, 1), ±1
        """
        return torch.nn.functional.binary_cross_entropy_with_logits(x, y)
    
    def accuracy(x, y):
        """
        x: tensor of dimension (N batch, 1), logit
        y: tensor of dimension (N batch, 1), ±1
        """
        return (torch.mean(((x>0)*2-1)*(2*y-1)).item()+1)/2
        return (torch.mean(((x>0)*2-1)*(2*y-1)).item()+1)/2
    
