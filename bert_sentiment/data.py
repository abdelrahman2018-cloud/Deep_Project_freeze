"""This module defines a configurable SSTDataset class."""

#import pytreebank
import torch
from loguru import logger
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset

#MY ADDITION START
import os
from torchtext.datasets import IMDB

train_iter = IMDB(split='train')
test_iter = IMDB(split='test')
#path_test_pos = "/home/aabuzaid/Desktop/bert-sentiment-master/bert-sentiment-master/aclImdb/test/pos"
#path_test_neg = "/home/aabuzaid/Desktop/bert-sentiment-master/bert-sentiment-master/aclImdb/test/neg"
#training_path = "/home/aabuzaid/Desktop/bert-sentiment-master/bert-sentiment-master/aclImdb/train"

#MY ADDITION END

logger.info("Loading the tokenizer")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#logger.info("Loading SST")
#sst = pytreebank.load_sst()


def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)


def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label == 'pos':
        return 1
    elif label == 'neg':
        return 0
    raise ValueError("Invalid label")


class SSTDataset(Dataset):
    """Configurable SST Dataset.
    
    Things we can configure:
        - split (train / val / test)
        - root / all nodes
        - binary / fine-grained
    """

    def __init__(self, split="train", root=True, binary=True):
        """Initializes the dataset with given configuration.

        Args:
            split: str
                Dataset split, one of [train, val, test]
            root: bool
                If true, only use root nodes. Else, use all nodes.
            binary: bool
                If true, use binary labels. Else, use fine-grained.
        """
        logger.info(f"Loading IMDB {split} set")
        #self.sst = sst[split]
        if(split == "train"):
            iterr = IMDB(split='train')
        else:   
            iterr = IMDB(split='test') 
        
        logger.info("Tokenizing")
        if root and binary:
            #MY ADDITION START
            self.data = [
                (
                    rpad(
                        #tokenizer.encode("[CLS] " + open(path + "/" + file, 'r').read() + " [SEP]"), n=66
                        tokenizer.encode("[CLS] " + line[:511] + " [SEP]"), n=66
                    ),
                    get_binary_label(label),
                )
                for label, line in iterr
                if(len(line) <= 512)
            ]
        else:
            print("Error")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        X = torch.tensor(X)
        return X, y
