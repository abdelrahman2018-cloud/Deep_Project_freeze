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
    if label < 2:
        return 0
    if label > 2:
        return 1
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
        logger.info(f"Loading iMDB {split} set")
        #self.sst = sst[split]
        if(split == "train") iter = train_iter
        else iter = test_iter
        
        
        logger.info("Tokenizing")
        if root and binary:
            #MY ADDITION START

            self.data = [
                (
                    rpad(
                        #tokenizer.encode("[CLS] " + open(path + "/" + file, 'r').read() + " [SEP]"), n=66
                        tokenizer.encode("[CLS] " + line + " [SEP]"), n=66
                    ),
                    label,
                )
                for label, line in iter
#                for file in os.listdir(path_test_pos) 
#                open(path + "/" + file, 'r') as f:
#                        word = f.read()

#                (
#                    rpad(
#                        tokenizer.encode("[CLS] " + open(path + "/" + file, 'r').read() + " [SEP]"), n=66
#                    ),
#                    0,
#                )
#                for file in os.listdir(path_test_neg)
#                open(path + "/" + file, 'r') as f:
#                        word = f.read()
            #MY ADDITION END
            ]
"""        elif root and not binary:
            self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66
                    ),
                    tree.label,
                )
                for tree in self.sst
            ]
        elif not root and not binary:
            self.data = [
                (rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66), label)
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
            ]
        else:
            self.data = [
                (
                    rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66),
                    get_binary_label(label),
                )
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
                if label != 2
            ]
            """
        else print("Invalid  run, not yet ready")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        X = torch.tensor(X)
        return X, y
