from torch.utils.data import Dataset
import numpy as np
from utils import data_utils

class ARDataset(Dataset):
    def __init__(self, input_dct_seq, output_dct_seq, all_seq):
        self.input_dct_seq = input_dct_seq
        self.output_dct_seq = output_dct_seq
        self.all_seqs = all_seq

    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]
