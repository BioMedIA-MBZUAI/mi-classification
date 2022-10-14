import os
import torchvision
from torchvision import transforms
import torch
import pandas as pd


class HMC(torchvision.datasets.VisionDataset):

    def __init__(self, split="train", fold=1):
        self.split = split
        self.patients = []
        self.fractions = []

        if split == 'train':
            self.root = "data/train/"
        else:
            self.root = "data/test/"

        df = pd.read_csv('FileList.csv')

        for f in os.listdir(self.root):
            row = df.loc[df['FileName'] == f.split('.')[0]]
            self.patients.append(self.root + f)
            self.fractions.append(row['EF'].item())

        print(len(self.patients), len(self.fractions))

    def __getitem__(self, index):
        name = self.patients[index]
        ef = self.fractions[index]

        clip = torch.load(name)
        clip = clip.permute(1, 0, 2, 3)

        return clip, ef

    def __len__(self):
        return len(self.patients)
