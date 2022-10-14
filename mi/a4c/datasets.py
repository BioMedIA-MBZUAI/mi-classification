import torchvision
import torch
from torchvision import transforms
import csv


class HMC(torchvision.datasets.VisionDataset):
    def __init__(self, split="train", root=None, fold=1, seg=None, mean=None, std=None, sub=0):
        self.root = root
        self.split = split
        self.patients = []
        self.frames = []
        self.segs = []
        self.mean = mean
        self.std = std
        self.seg = seg

        if sub == 0:
            subfile = 'hmcfold.csv'
        else:
            subfile = 'hmcfoldm.csv'

        print(subfile)

        with open(subfile) as mfile:
            reader = csv.DictReader(mfile, delimiter=',')

            if fold is None:
                for row in reader:
                    self.patients.append(row['name'])
                    self.frames.append([int(row['start_frame']), int(row['end_frame'])])
                    segs = [int(row[col] == 'MI') for col in ['SEG1', 'SEG2', 'SEG3', 'SEG5', 'SEG6', 'SEG7']]
                    self.segs.append(segs)

            else:
                for row in reader:
                    if split == "train" and fold != int(row['Fold']):
                        self.patients.append(row['name'])
                        self.frames.append([int(row['start_frame']), int(row['end_frame'])])
                        segs = [int(row[col] == 'MI') for col in ['SEG1', 'SEG2', 'SEG3', 'SEG5', 'SEG6', 'SEG7']]
                        self.segs.append(segs)

                    elif split == "test" and fold == int(row['Fold']):
                        self.patients.append(row['name'])
                        self.frames.append([int(row['start_frame']), int(row['end_frame'])])
                        segs = [int(row[col] == 'MI') for col in ['SEG1', 'SEG2', 'SEG3', 'SEG5', 'SEG6', 'SEG7']]
                        self.segs.append(segs)

        print(len(self.patients), len(self.segs), len(self.frames))

    def __getitem__(self, index):
        name = self.patients[index]
        segs = self.segs[index]

        if self.seg is None:
            segs = max(segs)
        else:
            segs = segs[self.seg]

        clip = torch.load('cycle12/' + name + '.pt')

        if self.split == 'train':
            transform = transforms.Compose([
                transforms.RandomAffine(10, scale=(
                    0.8, 1.1), translate=(0.1, 0.1))
            ])
            clip = transform(clip)

        clip = clip.permute(1, 0, 2, 3)
        return clip, segs

    def __len__(self):
        return len(self.patients)
