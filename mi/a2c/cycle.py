import torch
import os
import sys
import cv2
import csv
import numpy as np
from torchvision.transforms import Resize, Grayscale, CenterCrop
from torch.nn.functional import interpolate

src = sys.argv[1]
dst = 'cycle12/'

os.makedirs('cycle12', exist_ok=True)

tlist = []

with open('hmcfold.csv') as mfile:

    reader = csv.DictReader(mfile, delimiter=',')
    crop = CenterCrop((185, 185))
    res = Resize((224, 224))
    gray = Grayscale()

    for row in reader:

        start = int(row['start_frame']) - 1
        end = int(row['end_frame'])
        name = row['name']
        print(name, start, end)

        cap = cv2.VideoCapture(src + name + '.avi')
        cap.set(0, start)

        frames = []

        for i in range(end-start):
            ret, frame = cap.read()
            frames.append(frame)

        clip = np.stack(frames) / 255
        clip = torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2)
        clip = res(gray(clip))
        clip = res((crop(clip)))
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
        clip = interpolate(clip, size=(12, 224, 224), mode='trilinear')[
            0].permute(1, 0, 2, 3)
        print(clip.shape, clip.min(), clip.max())

        torch.save(clip, dst + name + '.pt')
