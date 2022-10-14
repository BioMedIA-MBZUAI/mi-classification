import torch
import os
import sys
import cv2
import csv
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, Grayscale, CenterCrop
from torch.nn.functional import interpolate

src = sys.argv[1]
dst = 'cycle12/'

os.makedirs('cycle12', exist_ok=True)

with open('hmcfold.csv') as mfile:

    reader = csv.DictReader(mfile, delimiter=',')
    crop = CenterCrop((170, 170))
    res = Resize((224, 224))
    gray = Grayscale()

    for row in reader:
        name = row['name']
        start = int(row['start_frame'])
        end = int(row['end_frame'])
        length = end - start
        frames = []
        cap = cv2.VideoCapture(src + name + '.avi')

        if 'ES00071' in name:
            step = 2
            length -= 1
        else:
            step = 1

        for i in range(0, length, step):
            cap.set(1, start+i-1)
            ret, frame = cap.read()
            frame = Image.fromarray(frame).convert('L').resize((224, 224))
            frame = np.array(frame)
            frames.append(frame)

        clip = np.stack(frames) / 255
        clip = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)
        clip = res(crop(clip)).unsqueeze(0)
        clip = interpolate(clip, size=(12, 224, 224), mode='trilinear')[
            0].permute(1, 0, 2, 3)
        print(clip.shape, clip.min(), clip.max())

        torch.save(clip, dst + name + '.pt')
