import os
import sys
import torch
import pickle
import numpy as np
from cv2 import VideoCapture
from torchvision.transforms import Resize, Grayscale
from tqdm import tqdm
from scipy.ndimage import zoom

root = sys.argv[1]

os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

for split in ['train', 'val', 'test']:
    with open(split, 'rb') as f:
        vids = pickle.load(f)

    res = Resize((224, 224))
    gray = Grayscale()

    if split in ['train', 'val']:
        dst = 'data/train/'
    else:
        dst = 'data/test/'

    for i, v in tqdm(enumerate(vids)):
        f1, f2 = v[1], v[2] + 1
        length = f2 - f1
        cap = VideoCapture(root + v[0] + '.avi')
        video = np.zeros((length, 112, 112, 3))
        for f in range(f1, f2):
            cap.set(1, f)
            ret, frame = cap.read()
            video[f-f1, :, :, :] = frame

        video = zoom(video, (12/length, 1, 1, 1))
        video = np.clip(video, 0, 255)
        video = video/255
        video = torch.tensor(np.transpose(video, (0, 3, 1, 2)))

        try:
            video = gray(res(video)).float()
            torch.save(video, dst + v[0] + '.pt')
        except:
            print('Error in: ', v[0])
