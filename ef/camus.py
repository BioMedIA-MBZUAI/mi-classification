import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torchvision.transforms import Resize
from torch.utils.data import Dataset
from torch.nn.functional import interpolate

# Parts of this code are based on https://www.kaggle.com/code/sontungtran/camus-eda


class Camus(Dataset):
    def __init__(
        self,
        data_type='train',
        root=None
    ):
        super(Camus, self).__init__()

        train_file = os.path.join(root, 'training')
        test_file = os.path.join(root, 'testing')

        if data_type == 'train' or data_type == "val":
            data_file = train_file
        elif data_type == 'test':
            data_file = test_file
        else:
            raise Exception('Wrong data_type for CamusIterator')

        self.data_type = data_type
        self.data_file = data_file
        self.patients = []

        for patient in os.listdir(self.data_file):
            self.patients.append(patient)

        if self.data_type == "train":
            self.patients = self.patients[0:450]
        elif self.data_type == "val":
            self.patients = self.patients[450:500]

    def __read_info(self, data_file):
        info = {}
        with open(data_file, 'r') as f:
            for line in f.readlines():
                info_type, info_details = line.strip('\n').split(': ')
                info[info_type] = info_details
        return info

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        patient_file = self.patients[index]
        image_file = '{}/{}/{}'.format(self.data_file,
                                       patient_file, patient_file+'_4CH_sequence.mhd')
        image_4CH_sequence = sitk.GetArrayFromImage(
            sitk.ReadImage(image_file, sitk.sitkFloat32))
        info_4CH = self.__read_info(
            '{}/{}/{}'.format(self.data_file, patient_file, 'Info_4CH.cfg'))

        if self.data_type == 'train' or self.data_type == "val":
            data = {
                'patient': patient_file,
                '4CH_sequence': image_4CH_sequence,
                'info_4CH': info_4CH}
        elif self.data_type == 'test':
            data = {
                'patient': patient_file,
                '4CH_sequence': image_4CH_sequence,
                'info_4CH': info_4CH}

        return data

    def __iter__(self):
        for i in range(len(self)):
            try:
                yield self[i]
            except:
                pass


root = sys.argv[1]

os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

train_ds = Camus(data_type='train', root=root)
val_ds = Camus(data_type='val', root=root)

print(len(train_ds), len(val_ds))

transform = Resize((224, 224))

troot = 'data/train/'
vroot = 'data/test/'

for x in tqdm(train_ds):
    patient = x['patient'] + '.pt'
    ef = x['info_4CH']['LVef']
    img = x["4CH_sequence"]
    img = torch.tensor(img)
    img = transform(img).unsqueeze(0).unsqueeze(0)
    img = interpolate(img, scale_factor=(
        12/img.shape[2], 1, 1), mode='trilinear')
    img = (img/255).float()
    img = img[0].permute(1, 0, 2, 3)
    torch.save(img, troot + patient)

for x in tqdm(val_ds):
    patient = x['patient'] + '.pt'
    ef = x['info_4CH']['LVef']
    img = x["4CH_sequence"]
    img = torch.tensor(img)
    img = transform(img).unsqueeze(0).unsqueeze(0)
    img = interpolate(img, scale_factor=(
        12/img.shape[2], 1, 1), mode='trilinear')
    img = (img/255).float()
    img = img[0].permute(1, 0, 2, 3)
    torch.save(img, vroot + patient)
