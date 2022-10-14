# End-to-End Myocardial Infarction Classification from Echocardiographic Scans

### Published in ASMSUS 2022 [[Paper](https://link.springer.com/content/pdf/10.1007/978-3-031-16902-1_6.pdf)]

This repository contains a PyTorch implementation of our method used in the paper above for MI classification from echocardiography videos using models pretrained on ejection fraction prediction.

### Installation

Clone this repository and enter the directory:
```bash
git clone https://github.com/BioMedIA-MBZUAI/mi-classification.git
cd mi-classification
```

The code is implemented for Python 3.8.10.

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

### Data

#### EchoNet-Dynamic (EF Pretraining)
1. Download the dataset from [EchoNet-Dynamic website](https://echonet.github.io/dynamic/index.html#dataset)
2. Run the following to extract one cardiac cycle from each video, preprocess and store it as a tensor:

```
cd ef
python3 echo.py path/to/EchoNet/Videos
```

#### CAMUS (EF Pretraining)
1. Download the dataset from [CAMUS challenge website](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html)
2. Run the following to extract one cardiac cycle from each video, preprocess and store it as a tensor:

```
cd ef
python3 camus.py path/to/CAMUS/files
```

#### HMC-QU (MI Classification)
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset)
2. Run the following to extract one cardiac cycle from each video, preprocess and store it as a tensor:

##### A4C

```
cd mi/a4c
python3 cycle.py /path/to/A4C/videos/
```

##### A2C

```
cd mi/a2c
python3 cycle.py /path/to/A2C/videos/
```


### Ejection Fraction Pretraining

```
cd mi/ef
python3 train.py
```

### MI Classification

##### A4C

```
cd mi/a4c
python3 train.py --pretrained
```

##### A2C

```
cd mi/a2c
python3 train.py --pretrained
```

### Citation

```bash
@inproceedings{saeed2022end,
  title={End-to-End Myocardial Infarction Classification from Echocardiographic Scans},
  author={Saeed, Mohamed and Yaqub, Mohammad},
  booktitle={International Workshop on Advances in Simplifying Medical Ultrasound},
  pages={54--63},
  year={2022},
  organization={Springer}
}
```
