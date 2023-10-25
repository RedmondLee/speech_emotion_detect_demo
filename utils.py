import io
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.io import wavfile
import speechpy
import math
from train import EmotionClassifier

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(path):
    sample_rate, audio = wavfile.read(path)
    features = speechpy.feature.mfcc(audio, sample_rate)
    if features.shape[0] < 500:
        pad_width = 500 - features.shape[0]
        features = np.pad(features, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:500, :]

    return np.array([features,]).astype(np.float32)

def load_module():
    model = EmotionClassifier()
    model.load_state_dict(torch.load("model/model.pth"))
    model.to(Device)
    return model

def bytes_to_tensor(fp):
    return torch.from_numpy(load_data(fp)).to(Device)

def post_result(arr: list):
    idx = 0 
    carry = -math.inf
    for i, t in enumerate(arr[0]):
        if t > carry:
            idx = i 
            carry = t
    return idx