# speech_emotion_detect_demo
Based on deep learning technology, this research aims to explore the feasibility of using this technology for emotion recognition in speech. For this type of data, which is a sequence of data in order, we combine two basic network structures, LSTM and CNN, and use attention mechanisms to optimize the results in the middle layer to achieve more ideal effects.

## Dataset preparation
Train dataset: [Emotional Voice Conversion: Theory, Databases and ESD](https://hltsingapore.github.io/ESD/index.html)

Each model is trained and evaluated using a dataset of five categories (sadness, happiness, neutrality, anger, and surprise), and the datasets are mixed together to evaluate whether the results can detect subtle differences in speech. In addition, the sampling rate of the data set is reduced to 8000 Hz. ​

## Model Design
```
EmotionClassifier(
  (lstm): LSTM(13, 64, batch_first=True)
  (cnn_layers): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
    (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
  )
  (fc_layers): Sequential(
    (0): Linear(in_features=185856, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=5, bias=True)
  )
)
```

## Quick start
Install dependencies
```
cd .
pip install -r requirements.txt
cd ./frontend
yarn
# unzip pre-trained model
cd ./model
unzip model.zip -O ./model.pth
```

Quick start
```
# frontend
cd frontend
yarn dev

# backend
uvicorn main:app
```
