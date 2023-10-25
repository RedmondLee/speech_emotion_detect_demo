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
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 尽量使用GPU

# 定义情绪标签映射
emotion_mapping = {
    'Neutral': 0,
    'Angry': 1,
    'Happy': 2,
    'Sad': 3,
    'Surprise': 4
}


# 定义模型
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=13, hidden_size=64, num_layers=1, batch_first=True)  # 设置 batch_first 为 True

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 121 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = x.squeeze(1)
        x, _ = self.lstm(x)
        x = x.unsqueeze(1)  # 添加维度以适应 CNN 输入要求
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 数据集处理
class EmotionDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        self.audio_files = []
        self.labels = []

        for subdir in os.listdir(data_folder):
            subdir_path = os.path.join(data_folder, subdir)
            if not os.path.isdir(subdir_path):
                continue  # 确保它是一个文件夹
            for emotion, label in emotion_mapping.items():
                emotion_folder = os.path.join(subdir_path, emotion)
                if not os.path.exists(emotion_folder):
                    continue
                files = os.listdir(emotion_folder)
                for file in files:
                    self.audio_files.append(os.path.join(emotion_folder, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        sample, label = self.load_sample(self.audio_files[idx], self.labels[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def load_sample(self, file_path, label):
        # 读取音频文件并提取特征
        sample_rate, audio = wavfile.read(file_path)
        features = speechpy.feature.mfcc(audio, sample_rate)


        # 将特征转换为[500, 39]的形状
        if features.shape[0] < 500:
            pad_width = 500 - features.shape[0]
            features = np.pad(features, pad_width=((0, pad_width), (0, 0)), mode='constant')
        else:
            features = features[:500, :]

        return features, label




    # 训练函数


def train(model, train_loader,test_loader, criterion, optimizer, num_epochs):
    model.to(Device)
    criterion = criterion.to(Device)
    model.train()
    train_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        train_loss_epoch = 0
        test_loss_epoch = 0
        train_corrects = 0
        test_corrects = 0

        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Device), labels.to(Device)
            # debug
            print(inputs.shape, inputs.dtype)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            pre_lab = torch.argmax(outputs, 1)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * inputs.size(0)
            train_corrects += torch.sum(pre_lab == (labels).data)
        ## 计算一个epoch的损失和精度
        train_loss_epoch /= len(train_loader.dataset)
        train_loss.append(train_loss_epoch)
        train_acc = train_corrects.double() / len(train_loader.dataset)
        print('Train epoch {}\t Loss {:.6f} \t Acc {:.6f}'.format(epoch + 1, train_loss_epoch , train_acc.item()))

        ## 计算在测试集上的表现
        model.eval()
        for inputs_test, labels_test in test_loader:
            inputs_test, labels_test = inputs_test.to(Device), labels_test.to(Device)
            output = model(inputs_test.float())
            loss = criterion(output, labels_test)
            pre_lab = torch.argmax(output, 1)
            test_loss_epoch += loss.item() * inputs_test.size(0)
            test_corrects += torch.sum(pre_lab == (labels_test).data)
        ## 计算一个epoch的损失和精度
        test_loss_epoch /= len(test_loader.dataset)
        test_loss.append(test_loss_epoch)
        test_acc = test_corrects.double() / len(test_loader.dataset)
        print('Test epoch {}\t Loss {:.6f} \t Acc {:.6f}'.format(epoch + 1, test_loss_epoch , test_acc.item()))
        torch.save(model.state_dict(), f'model_{epoch}.pth')

    torch.save(model.state_dict(), 'model.pth')
    return train_loss, test_loss


# 主函数
if __name__ == '__main__':
    # 数据集路径
    data_dir = r"C:\Users\WEN\Downloads\fxf\Emotion Speech Datasets"

    # 创建数据集
    dataset = EmotionDataset(data_dir)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    # # 划分训练集和测试集

    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    model = EmotionClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    train_loss, test_loss = train(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs)

    # 绘制训练损失
    plt.plot(range(1, num_epochs + 1), train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Time')
    plt.show()
    # 绘制测试损失
    plt.plot(range(1, num_epochs + 1), test_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Over Time')
    plt.show()




