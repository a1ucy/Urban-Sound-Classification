# 使用卷积神经网络（CNN）或循环神经网络（RNN），搭建一个简单的模型，对音频数据进行分类，例如区分说话人的语音、城市声音等等，数据集方面，公开数据集很多，比如UrbanSound8K城市声音数据集等
import pandas as pd
import torch
# import torchaudio
from torch import nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import librosa
import librosa.feature

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
sound_csv_path = './sounds/UrbanSound8K.csv'
df = pd.read_csv(sound_csv_path)
num_classes = df['class'].unique().shape[0]

sample_rate = 32000
num_epochs = 50
batch_size = 128

class_name = ['siren', 'car_horn', 'gun_shot']
max_class = max(df['class'].value_counts())
for i in class_name:
    while df[df['class'] == i].shape[0] < max_class:
        count = df[df['class'] == i].shape[0]
        samples_to_draw = max_class - count
        samples_to_add = df[df['class'] == i].sample(n=min(samples_to_draw, count))
        df = pd.concat([df, samples_to_add])

df['path'] = '../sound/sounds/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
paths, labels = df['path'].tolist(), df['classID'].values.tolist()


""" visual sample from each class
samples = df.groupby('class').sample(1)
audio_samples, samples_label = samples['path'].tolist(), samples['class'].tolist()

def plot_waveform(waveform, sample_rate, label):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(label)
    plt.show()

def plot_specgram(waveform, sample_rate, label):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(label)

for i in range(len(audio_samples)):
    waveform, sample_rate = torchaudio.load(audio_samples[i])
    plot_waveform(waveform, sample_rate, samples_label[i])
    plot_specgram(waveform, sample_rate, samples_label[i])

"""

class SoundDataset(Dataset):
    def __init__(self, data, labels, target_rate):
        self.data = data
        self.labels = labels
        # self.transformation = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=sample_rate,
        #     n_fft=2048, # feature
        #     hop_length=512, # linear output
        #     n_mels=128, #
        #     normalized=True,
        # )
        self.target_rate = target_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.labels[idx]
        # waveform, sr = torchaudio.load(path, normalize=True)
        # # resample if sample_rate !=
        # if sr != self.target_rate:
        #     resampler = torchaudio.transforms.Resample(sr, self.target_rate)
        #     waveform = resampler(waveform)
        # # channel > 1, drop
        # if waveform.shape[0] > 1:
        #     waveform = torch.mean(waveform, dim=0, keepdim=True)
        # # pad or cut
        # if waveform.shape[1] >= self.target_rate:
        #     waveform = waveform[:, :self.target_rate]
        # else:
        #     waveform = nn.functional.pad(waveform, (0, self.target_rate - waveform.shape[1]))
        # waveform = self.transformation(waveform)  # (channel, n_mels, time)
        waveform, sr = librosa.load(path,sr=self.target_rate)
        # print(waveform.shape)
        if sr != self.target_rate:
            waveform = librosa.resample(y=waveform,orig_sr=sr, target_sr=self.target_rate)
        features = librosa.feature.melspectrogram(y=waveform,sr=self.target_rate, pad_mode='constant')
        features = librosa.power_to_db(features, ref=np.max)
        # print(features.shape)
        return torch.tensor(features).unsqueeze(0), torch.tensor(label)


X_train, X_test, y_train, y_test = train_test_split(paths, labels, test_size=0.2, random_state=42)

train_loader = DataLoader(SoundDataset(X_train, y_train, sample_rate), batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(SoundDataset(X_test, y_test, sample_rate), batch_size=batch_size,
                         shuffle=False)


class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(14336, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.ac = nn.ReLU()
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)


    def forward(self, input):
        x = self.pool(self.ac(self.bn1(self.conv1(input))))
        x = self.pool(self.ac(self.bn2(self.conv2(x))))
        x = self.pool(self.ac(self.bn3(self.conv3(x))))
        # print(x.shape)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.ac(self.fc1(x))
        y = self.fc2(x)
        # print(y.shape)
        return y

# model = models.resnet50().to(device)
model = CNN(num_classes).to(device)


# loss & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
best_loss = np.inf
patience = 5
patience_counter = 0

train_loss, test_loss, acc = [], [], []
# Training loop
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, desc='Train')
    total_train_loss = []
    for audio, labels in loop:
        # Forward pass
        audio = audio.to(device)
        labels = labels.to(device)
        outputs = model(audio)
        loss = loss_fn(outputs, labels)
        total_train_loss.append(loss.item())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test the model
    model.eval()
    total_val_loss = []
    all_labels = []
    all_predictions = []
    all_outputs = []
    with torch.no_grad():
        correct = 0
        total = 0
        loop2 = tqdm(test_loader, desc='Test')
        for audio, labels in loop2:
            audio = audio.to(device)
            labels = labels.to(device)
            outputs = model(audio)

            loss = loss_fn(outputs, labels)
            total_val_loss.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_outputs.extend(outputs.squeeze().cpu().numpy())

    logits_tensor = torch.tensor(np.array(all_outputs), dtype=torch.float32)
    prob = torch.softmax(logits_tensor, dim=1).numpy()

    avg_val_loss = sum(total_val_loss) / len(test_loader)
    avg_train_loss = sum(total_train_loss) / len(train_loader)
    accuracy = correct / total
    train_loss.append(avg_train_loss)
    test_loss.append(avg_val_loss)
    acc.append(accuracy)

    f1 = f1_score(all_labels, all_predictions, average='macro')
    auc = roc_auc_score(all_labels, prob, multi_class='ovo', average='macro')
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}, AUC: {auc:.4f}')

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Stopping early')
            sound_classes = {0:'air_conditioner',
            1: 'car_horn',
            2: 'children_playing',
            3: 'dog_bark',
            4: 'drilling',
            5: 'engine_idling',
            6: 'gun_shot',
            7: 'jackhammer',
            8: 'siren',
            9: 'street_music'}
            exam = []
            for i in range(len(all_predictions)):
                if all_predictions[i] != all_labels[i]:
                    exam.append(f"pred:{sound_classes[all_predictions[i]]}, label:{sound_classes[all_labels[i]]}")
            exam_df = pd.DataFrame(exam)
            print(exam_df.value_counts())
            break
    print(patience_counter, best_loss)
    torch.save(model.state_dict(), 'sound_model.pt')

plt.plot(train_loss)
plt.plot(test_loss)
plt.plot(acc)
plt.xlabel('Epoch')
plt.legend(['Train', 'Test', 'Acc'], loc='upper left')
plt.show()