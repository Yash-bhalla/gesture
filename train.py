import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class GestureDataset(Dataset):
    def __init__(self, root_dir, seq_length=20):
        self.data = []
        self.labels = []
        self.seq_length = seq_length

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)

            for sample in os.listdir(cls_path):
                sample_path = os.path.join(cls_path, sample)

                
                if os.path.isdir(sample_path):
                    self.data.append(sample_path)
                    self.labels.append(self.class_to_idx[cls])

        print("Classes:", self.classes)
        print("Total samples:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder = self.data[idx]
        label = self.labels[idx]

        frames = []

        for i in range(self.seq_length):
            img_path = os.path.join(folder, f"{i}.jpg")

            if not os.path.exists(img_path):
                
                img = np.zeros((64, 64, 3))
            else:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))

            img = img / 255.0
            frames.append(img)

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))  

        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x



class GestureModel(nn.Module):
    def __init__(self, num_classes):
        super(GestureModel, self).__init__()

        self.cnn = CNN()
        self.gru = nn.GRU(input_size=128, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.cnn(x)

        features = features.view(B, T, -1)

        out, _ = self.gru(features)
        out = out[:, -1, :]

        out = self.fc(out)
        return out


def train():
    device = torch.device("cpu")

    dataset = GestureDataset("dataset")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = GestureModel(num_classes=len(dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "model.pth")
    print(" Model saved as model.pth")



if __name__ == "__main__":
    train()