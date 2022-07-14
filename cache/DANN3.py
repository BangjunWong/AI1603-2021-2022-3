import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MNISTDataset(datasets.VisionDataset):
    def __init__(self, root: str) -> None:
        super(MNISTDataset, self).__init__(root)
        self.data_label = torch.load(root)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images = []
        self.targets = []
        for image, target in self.data_label:
            self.images.append(np.array(image))
            self.targets.append(target)
        self.images = torch.tensor(np.array(self.images))
        self.targets = torch.tensor(np.array(self.targets))

    def __len__(self) -> int:
        return 20000

    def __getitem__(self, index):
        image, target = self.data_label[index]
        return self.transform(image), target


train1_dataset = MNISTDataset('./dataset/ColoredMNIST/train1.pt')
train1_dataloader = DataLoader(dataset=train1_dataset, batch_size=2000, shuffle=True)
train2_dataset = MNISTDataset('./dataset/ColoredMNIST/train2.pt')
train2_dataloader = DataLoader(dataset=train2_dataset, batch_size=2000, shuffle=True)
test_dataset = MNISTDataset('./dataset/ColoredMNIST/test.pt')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=2000, shuffle=True)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Linear(3 * 28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 36),
            nn.ReLU(True),
            nn.Linear(36, 18),
            nn.ReLU(True),
            nn.Linear(18, 9),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(True),
            nn.Linear(18, 36),
            nn.ReLU(True),
            nn.Linear(36, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = AutoEncoder().to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

total_epoch = 8
record_loss = []

for epoch in range(total_epoch):
    for group in [train1_dataloader, train2_dataloader]:
        for images, _ in group:
            images = images.to(device).reshape(-1, 3 * 28 * 28)
            outputs = model(images)
            loss = loss_function(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            record_loss.append(loss.item())
    for group in [test_dataloader]:
        for images, _ in group:
            images = images.to(device).reshape(-1, 3 * 28 * 28)
            outputs = model(images)
            loss = 10.0 * loss_function(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            record_loss.append(loss.item())
    print("Epoch: {}, Loss: {}".format(epoch, record_loss[-1]))

model.cpu()

inputs = train1_dataset.images.reshape((20000, -1)).to(torch.float32)
train1_x = model.encoder(inputs).detach().numpy()
train1_y = train1_dataset.targets.detach().numpy()

inputs = train2_dataset.images.reshape((20000, -1)).to(torch.float32)
train2_x = model.encoder(inputs).detach().numpy()
train2_y = train2_dataset.targets.detach().numpy()

train_x = np.concatenate((train1_x, train2_x), axis=0)
train_y = np.concatenate((train1_y, train2_y), axis=0)

inputs = test_dataset.images.reshape((20000, -1)).to(torch.float32)
test_x = model.encoder(inputs).detach().numpy()
test_y = test_dataset.targets.detach().numpy()

depths = np.linspace(1, 50, 50, dtype=int)
results = []

for depth in depths:
    classifier = RandomForestClassifier(n_estimators=depth, n_jobs=4)
    classifier.fit(train_x, train_y)
    predict_y = classifier.predict(test_x)
    accuracy = metrics.accuracy_score(predict_y, test_y)
    results.append(accuracy)
    print("Depth: {}, Accuracy: {}".format(depth, accuracy))

plt.figure(1, figsize=(7, 7))
plt.plot(depths, results)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()