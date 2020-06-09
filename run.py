import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


random_seed = 1
batch_size = 50
loss_list = []
epochs = 200


# 自定义数据集
class MyData(Dataset):
    def __init__(self):
        fh = open('./data.txt', 'r')
        points = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            point = line.split('***', 1)
            points.append((point[0], point[1]))
        self.points = points
        self.transform = transforms.Compose([transforms.ToTensor()])
        fh.close()

    def __getitem__(self, index):
        x, y = self.points[index]
        return torch.tensor([float(x), float(y)])

    def __len__(self):
        return len(self.points)


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, data, cur):
        losses = torch.abs(data[:, :] - cur[:]).sum()
        return losses


if __name__ == '__main__':
    torch.manual_seed(random_seed)
    result = torch.tensor([[0.0, 0.0]], requires_grad=True)
    train_data = MyData()
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    criterion = MyLoss()
    optimizer = optim.Adam([result], lr=0.0001)
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_index, data in enumerate(train_loader):
            inputs = data
            loss = criterion(inputs, result)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss)
        print(running_loss)
        loss_list.append(running_loss/2000.0)
    print(result)
    x1 = range(epochs)
    y1 = loss_list
    plt.plot(x1, y1, '-')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.savefig("loss.jpg")
    plt.show()

