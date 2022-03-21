import torch

import torchvision


def scheduler(epoch, lr):
  lr0=1e-3
  if epoch >=81 and epoch <121:
    return lr0*1e-1
  elif epoch >=121 and epoch <161:
    return lr0*1e-2
  elif epoch >=161 and epoch <181:
    return lr0*1e-3
  elif epoch >=181:
    return lr0*0.5*1e-3
  else:
    return lr


batch_size = 64

transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(48),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), ( 0.5))])

train_dataset = torchvision.datasets.ImageFolder(root="data/train", transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root="data/test", transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

import torch.optim as optim
import torch.nn as nn


from EmotionNetNano.model import EmotionNano

net = EmotionNano(7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

epochs = 10

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100  == 99:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')

print('Finished Training')