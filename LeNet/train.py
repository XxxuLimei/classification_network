import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. ready for dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=36, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=10000, shuffle=False, num_workers=0)

test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 2. ready for network, loss function and optimizer

net = LeNet()
loss_function  = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 3. start to train

for epoch in range(10):
    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        # get inputs: data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:
            with torch.no_grad():
                outputs = net(test_image)
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %(epoch+1, step+1, running_loss/500, accuracy))
                running_loss = 0.0

print('Finished Training!')

# 4. save the model

save_path = './lenet.pth'
torch.save(net.state_dict(), save_path)