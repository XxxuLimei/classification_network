import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

transform = transforms.Compose(
    [transforms.Resize((32, 32)),  # resize to adapt the network
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # beacuse the training process uses the normalization, the prediction must use this as well
)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('lenet.pth'))

im = Image.open('fly.jpg')  # (H, W, C)
im = transform(im)
im = torch.unsqueeze(im, dim=0)  # add the batch dimension, [N, C, H, W]

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()

print(classes[int(predict)])
