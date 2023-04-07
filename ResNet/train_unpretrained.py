import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model import resnet34, resnet101
# using this file to download the pretrained file
import torchvision.models.resnet

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(225),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

data_root = os.path.join(os.getcwd(), "data/")

train_dataset = datasets.ImageFolder(root=data_root+"train", transform=data_transform["train"])
train_num = len(train_dataset)

flower_list = train_dataset.class_to_idx
class_dict = dict((val, key) for key, val in flower_list.items())

json_str = json.dumps(class_dict, indent=4)
with open("class_indices.json", 'w') as json_file:
    json_file.write(json_str)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

validate_dataset = datasets.ImageFolder(root=data_root+"val", transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# # here we don't initialize the parameters
# net = resnet34()
# # load the pretrained model weights
# model_weight_path = "./resnet34-b627a593.pth"
# missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
# inchannel = net.fc.in_features
# net.fc = nn.Linear(inchannel, 5)
# # when modified the layer, remember to put it on the cuda device
net = resnet34(num_classes=5)
net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = "./ResNet34.pth"

for epoch in range(30):
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        rate = (step+1) / len(train_loader)
        a = "*" * int(rate*50)
        b = "-" * int((1 - rate)*50)
        print("\r train loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accuracy_test = acc / val_num
        if accuracy_test > best_acc:
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train loss: %.3f test_accuracy: %.3f' %(epoch+1, running_loss/step, accuracy_test))

print("Finished Training!")