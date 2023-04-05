import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# load image
img = Image.open("./fly.jpg")
plt.imshow(img)
plt.show()

img = data_transform(img)
img = torch.unsqueeze(img, dim=0) # [B, C, H, W]

# read class_indices
try:
    json_file = open('./class_indices.json', 'r')
    class_indice = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = "./Alexnet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval() # turn off the dropout mode
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img)) # without batch_size dimension
    predict = torch.softmax(output, dim=0) # turn the output to probability distribution
    predict_class = torch.argmax(predict).numpy() # get the highest probability's index
print(class_indice[str(predict_class)], predict[predict_class].item()) # print the class label and its predict probability