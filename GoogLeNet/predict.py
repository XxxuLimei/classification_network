import torch
from model import GoogLeNet
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
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indice
try:
    json_file = open("./class_indices.json", 'r')
    class_indice = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = GoogLeNet(num_classes=5, aux_logits=False)
# load model weights
model_weights_path = "./GoogLeNet.pth"
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weights_path), strict=False)
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_class = torch.argmax(predict).numpy()
print(class_indice[str(predict_class)], predict[predict_class].item())
plt.show()