import torch
from helper import load_model
from LinkNetModel import DenseLinkModel
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


# Image Path
IMG_PATH = "../../example_imgs/ex1.jpg"

# Binary Path
BIN_PATH  = "../../binaries/dense_linknet_512_sgd_bce.pt"

# HyperParameter
img_size=512

tsfm = transforms.Compose([transforms.Resize((img_size, img_size)) , transforms.ToTensor()])

img = Image.open(IMG_PATH).convert("L")
img = tsfm(img)
img = img.unsqueeze(dim=0)
segm_model = DenseLinkModel(input_channels=1, num_classes=3)
segm_model = load_model(segm_model, model_dir=BIN_PATH)


out = segm_model(img)
out = out.squeeze(dim=0)
out = out.detach().numpy()

plt.imshow(out[0, :, :])
plt.show()

plt.imshow(out[1, :, :])
plt.show()

plt.imshow(out[2, :, :])
plt.show()