from tqdm import tqdm

from torchvision import transforms

from LinkNetModel import DenseLinkModel
from dataloader import PhaseContrastTrainValidLoader


# CSV FILES
INPUT_CSV = "../data/AugmentedData/train_input_imgs.csv"
LABEL_CSV = "../data/AugmentedData/train_label_imgs.csv"

# Parameters
input_channels = 1
num_classes = 3
pretrained_flag = True
img_size = 512

# Transform
mul_transf = [ transforms.Resize(size=(img_size, img_size)), transforms.ToTensor() ]

train_loader, valid_loader = PhaseContrastTrainValidLoader(INPUT_CSV, LABEL_CSV, input_chnls=1, data_transform=transforms.Compose(mul_transf), mode="train", validation_split=0.1, batch_sz=2, workers=0, data_root="../data/AugmentedData")

model = DenseLinkModel(input_channels, pretrained_flag, num_classes)

for input_img, label, contour_label, watershed_label in tqdm(train_loader, total=len(train_loader)):
    pass