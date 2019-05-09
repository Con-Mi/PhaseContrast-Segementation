import torch 
from torch import nn
from torch import optim
from torchvision import transforms

from LinkNetModel import DenseLinkModel
from helper import jaccard, dice, save_model
from dataloader import PhaseContrastTrainValidLoader

import time
import copy
from tqdm import tqdm


# GPU FLAG
use_cuda = torch.cuda.is_available()

# CSV FILES
INPUT_CSV = "../data/AugmentedData/train_input_imgs.csv"
LABEL_CSV = "../data/AugmentedData/train_label_imgs.csv"

# Hyperparameters
batch_size = 16
nr_epochs = 20
momentum = 0.93
lr_rate = 0.035
milestones = [ 7, 13, 18, 25, 30, 35, 41, 46, 48 ]
img_size = 512
gamma = 0.5
pretrained_flag = True
num_classes = 3
input_channels=1

segm_model = DenseLinkModel(input_channels, pretrained_flag, num_classes)
if use_cuda:
    segm_model.cuda()
segm_model = nn.DataParallel(segm_model)

mul_transf = [ transforms.Resize(size=(img_size, img_size)), transforms.ToTensor() ]

optimizerSGD = optim.SGD(segm_model.parameters(), lr=lr_rate, momentum=momentum)
criterion = nn.BCEWithLogitsLoss().cuda() if use_cuda else nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.MultiStepLR(optimizerSGD, milestones=milestones, gamma=gamma)

train_loader, valid_loader = PhaseContrastTrainValidLoader(INPUT_CSV, LABEL_CSV, input_chnls=1, data_transform=transforms.Compose(mul_transf), mode="train", validation_split=0.1, batch_sz=batch_size, workers=0, data_root="../data/AugmentedData")

dict_loaders = {"train":train_loader, "valid":valid_loader}

def train_model(cust_model, dataloaders, criterion, optimizer, num_epochs, scheduler=None):
    start_time = time.time()
    val_acc_history = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(cust_model.state_dict())

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("_"*15)
        for phase in ["train", "valid"]:
            if phase == "train":
                cust_model.train()
            if phase == "valid":
                cust_model.eval()
            running_loss = 0.0
            jaccard_acc = 0.0
            label_jaccard_acc = 0.0
            watersh_jaccard_acc = 0.0
            contour_jaccard_acc = 0.0
            # dice_loss = 0.0

            for inputs in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                input_img = inputs[0].cuda() if use_cuda else inputs[0]
                labels = inputs[1].cuda() if use_cuda else inputs[1]
                contour_labels = inputs[2].cuda() if use_cuda else inputs[2]
                watershed_labels = inputs[3].cuda() if use_cuda else inputs[3]

                out = torch.cat([labels, contour_labels, watershed_labels], dim=1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    
                    preds = cust_model(input_img)
                    loss = criterion(preds, out)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input_img.size(0)
                jaccard_acc += jaccard(out, torch.sigmoid(preds))
                label_jaccard_acc += jaccard(labels, torch.sigmoid(preds[:, 0, :, :]))
                watersh_jaccard_acc += jaccard(contour_labels, torch.sigmoid(preds[:, 1, :, :]))
                contour_jaccard_acc += jaccard(watershed_labels, torch.sigmoid(preds[:, 2, :, :]))
                # dice_acc += dice(labels, torch.sigmoid(preds))
            
            epoch_loss = running_loss / len(dataloaders[phase])
            aver_jaccard = jaccard_acc / len(dataloaders[phase])
            aver_label_jaccard = label_jaccard_acc / len(dataloaders[phase])
            aver_watersh_jaccard = watersh_jaccard_acc / len(dataloaders[phase])
            aver_contour_jaccard = contour_jaccard_acc / len(dataloaders[phase])

            print("| {} Loss: {:.4f} | Jaccard Average Acc: {:.4f} |".format(phase, epoch_loss, aver_jaccard))
            print("| Label Jaccard Average Acc: {:.4f} | Watershed Jaccard Average Acc: {:.4f} | Contour Jaccard Average Acc: {:.4f} |".format(aver_label_jaccard, aver_watersh_jaccard, aver_contour_jaccard))
            print("_"*15)
            if phase == "valid" and aver_jaccard > best_acc:
                best_acc = aver_jaccard
                best_model_wts = copy.deepcopy(cust_model)
            if phase == "valid":
                val_acc_history.append(aver_jaccard)
        print("^"*15)
        print(" ")
        scheduler.step()
    time_elapsed = time.time() - start_time
    print("Training Complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best Validation Accuracy: {:.4f}".format(best_acc))
    # best_model_wts = copy.deepcopy(cust_model.state_dict())
    cust_model.load_state_dict(best_model_wts.state_dict())
    return cust_model, val_acc_history

segm_model, acc = train_model(segm_model, dict_loaders, criterion, optimizerSGD, nr_epochs, scheduler=scheduler)
save_model(segm_model, name="dense_linknet_512_sgd_bce.pt")
