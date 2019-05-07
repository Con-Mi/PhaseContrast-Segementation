from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import pandas as pd
import numpy as np
from skimage import io
from PIL import Image

import os


class PhaseContrastData(Dataset):
    def __init__(self, input_channels, data_root, file_data_idx, file_label_idx, watershed_label_idx=None, contour_label_idx=None, transform=None, mode="train"):
        self.data_root = data_root
        self.file_data_idx = file_data_idx
        self.label_data_idx = file_label_idx
        self.watershed_label_flag = True if watershed_label_idx is not None else False
        self.contour_label_flag = True if contour_label_idx is not None else False
        
        self.watershed_label_idx = watershed_label_idx
        self.contour_label_idx = contour_label_idx

        self.transform = transform
        self.input_channels = input_channels
        self.mode = mode

        if self.mode is "train":
            self.data_dir = os.path.join(self.data_root, "images/")
            self.label_dir = os.path.join(self.data_root, "labels/")
            if self.watershed_label_flag and self.contour_label_flag:
                self.watershed_label_dir = os.path.join(self.data_root, "watershed_labels/")
                self.contour_label_dir = os.path.join(self.data_root, "contour_labels/")
            elif self.contour_label_flag:
                self.contour_label_dir = os.path.join(self.data_root, "contour_labels/")
            elif self.watershed_label_flag:
                self.watershed_label_dir = os.path.join(self.data_root, "watershed_labels/")
            else:
                pass

        elif self.mode is "validation":
            pass
        elif self.mode is "test":
            pass
    
    def __len__(self):
        return len(self.file_data_idx)

    def __getitem__(self, index):
        file_id = self.file_data_idx["ids"].iloc[index]
        label_id = self.label_data_idx["ids"].iloc[index]
        if self.mode is "train":
            self.image_path = os.path.join(self.data_dir, file_id)
            self.label_path = os.path.join(self.label_dir, label_id)
            image = io.imread(self.image_path)
            image = (image/256).astype("uint8")
            # image = np.expand_dims(image, axis=2)
            label = io.imread(self.label_path)

            if self.contour_label_flag and self.watershed_label_flag:
                self.watershed_label_path = os.path.join(self.watershed_label_dir, label_id)
                self.contour_label_path = os.path.join(self.contour_label_dir, label_id)
                contour_label = io.imread(self.contour_label_path)
                watershed_label = io.imread(self.watershed_label_path)
                if self.transform is not None:
                    image = self.transform(image)
                    label = self.transform(label)
                    contour_label = self.transform(contour_label)
                    watershed_label = self.transform(watershed_label)
                return image, label, contour_label, watershed_label
            
            elif self.watershed_label_flag:
                self.watershed_label_path = os.path.join(self.watershed_label_dir, label_id)
                watershed_label = io.imread(self.watershed_label_path)
                if self.transform is not None:
                    image = self.transform(image)
                    label = self.transform(label)
                    watershed_label = self.transform(watershed_label)
                return image, label, watershed_label

            elif self.contour_label_flag:
                self.contour_label_path = os.path.join(self.contour_label_dir, label_id)
                contour_label = io.imread(self.contour_label_path)
                if self.transform is not None:
                    image = self.transform(image)
                    label = self.transform(label)
                    contour_label = self.transform(contour_label)
                return image, label, contour_label
            
            else:
                if self.transform is not None:
                    image = self.transform(image)
                    label = self.transform(label)
                return image, label

        if self.mode is "validation":
            pass
        if self.mode is "test":
            pass

def PhaseContrastDataset(input_csv_file, label_csv_file, input_chnls=1, data_transform=None, mode="train", batch_sz=2, workers=0):
    data_root = "../data/TrainingData"
    file_idxs = pd.read_csv(input_csv_file)
    label_idxs = pd.read_csv(label_csv_file)
    # if data_transform is None:
        # data_transform = transforms.ToTensor()

    dataset = PhaseContrastData(input_chnls, data_root, file_idxs, label_idxs, watershed_label_idx=True, contour_label_idx=True, transform=data_transform, mode=mode)
    # dataloader = DataLoader(dataset, batch_size=batch_sz, num_workers=workers, shuffle=True)
    return dataset



def PhaseContrastTrainValidLoader(input_csv_file, label_csv_file, input_chnls=3, data_transform=None, mode="train", validation_split=0.1, batch_sz=2, workers=0):
    file_idxs = pd.read_csv(input_csv_file)
    label_idxs = pd.read_csv(label_csv_file)
    if data_transform:
        data_transform = transforms.ToTensor()
    # dataset = PhaseContrastData(input_chnls, file_idxs, label_idxs, transform=data_transform, mode=mode)

    shuffle_dataset = True
    random_seed = 1234

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_sz, sampler=train_sampler, num_workers=workers)
    validation_loader = DataLoader(dataset, batch_size=batch_sz, sampler=valid_sampler, num_workers=workers)

    return train_loader, validation_loader