from dataloader_skimage import PhaseContrastDataset


input_csv_file = "../data/TrainingData/train_input_imgs.csv"
label_csv_file = "../data/TrainingData/train_label_imgs.csv"

data = PhaseContrastDataset(input_csv_file, label_csv_file, input_chnls=1, data_transform=None, mode="train", batch_sz=1, workers=0)


#loader = PhaseContrastDataLoader(input_csv_file, label_csv_file, input_chnls=1, data_transform=None, mode="train", batch_sz=1, workers=0)

for i, label in enumerate(data):
    # print(label[0])
    print(label[3].shape)
    # print(i)