from os import listdir
import pandas as pd


def get_ids_in_csv(inputs, labels, destination_name_inputs, destination_name_labels):
    list_ids_train_input = [f for f in listdir(inputs)]
    list_ids_train_labels = [f for f in listdir(labels)]

    list_ids_train_input = sorted(list_ids_train_input)
    list_ids_train_labels = sorted(list_ids_train_labels)

    dict_train_input = {"ids": list_ids_train_input}
    dict_train_labels = {"ids": list_ids_train_labels}

    del list_ids_train_input, list_ids_train_labels
    train_input_df = pd.DataFrame(dict_train_input, index=None)
    train_labels_df = pd.DataFrame(dict_train_labels, index=None)
    train_input_df.to_csv(destination_name_inputs)
    train_labels_df.to_csv(destination_name_labels)

def get_ids_in_list(inputs):
    label_list = [f for f in listdir(inputs)]
    label_list = sorted(label_list)
    return label_list


get_ids_in_csv( "../data/TrainingData/images/", "../data/TrainingData/labels/", "../data/TrainingData/train_input_imgs.csv", "../data/TrainingData/train_label_imgs.csv" )
get_ids_in_csv( "../data/AugmentedData/images/", "../data/AugmentedData/labels/", "../data/AugmentedData/train_input_imgs.csv", "../data/AugmentedData/train_label_imgs.csv" )