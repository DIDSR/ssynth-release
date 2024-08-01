import numpy as np
import pandas as pd
import glob
import os
import random
import argparse

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--real_data_ratio', type=float, help='ratio of the real data to be used')
parser.add_argument('--synth_data_ratio', type=float, help='ratio of adding or replacing with synthetic data')
parser.add_argument('--replace_with_synth', action='store_true', help='replace with synth data or not')
parser.add_argument('--add_synth', action='store_true', help='add synth data or not')
parser.add_argument('--saveDir', type=str, default='../../data/outputs/dataset_splits/', help='where to store output splits')
args = parser.parse_args()

real_data_ratio = args.real_data_ratio
real_data_ratio = args.real_data_ratio
synth_data_ratio = args.synth_data_ratio
replace_with_synth = args.replace_with_synth
add_synth = args.add_synth
path = args.saveDir

real_data_path = '../../data/real_dataset/HAM10k/'
regular_lesions_path = '../../data/synthetic_dataset/files/images_10k.txt' #list of synthetic images
dataset_type = "real_HAM" 
shuffle_synth = True #whether shuffle the synthetic dataset or not

HAM_metadata = pd.read_csv(real_data_path + 'HAM10000_metadata', delimiter=',')
HAM_metadata.tail()

def get_synth_data(path, shuffle_synth):
    with open(path, 'r') as file:
        synth_images_l = file.readlines()
        synth_images_l = [file.strip('\n') for file in synth_images_l]
        if shuffle_synth == True:
            random.seed(4)
            random.shuffle(synth_images_l)
    print("Number of synthetic images:", len(synth_images_l))
    return synth_images_l

def get_real_data_HAM(path):
    HAM_metadata = pd.read_csv(path + 'HAM10000_metadata', delimiter=',')
    real_images_l = list(HAM_metadata['image_id'])
    return real_images_l
    
def split_data_HAM(dataset):
    Train_img = dataset[0:7200]
    Validation_img = dataset[7200:7200+1800]
    Test_img = dataset[7200+1800:10015]
    print("Number of HAM training images:", len(Train_img))
    print("Number of HAM validation images:", len(Validation_img))
    print("Number of HAM test images:", len(Test_img))
    return Train_img, Validation_img, Test_img

def real_data_subset(dataset_real_train, dataset_real_val, dataset_real_test, real_data_ratio):
    real_count_train = int(np.floor(real_data_ratio*len(dataset_real_train)))
    combined_dataset_train = dataset_real_train[:real_count_train]
    
    real_count_val = int(np.floor(real_data_ratio*len(dataset_real_val)))
    combined_dataset_val = dataset_real_val[:real_count_val]

    real_count_test = len(dataset_real_test) 
    combined_dataset_test = dataset_real_test[:real_count_test]
    
    print("Number of Train images: {} real --> {} total".format(real_count_train, len(combined_dataset_train)))
    print("Number of Validation images: {} real  --> {} total".format(real_count_val, len(combined_dataset_val)))
    print("Number of Test images: {} real --> {} total".format(real_count_test, len(combined_dataset_test)))
    return combined_dataset_train, combined_dataset_val, combined_dataset_test


def replace_with_synth_data(dataset_real_train, dataset_real_val, dataset_real_test, dataset_synth, synth_data_ratio):
    synth_count_train = min(int(np.floor(synth_data_ratio*len(dataset_real_train))),len(dataset_synth))
    if int(np.floor(synth_data_ratio*len(dataset_real_train))) > len(dataset_synth):
        print("*** Warning: there are not enough synthetic images to replace the Training set ***")
    real_count_train = max(len(dataset_real_train)-synth_count_train,0) 
    combined_dataset_train = dataset_real_train[:real_count_train] + dataset_synth[:synth_count_train] 
    
    synth_count_val = min(int(np.floor(synth_data_ratio*len(dataset_real_val))),len(dataset_synth)-synth_count_train)
    if (int(np.floor(synth_data_ratio*len(dataset_real_train))) + int(np.floor(synth_data_ratio*len(dataset_real_val)))) > len(dataset_synth):
        print("*** Warning: there are not enough synthetic images to replace the Validation set ***")
    real_count_val = max(len(dataset_real_val)-synth_count_val,0) 
    combined_dataset_val = dataset_real_val[:real_count_val] + dataset_synth[synth_count_train:synth_count_train+synth_count_val] 

    synth_count_test = 0
    real_count_test = len(dataset_real_test) 
    combined_dataset_test = dataset_real_test[:real_count_test]
    
    print("Number of Train images: {} real and {} synthetic --> {} total".format(real_count_train, synth_count_train, len(combined_dataset_train)))
    print("Number of Validation images: {} real and {} synthetic --> {} total".format(real_count_val, synth_count_val, len(combined_dataset_val)))
    print("Number of Test images: {} real and {} synthetic --> {} total".format(real_count_test, synth_count_test, len(combined_dataset_test)))

    return combined_dataset_train, combined_dataset_val, combined_dataset_test

def add_synth_data(dataset_real_train, dataset_real_val, dataset_real_test, dataset_synth, synth_data_ratio):
    synth_count_train = min(int(np.floor(synth_data_ratio*len(dataset_real_train))),len(dataset_synth))
    if int(np.floor(synth_data_ratio*len(dataset_real_train))) > len(dataset_synth):
        print("*** Warning: there are not enough synthetic images to replace the Training set ***")
    real_count_train = len(dataset_real_train)
    combined_dataset_train = dataset_real_train + dataset_synth[:synth_count_train] 
    
    synth_count_val = min(int(np.floor(synth_data_ratio*len(dataset_real_val))),len(dataset_synth)-synth_count_train)
    if (int(np.floor(synth_data_ratio*len(dataset_real_train))) + int(np.floor(synth_data_ratio*len(dataset_real_val)))) > len(dataset_synth):
        print("*** Warning: there are not enough synthetic images to replace the Validation set ***")
    real_count_val = len(dataset_real_val)
    combined_dataset_val = dataset_real_val + dataset_synth[synth_count_train:synth_count_train+synth_count_val] 

    synth_count_test = 0
    real_count_test = len(dataset_real_test) 
    combined_dataset_test = dataset_real_test[:real_count_test]
    
    print("Number of Train images: {} real and {} synthetic --> {} total".format(real_count_train, synth_count_train, len(combined_dataset_train)))
    print("Number of Validation images: {} real and {} synthetic --> {} total".format(real_count_val, synth_count_val, len(combined_dataset_val)))
    print("Number of Test images: {} real and {} synthetic --> {} total".format(real_count_test, synth_count_test, len(combined_dataset_test)))

    return combined_dataset_train, combined_dataset_val, combined_dataset_test

def image_and_mask_path_HAM(dataset, data_dir_real):
    dataset_image = []
    dataset_mask = []
    for image_ID in dataset:
        if os.path.exists(image_ID): #synthetic
            dataset_image.append(image_ID)
            mask_path = image_ID.replace('image.png','mask.png')
            dataset_mask.append(mask_path)
        else: #real
            nameimage_path = os.path.join(data_dir_real + 'HAM10000_images_part_1' +  '/' + image_ID +'.jpg')
            if os.path.exists(nameimage_path):
                dataset_image.append(nameimage_path)
            else: 
                nameimage_path = os.path.join(data_dir_real + 'HAM10000_images_part_2' +  '/' + image_ID +'.jpg')
                dataset_image.append(nameimage_path)
            mask_path = os.path.join(data_dir_real + 'HAM10000_segmentations_lesion_tschandl' +  '/' + image_ID + '_segmentation' +'.png')
            dataset_mask.append(mask_path)
            

    return dataset_image, dataset_mask

#Get the image IDs for real and synthetic data
synth_data_list = get_synth_data(regular_lesions_path, shuffle_synth)
    
real_data_list = get_real_data_HAM(real_data_path)
dataset_real_train, dataset_real_val, dataset_real_test = split_data_HAM(real_data_list)

if real_data_ratio != 1:
    print("Using a subset of the real data ...")
    combined_dataset_train, combined_dataset_val, combined_dataset_test = real_data_subset(dataset_real_train, dataset_real_val, dataset_real_test, real_data_ratio)
elif replace_with_synth == True:
    print("Replacing real with synthic data ...")
    combined_dataset_train, combined_dataset_val, combined_dataset_test = replace_with_synth_data(dataset_real_train, dataset_real_val, dataset_real_test, synth_data_list, synth_data_ratio)
elif add_synth == True:
    print("Adding synthetic data to real data! ")
    combined_dataset_train, combined_dataset_val, combined_dataset_test = add_synth_data(dataset_real_train, dataset_real_val, dataset_real_test, synth_data_list, synth_data_ratio)
else:
    print("Using real data only! ")
    combined_dataset_train, combined_dataset_val, combined_dataset_test = dataset_real_train, dataset_real_val, dataset_real_test
    
#Get the image and mask paths
train_images, train_masks = image_and_mask_path_HAM(combined_dataset_train, real_data_path)
val_images, val_masks = image_and_mask_path_HAM(combined_dataset_val, real_data_path)
test_images, test_masks = image_and_mask_path_HAM(combined_dataset_test, real_data_path)

print("Number of all images for training: {}".format(len(train_images)))
print("Number of all  masks for training: {}".format(len(train_masks)))
print("Number of all  images for validation: {}".format(len(val_images)))
print("Number of all  masks for validation: {}".format(len(val_masks)))
print("Number of all  images for test: {}".format(len(test_images)))
print("Number of all  masks for test: {}".format(len(test_masks)))

#Check for duplicates:
train_images_set = set(train_images)
val_images_set = set(val_images)
test_images_set = set(test_images)
train_masks_set = set(train_masks)
val_masks_set = set(val_masks)
test_masks_set = set(test_masks)

subsets = [train_images_set, val_images_set, test_images_set, train_masks_set, val_masks_set ,test_masks_set]
i=0

for subset1 in subsets:
    i += 1
    j=0
    for subset2 in subsets:
        j += 1
        if list(set(subset1) & set(subset2)) == []:
#             print("No data leakage")
            pass
        else:
            if i == j:
                pass
            else:
                print("Error: Data leakage!!!")
                print(i)
                print(j)
                
# make data split text files
datasets_dict = {
    "train_images": train_images,
    "train_masks": train_masks,
    "val_images": val_images,
    "val_masks": val_masks,
    "test_images": test_images,
    "test_masks": test_masks
}

foldername = 'all_tones_' + dataset_type + '_' + str(real_data_ratio)
if replace_with_synth == True:
    foldername = foldername + '_replace_with_synth_' + str(synth_data_ratio)
if add_synth == True:
    foldername = foldername + '_add_synth_'  + str(synth_data_ratio)
print("Folder name: ", foldername)

os.makedirs(os.path.join(path, foldername), exist_ok=True)

for dataset_name, dataset in datasets_dict.items():
    name = dataset_name
    print(name)
    with open(os.path.join(path, foldername)+ '/' + name + '.txt', 'w') as file:
        file.writelines(f"{name}\n" for name in dataset)