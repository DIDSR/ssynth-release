import sys

sys.path.append('src/')
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from utils.helper_funcs import (
    load_config,
    mean_of_list_of_tensors,
)
from forward.forward_schedules import ForwardSchedule
from reverse.reverse_process import sample
from models import *
# # https://github.com/lucidrains/ema-pytorch
from ema_pytorch import EMA
from argument import get_argparser
from metrics import get_binary_metrics
import warnings

warnings.filterwarnings("ignore")
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import argparse
import warnings
from tqdm import tqdm

jet = plt.get_cmap("jet")
device = 'cuda'
batch_size = 256


def get_image_list(im_test_list):
    '''
    load testing dataset

    Parameters:
        - im_test_list: text file with each line corresponds to the path for a test image
        
    Returns:
        - test_indices_real_all: list of all paths for the test images
    '''
    test_images_file = im_test_list
    with open(test_images_file, 'r') as f:
        lines = f.readlines()
    test_indices_real_all = [line.split('\n')[0] for line in lines]  # if real_data_key_word in line]
    print('Number of test images found:', len(test_indices_real_all))

    return test_indices_real_all


def get_transforms(INPUT_SIZE):
    '''
    transformations applied to each image/mask pair as an input to the model
    Parameters:
        - INPUT_SIZE: size of the input image/mask
        
    Returns:
        - transforms for the image/mask pair
    '''
    # transform for image
    img_transform = transforms.Compose(
        [
            transforms.Resize(
                size=[INPUT_SIZE, INPUT_SIZE],
                interpolation=transforms.functional.InterpolationMode.BILINEAR,
            ),
        ]
    )
    # transform for mask
    msk_transform = transforms.Compose(
        [
            transforms.Resize(
                size=[INPUT_SIZE, INPUT_SIZE],
                interpolation=transforms.functional.InterpolationMode.NEAREST,
            ),
        ]
    )
    return img_transform, msk_transform


def get_skin_color_csv(test_datatype, test_indices_real_all):
    '''
    get pre-generated skin tone values
    
    Parameters: 
        - test_datatype: datatype of test images: 'real_HAM', 'real_ISIC', or 'synthetic'
        - test_indices_real_all: list of all paths for the test images
        
    Returns: 
        -skin_color_df: pre-generated dataframe containing a list of paths to all the images and their corresponding pre-estimated skin tone values
        - test_dataset: list of all paths for the test images
        - data_dir: location of image/mask pairs
    '''
    # contains image_ID (name), ITA (value) color (dark, lt1 etc)

    saveDir = '../../files'  # location of the pre-generated ITA score .csv files
    if test_datatype == "real_HAM":
        test_dataset = test_indices_real_all
        skin_color_df = pd.read_csv(saveDir + '/skin_color_df_' + 'HAM' + '.csv')
        # data_dir = '../../data/real_dataset/HAM10k/'
        data_dir = ''

    elif test_datatype == "real_ISIC":
        skin_color_df = pd.read_csv(saveDir + '/skin_color_df_' + 'ISIC' + '.csv')
        test_dataset = test_indices_real_all
        data_dir = ''
    else:  # test_datatype == 'synth_light':
        skin_color_df = None
        test_dataset = test_indices_real_all
        data_dir = 'data_dir_not'
    return skin_color_df, test_dataset, data_dir


def get_data(test_dataset):
    '''
    load image and mask paths
    
    Parameters:
        - test_dataset: list of all paths for the test images
        
    Returns: 
        - l_image_filenames: list of all paths for the test images
        - l_mask_filenames: list of all paths for corresponding masks
    '''
    if test_datatype == "real_HAM":
        l_image_filenames = []
        l_mask_filenames = []

        for image_ID in test_dataset:
            _nameimage_path = image_ID

            if 'HAM10000_images_part_1' in _nameimage_path:
                _mask_path = _nameimage_path.replace("/HAM10000_images_part_1/",
                                                     "/HAM10000_segmentations_lesion_tschandl/")
                _mask_path = _mask_path.replace(".jpg", "_segmentation.png")
            else:
                _mask_path = _nameimage_path.replace("/HAM10000_images_part_2/",
                                                     "/HAM10000_segmentations_lesion_tschandl/")
                _mask_path = _mask_path.replace(".jpg", "_segmentation.png")

            l_image_filenames.append(_nameimage_path)
            l_mask_filenames.append(_mask_path)

    elif test_datatype == "real_ISIC":
        l_image_filenames = []
        l_mask_filenames = []
        for _nameimage_path in test_dataset:
            l_image_filenames.append(_nameimage_path)
            _mask_path = os.path.join(
                os.path.dirname(_nameimage_path).replace('ISIC2018_Task1-2_Training_Input',
                                                         'ISIC2018_Task1_Training_GroundTruth') + '/' +
                _nameimage_path.split('/')[-1].split('.')[0] + '_segmentation' + '.png')
            l_mask_filenames.append(_mask_path)

    else:
        l_image_filenames = test_dataset
        l_mask_filenames = [img_path.replace('image.png', 'mask.png') for img_path in l_image_filenames]
    return l_image_filenames, l_mask_filenames


def get_model(my_best_model_path, config):
    '''
    load model
    
    Parameters:
        - my_best_model_path: path for pre-trained model 
        - config: configurations for the model 
        
    Returs:
        - model: model
    '''
    checkpoint = torch.load(my_best_model_path, map_location="cpu")
    print('Loaded ' + my_best_model_path + ' checkpoint!')

    # load ema
    model = DermoSegDiff(**config["model"]["params"])
    model = model.to(device)

    # ema_params = {'beta': 0.9999, 'update_after_step': 500, 'update_every': 1, 'inv_gamma': 1.0, 'power': 0.9}
    ema = EMA(model=model, **config["training"]["ema"]["params"])
    ema = ema.to(device)

    ema.load_state_dict(checkpoint["ema"])
    model = ema.ema_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.to(device)
    model = model.eval()
    return model


def read_example(test_datatype, img_name, mask_name):
    '''
    normalize input image and mask
    
    Parameters:
        - test_datatype: datatype of test images: 'real_HAM', 'real_ISIC', or 'synthetic'
        - img_name: path for the test image 
        - mask_name: path for the mask
        
    Returns:
        - img: normalized image array
        - msk: normalized mask array
    '''
    img = read_image(img_name, ImageReadMode.RGB)
    msk = read_image(mask_name, ImageReadMode.GRAY)
    msk[0] = 255 * (msk[0] > 0)  # threshold to binary new

    img = img_transform(img)
    img = (img - img.min()) / (img.max() - img.min())

    msk = msk_transform(msk)
    msk = (msk - msk.min()) / (msk.max() - msk.min())
    return img, msk


def get_predictions(model, batch_imgs, forward_schedule, timesteps):
    '''
    evaluate model on a batch of images - adopted based on DermoSegDiff model
    
    Parameters:
        - model: model
        - batch_images: batch of test images
        - forward_Schedule: forward_Schedule provided as a configuration by DermoSegDiff
        - timesteps: timesteps provided as a configuration by DermoSegDiff
        
    Returns: 
        - preds: output prediction masks
    '''
    samples_list, mid_samples_list = [], []
    all_samples_list = []
    for en in range(ensemble):
        samples = sample(
            forward_schedule,
            model,
            images=batch_imgs,
            out_channels=1,
            desc=f"ensemble {en + 1}/{ensemble}",
        )

        samples_list.append(samples[-1][:, :1, :, :].to(device))
        mid_samples_list.append(
            samples[-int(0.1 * timesteps)][:, :1, :, :].to(device)
        )
        all_samples_list.append([s[:, :1, :, :] for s in samples])
    preds = mean_of_list_of_tensors(samples_list)
    return preds


def get_ita_value(img_path, mask_path):
    '''
    calculate ita and color from image/mask pair
    
    Parameters: 
        - img_path: path for an image
        - mask_path: path for the corresponding ground truth mask
        
    Returns: 
        - ITA: estimated ITA score for the skin
        - color: estimated color of the skin based the ITA score
    '''

    def calculate_mean_cannel_value(channel_array, mask):
        c_mean_temp = np.median(channel_array[mask == 0], axis=0)
        c_std = np.std(channel_array[mask == 0], axis=0)
        c_mean = np.median(
            channel_array[(channel_array >= c_mean_temp - c_std) & (channel_array <= c_mean_temp + c_std)], axis=0)
        return c_mean

    def calculate_ITA(L, b):
        ITA = np.arctan((L - 50) / b) * (180 / np.pi)
        return ITA

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)[:, :, 1]
    mask = (mask > 0).astype(int)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L_channel = np.multiply(lab[:, :, 0], (mask == 0))
    b_channel = np.multiply(lab[:, :, 2], (mask == 0))
    L = calculate_mean_cannel_value(L_channel, mask)
    b = calculate_mean_cannel_value(b_channel, mask)
    ITA = calculate_ITA(L, b)

    if ITA <= 10:
        color = "dark"
    elif 10 < ITA <= 19:
        color = "tan1"
    elif 19 < ITA <= 28:
        color = "tan2"
    elif 28 < ITA <= 34.5:
        color = "int1"
    elif 34.5 < ITA <= 41:
        color = "int2"
    elif 41 < ITA <= 48:
        color = "lt1"
    elif 48 < ITA <= 55:
        color = "lt2"
    elif ITA >= 55:
        color = "very_lt"
    return ITA, color


def get_full_test_dataset(test_dataset):
    '''
    get images masks, pregenerated ITA scores
    
    Parameters:
        - test_dataset: list of all paths for the test images
        
    Returns:
        - l_image_filenames: list of all paths for the test images
        - l_mask_filenames: list of all paths for corresponding masks
        - l_colors: list of colors for all the images
        - l_ita_vals: list of ita values for all the images
    '''
    # get full test dataset
    l_image_filenames, l_mask_filenames = get_data(test_dataset)
    print("number of images used:" + str(len(l_image_filenames)))
    print("number of masks used:" + str(len(l_mask_filenames)))
    if test_datatype == 'real_HAM' or test_datatype == 'real_ISIC':  # skin_color_df:
        # get pre-generated ITA values for dataset
        l_image_ids = [l_image_filename.split('/')[-1].split('.')[0] for l_image_filename in l_image_filenames]
        l_colors = [skin_color_df[skin_color_df['image_ID'].values == l_image_ids[i]]['color'].item() for i in
                    range(len(l_image_ids))]
        l_ita_vals = [skin_color_df[skin_color_df['image_ID'].values == l_image_ids[i]]['ITA'].item() for i in
                      range(len(l_image_ids))]
    else:
        # calculate ITA scores and values
        print('calculating ITA values..')
        l_ita_vals = []
        l_colors = []
        for img_id in tqdm(range(len(l_image_filenames))):
            try:
                ita_value, color = get_ita_value(l_image_filenames[img_id], l_mask_filenames[img_id])
                l_ita_vals.append(ita_value)
                l_colors.append(color)
            except:
                print("ERROR img_id " + str(img_id))
                print(l_image_filenames[img_id])
                sys.exit(0)
    return l_image_filenames, l_mask_filenames, l_colors, l_ita_vals


def read_batch(test_datatype, batch_image_filenames, batch_mask_filenames):
    '''
    read batch of images
    
    Parameters: 
        - test_dataset: list of all paths for the test images
        - batch_image_filenames: list of paths for the batch of images
        - batch_mask_filenames: list of paths for the batch of corresponding ground truth masks
        
    Returns:
        - imgs: bactch of images
        - msks: batch of masks
    '''
    imgs = []
    msks = []
    for example_id in range(len(batch_image_filenames)):
        # read example
        img, msk = read_example(test_datatype, batch_image_filenames[example_id], batch_mask_filenames[example_id])

        imgs.append(img.numpy())
        msks.append(msk.numpy())
    return imgs, msks


def get_metrics(batch_msks, preds):
    '''
    get set of dice scores - adopted based on DermSegDiff model
    
    Parameters:
        - batch_msks: batch of ground truth masks
        - preds: output prediction masks
        
    Returns:
        - individual_dice_scores: list of dice scores for the batch  
    '''
    individual_dice_scores = []
    if batch_msks.shape[1] > 1:
        batch_msks = batch_msks[:, 0, :, :].unsqueeze(1)

    if batch_msks.shape[1] > 1:
        preds_ = torch.argmax(preds, 1, keepdim=False).float()
        msks_ = torch.argmax(batch_msks, 1, keepdim=False)
    else:
        preds_ = torch.where(preds > 0, 1, 0).float()
        msks_ = torch.where(batch_msks > 0, 1, 0)

    for i in range(np.shape(preds_)[0]):  # calculate individual dice scores
        pred0 = preds_[i, :]
        msks0 = msks_[i, :]
        test_metrics = get_binary_metrics()
        test_metrics.update(pred0, msks0)
        result = test_metrics.compute()
        individual_dice_score = result['metrics/BinaryF1Score'].item()
        individual_dice_scores.append(individual_dice_score)
    return individual_dice_scores


def evaluate(model, model_path, l_image_filenames, l_mask_filenames, l_colors, l_ita_vals,
             test_datatype, forward_schedule, timesteps, source_dir):
    df_model = pd.DataFrame(columns=["individual_image_name", "tone", "ita", "model_name", "dice"])
    '''
    get dataframe of the model performance
    
    Parameters:
        - model: model
        - model_path: path for the model
        - l_image_filenames: list of all paths for the test images
        - l_mask_filenames: list of all paths for corresponding masks
        - l_colors: list of colors for all the images
        - l_ita_vals: list of ita values for all the images
        - test_datatype: datatype of test images: 'real_HAM', 'real_ISIC', or 'synthetic'
        - forward_Schedule: forward_Schedule provided as a configuration by DermoSegDiff
        - timesteps: timesteps provided as a configuration by DermoSegDiff
        
    Returns:
        - df_model: dataframe of the model performance on all test images
    '''
    # run a single batch and save results
    individual_image_names = []
    for i in range(0, len(l_image_filenames), batch_size):
        print("Processing images {} to {} out of {}".format(i, min(i + batch_size,len(l_image_filenames)), len(l_image_filenames)))
        batch_image_filenames = l_image_filenames[i:i + batch_size]
        batch_mask_filenames = l_mask_filenames[i:i + batch_size]
        batch_image_filenames = [source_dir + batch_image_filename for batch_image_filename in batch_image_filenames]
        batch_mask_filenames = [source_dir + batch_mask_filename for batch_mask_filename in batch_mask_filenames]

        l_colors_batch = l_colors[i:i + batch_size]
        l_ita_vals_batch = l_ita_vals[i:i + batch_size]

        # read batch
        imgs, msks = read_batch(test_datatype, batch_image_filenames, batch_mask_filenames)

        imgs = torch.tensor(np.array(imgs))
        msks = torch.tensor(np.array(msks))
        batch_imgs = img_transform(imgs).cuda()
        batch_msks = msk_transform(msks).cuda()

        preds = get_predictions(model, batch_imgs, forward_schedule, timesteps)  # for DermoSegDiff

        individual_dice_scores = get_metrics(batch_msks, preds)  # for DermoSegDiff
        individual_image_names.append(batch_image_filenames)

        df_batch = pd.DataFrame({'individual_image_name': batch_image_filenames, 'tone': l_colors_batch,
                                 'ita': l_ita_vals_batch, 'model_name': [model_path] * len(batch_image_filenames),
                                 'dice': individual_dice_scores})

        df_model = pd.concat([df_model, df_batch], ignore_index=True)
    return df_model


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--im_test_list', type=str,
                        help='file containing list of images to test')
    parser.add_argument('--saveDir', type=str, default='../../data/outputs/segmentation_results/',
                        help='where to save results')
    parser.add_argument('--model', type=str, help='model to evaluate', required=True)
    parser.add_argument('--fast', help='fast test run', action='store_true')

    eval_args = parser.parse_args()

    if 'HAM' in eval_args.im_test_list:
        test_datatype = 'real_HAM'
    elif 'ISIC' in eval_args.im_test_list:
        test_datatype = 'real_ISIC'
    else:
        test_datatype = ''

    print("test_datatype: {}".format(test_datatype))
    print("Model: {}".format(eval_args.model))

    # ------------------- params (DermoSegDiff)--------------------
    argparser = get_argparser()
    l_args = ['-c', 'metadata/skeleton.yaml']

    args = argparser.parse_args(l_args)
    config = load_config(args.config_file)
    INPUT_SIZE = config["dataset"]["input_size"]
    source_dir = config["run"]["source_dir"]

    device = torch.device(config["run"]["device"])

    # load eval properties
    timesteps = config["diffusion"]["schedule"]["timesteps"]
    ensemble = config["testing"]["ensemble"]

    forward_schedule = ForwardSchedule(**config["diffusion"]["schedule"])
    # ------------------- end of params (DermoSegDiff) --------------------

    # data preprocessing
    img_transform, msk_transform = get_transforms(INPUT_SIZE)
    test_indices_real_all = get_image_list(eval_args.im_test_list)

    # get pre-generated ITA colors
    skin_color_df, test_dataset, data_dir = get_skin_color_csv(test_datatype, test_indices_real_all)

    # get filenames
    l_image_filenames, l_mask_filenames, l_colors, l_ita_vals = get_full_test_dataset(test_dataset)
    test_metrics = get_binary_metrics()

    # evaluate model
    df_all = pd.DataFrame(columns=["individual_image_name", "tone", "ita", "model_name", "dice"])

    # load model
    model = get_model(eval_args.model, config)
    if eval_args.fast:
        l_image_filenames = l_image_filenames[0:10]
        l_mask_filenames = l_mask_filenames[0:10]
        l_colors = l_colors[0:10]
        l_ita_vals = l_ita_vals[0:10]

    df_model = evaluate(model,
                        eval_args.model,
                        l_image_filenames,
                        l_mask_filenames,
                        l_colors,
                        l_ita_vals,
                        test_datatype,
                        forward_schedule,
                        timesteps,
                        source_dir)
    df_all = pd.concat([df_all, df_model], ignore_index=True)

    df_all.model_name = eval_args.model
    df_all.im_test_list = eval_args.im_test_list

    if not os.path.exists(eval_args.saveDir):
        os.makedirs(eval_args.saveDir)

    name = 'MODEL-' + eval_args.model.replace(source_dir, '').replace('/', '--')
    name += 'DATASET-' + '--'.join(eval_args.im_test_list.split('/')[-2:]).replace('/', '--').replace('.txt', '')

    df_all.to_csv(eval_args.saveDir + name + '.csv', index=False)
