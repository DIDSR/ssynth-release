import os
import glob
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torch.nn.functional as F
from utils.helper_funcs import (
    calc_edge,
    calc_distance_map,
    normalize
)

np_normalize = lambda x: (x - x.min()) / (x.max() - x.min())


class SyntheticDatasetFast(Dataset):
    def __init__(self,
                 mode,
                 data_dir=None,
                 source_dir=None,
                 name=None,
                 one_hot=True,
                 image_size=224,
                 aug=None,
                 aug_empty=None,
                 transform=None,
                 img_transform=None,
                 msk_transform=None,
                 add_boundary_mask=False,
                 add_boundary_dist=False,
                 logger=None,
                 **kwargs):
        self.print = logger.info if logger else print
        self.name = name
        # pre-set variables
        # self.data_dir = data_dir if data_dir else "/path/to/datasets/synthetic"
        self.data_dir = data_dir + '/' + self.name
        print("self.data_dir ", self.data_dir)
        self.source_dir = source_dir

        # input parameters
        self.one_hot = one_hot
        self.image_size = image_size
        self.aug = aug
        self.aug_empty = aug_empty
        self.transform = transform
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.mode = mode

        self.add_boundary_mask = add_boundary_mask
        self.add_boundary_dist = add_boundary_dist

        data_preparer = PrepareSynthetic(
            data_dir=self.data_dir,
            source_dir=self.source_dir,
            name=self.name, image_size=self.image_size, logger=logger
        )
        data = data_preparer.get_data()
        if mode == "tr":
            X = data["x_train"]
            Y = data["y_train"]
        elif mode == "vl":
            X = data["x_val"]
            Y = data["y_val"]
        # elif mode == "vl2real":
        #     X = data["x_val2real"]
        #     Y = data["y_val2real"]
        # elif mode == "vl2synth":
        #     X = data["x_val2synth"]
        #     Y = data["y_val2synth"]
        elif mode == "te":
            X = data["x_test"]
            Y = data["y_test"]
        else:
            raise ValueError()
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        self.imgs = X
        self.msks = Y

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_id = idx
        img = self.imgs[idx]
        msk = self.msks[idx]
        if self.one_hot:
            msk = (msk - msk.min()) / (msk.max() - msk.min())
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)

        if self.aug:
            if self.mode == "tr":
                img_ = np.uint8(torch.moveaxis(img * 255, 0, -1).detach().numpy())
                msk_ = np.uint8(torch.moveaxis(msk * 255, 0, -1).detach().numpy())
                augmented = self.aug(image=img_, mask=msk_)
                img = torch.moveaxis(torch.tensor(augmented['image'], dtype=torch.float32), -1, 0)
                msk = torch.moveaxis(torch.tensor(augmented['mask'], dtype=torch.float32), -1, 0)
            elif self.aug_empty:  # "tr", "vl", "te"
                img_ = np.uint8(torch.moveaxis(img * 255, 0, -1).detach().numpy())
                msk_ = np.uint8(torch.moveaxis(msk * 255, 0, -1).detach().numpy())
                augmented = self.aug_empty(image=img_, mask=msk_)
                img = torch.moveaxis(torch.tensor(augmented['image'], dtype=torch.float32), -1, 0)
                msk = torch.moveaxis(torch.tensor(augmented['mask'], dtype=torch.float32), -1, 0)
            img = img.nan_to_num(127)
            img = normalize(img)
            msk = msk.nan_to_num(0)
            msk = normalize(msk)

        if self.add_boundary_mask or self.add_boundary_dist:
            msk_ = np.uint8(torch.moveaxis(msk * 255, 0, -1).detach().numpy())

        if self.add_boundary_mask:
            boundary_mask = calc_edge(msk_, mode='canny')
            # boundary_mask = np_normalize(boundary_mask)
            msk = torch.concatenate([msk, torch.tensor(boundary_mask).unsqueeze(0)], dim=0)

        if self.add_boundary_dist:
            boundary_mask = boundary_mask if self.add_boundary_mask else calc_edge(msk_, mode='canny')
            distance_map = calc_distance_map(boundary_mask, mode='l2')
            distance_map = distance_map / (self.image_size * 1.4142)
            distance_map = np.clip(distance_map, a_min=0, a_max=0.2)
            distance_map = (1 - np_normalize(distance_map)) * 255
            msk = torch.concatenate([msk, torch.tensor(distance_map).unsqueeze(0)], dim=0)

        if self.img_transform:
            img = self.img_transform(img)
        if self.msk_transform:
            msk = self.msk_transform(msk)

        img = img.nan_to_num(0.5)
        msk = msk.nan_to_num(-1)

        sample = {"image": img, "mask": msk, "id": data_id}
        return sample


class PrepareSynthetic:
    def __init__(self, data_dir, source_dir, name, image_size, logger=None):
        self.print = logger.info if logger else print

        self.data_dir = data_dir
        self.source_dir = source_dir
        os.makedirs(data_dir, exist_ok=True)
        self.name = name
        self.image_size = image_size
        # preparing input info.
        self.data_prefix = "ISIC_"
        self.target_postfix = "_segmentation"
        self.target_fex = "png"
        self.input_fex = "jpg"
        self.data_dir = self.data_dir
        self.npy_dir = os.path.join(self.data_dir, "np")

    def __get_data_path(self):
        x_train_path = f"{self.npy_dir}/X_tr_{self.image_size}x{self.image_size}.npy"
        y_train_path = f"{self.npy_dir}/Y_tr_{self.image_size}x{self.image_size}.npy"

        x_val_path = f"{self.npy_dir}/X_val_{self.image_size}x{self.image_size}.npy"
        y_val_path = f"{self.npy_dir}/Y_val_{self.image_size}x{self.image_size}.npy"

        # x_val2real_path = f"{self.npy_dir}/X_val2real_{self.image_size}x{self.image_size}.npy"
        # y_val2real_path = f"{self.npy_dir}/Y_val2real_{self.image_size}x{self.image_size}.npy"
        #
        # x_val2synth_path = f"{self.npy_dir}/X_val2synth_{self.image_size}x{self.image_size}.npy"
        # y_val2synth_path = f"{self.npy_dir}/Y_val2synth_{self.image_size}x{self.image_size}.npy"

        x_test_path = f"{self.npy_dir}/X_test_{self.image_size}x{self.image_size}.npy"
        y_test_path = f"{self.npy_dir}/Y_test_{self.image_size}x{self.image_size}.npy"

        return {"x_train": x_train_path, "y_train": y_train_path,
                "x_val": x_val_path, "y_val": y_val_path,
                # "x_val2real": x_val2real_path, "y_val2real": y_val2real_path,
                # "x_val2synth": x_val2synth_path, "y_val2synth": y_val2synth_path,
                "x_test": x_test_path, "y_test": y_test_path}

    def __get_img_by_id(self, id):
        # img_dir = os.path.join(
        #     self.imgs_dir, f"{self.data_prefix}{id}.{self.input_fex}"
        # )
        img = read_image(id, ImageReadMode.RGB)
        return img

    def __get_msk_by_id(self, id):
        # msk_dir = os.path.join(
        #     self.msks_dir,
        #     f"{self.data_prefix}{id}{self.target_postfix}.{self.target_fex}",
        # )
        msk = read_image(id, ImageReadMode.GRAY)
        msk[0] = 255 * (msk[0] > 0)  # threshold to binary new

        return msk

    def _get_transforms(self):
        # transform for image
        img_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=[self.image_size, self.image_size],
                    interpolation=transforms.functional.InterpolationMode.BILINEAR,
                ),
            ]
        )
        # transform for mask
        msk_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=[self.image_size, self.image_size],
                    interpolation=transforms.functional.InterpolationMode.NEAREST,
                ),
            ]
        )
        return {"img": img_transform, "msk": msk_transform}

    def is_data_existed(self):
        for k, v in self.__get_data_path().items():
            if not os.path.isfile(v):
                return False
        return True

    def __organize_images(self, lines_img, lines_msk):
        # gathering images
        imgs = []
        msks = []
        for data_id in tqdm(range(len(lines_img))): #tqdm(range(len(lines_img))):
            img_path = lines_img[data_id]
            msk_path = lines_msk[data_id]
            if data_id < 2:
                print(img_path)
                print(msk_path)
                print()
            try:
                img = self.__get_img_by_id(img_path)
            except:
                print(img_path)
                # continue
                import sys
                sys.exit(0)
            msk = self.__get_msk_by_id(msk_path)

            img = self.transforms["img"](img)
            img = (img - img.min()) / (img.max() - img.min())

            msk = self.transforms["msk"](msk)
            msk = (msk - msk.min()) / (msk.max() - msk.min())

            imgs.append(img.numpy())
            msks.append(msk.numpy())

        X = np.array(imgs)
        Y = np.array(msks)
        return X, Y

    def __get_image_lists(self, path_img, path_mask, source_dir):
        flOpen_img = open(path_img, 'r')
        lines_img = flOpen_img.readlines()
        flOpen_img.close()
        lines_img = [self.source_dir + '/' + line.strip() for line in lines_img]


        flOpen_msk = open(path_mask, 'r')
        lines_msk = flOpen_msk.readlines()
        flOpen_msk.close()
        lines_msk = [self.source_dir + '/' + line.strip() for line in lines_msk]
        assert (len(lines_img) == len(lines_msk))  # number of images and labels needs to be the same
        return lines_img, lines_msk

    def prepare_data(self):
        data_path = self.__get_data_path()
        # Parameters
        self.transforms = self._get_transforms()

        # check dir
        Path(self.npy_dir).mkdir(exist_ok=True)  # create general save dir

        # open training images + masks
        for sp in ['train', 'val', 'test']:#, 'val2real', 'val2synth']:
            path_img = (self.source_dir + '/dataset_splits/' + self.name + '/' + sp +
                        '_images.txt')
            path_mask = (self.source_dir + '/dataset_splits/' + self.name + '/' + sp +
                         '_masks.txt')

            lines_img, lines_msk = self.__get_image_lists(path_img, path_mask, self.source_dir)
            if sp == 'val2real' or sp == 'val2synth':
                N = 32 # subset for quick testing
                lines_img = lines_img[:N]
                lines_msk = lines_msk[:N]

            X, Y = self.__organize_images(lines_img, lines_msk)

            self.print("Saving data...")
            np.save(data_path["x_" + sp].split(".npy")[0], X)
            np.save(data_path["y_" + sp].split(".npy")[0], Y)
            self.print(f"Saved at:\n  X: {data_path['x_' + sp]}\n  Y: {data_path['y_' + sp]}")
        return

    def get_data(self):

        data_path = self.__get_data_path()

        self.print("Checking for pre-saved files...")
        if not self.is_data_existed():
            self.print("There are no pre-saved files.")
            self.print("Preparing data...")
            self.prepare_data()
        else:
            self.print(f"Found pre-saved files at {self.npy_dir}")

        self.print("Loading...")
        X_train = np.load(data_path["x_train"])
        Y_train = np.load(data_path["y_train"])

        X_val = np.load(data_path["x_val"])
        Y_val = np.load(data_path["y_val"])

        X_test = np.load(data_path["x_test"])
        Y_test = np.load(data_path["y_test"])

        # X_val2real = np.load(data_path["x_val2real"])
        # Y_val2real = np.load(data_path["y_val2real"])
        #
        # X_val2synth = np.load(data_path["x_val2synth"])
        # Y_val2synth = np.load(data_path["y_val2synth"])

        self.print("Loaded X and Y npy format")

        return {"x_train": X_train, "y_train": Y_train,
                "x_val": X_val, "y_val": Y_val,
                # "x_val2real": X_val2real, "y_val2real": Y_val2real,
                # "x_val2synth": X_val2synth, "y_val2synth": Y_val2synth,
                "x_test": X_test, "y_test": Y_test}
