from datasets.synthetic import SyntheticDatasetFast
from torch.utils.data import DataLoader, Subset
from modules.transforms import DiffusionTransform, DataAugmentationTransform
import albumentations as A
import glob



def get_synthetic(config, logger=None, verbose=False):

    if logger: print = logger.info
    INPUT_SIZE = config["dataset"]["input_size"]
    DT = DiffusionTransform((INPUT_SIZE, INPUT_SIZE))
    AUGT = DataAugmentationTransform((INPUT_SIZE, INPUT_SIZE))

    img_path_list = None
    pixel_level_transform = AUGT.get_pixel_level_transform(config["augmentation"], img_path_list=img_path_list)
    spacial_level_transform = AUGT.get_spacial_level_transform(config["augmentation"])
    tr_aug_transform = A.Compose([
        A.Compose(pixel_level_transform, p=config["augmentation"]["levels"]["pixel"]["p"]), 
        A.Compose(spacial_level_transform, p=config["augmentation"]["levels"]["spacial"]["p"])
    ], p=config["augmentation"]["p"])

    # ----------------- dataset --------------------
    # if config["dataset"]["class_name"] == "ISIC2018Dataset":
    #
    #     dataset = ISIC2018Dataset(
    #         data_dir=config["dataset"]["data_dir"],
    #         one_hot=False,
    #         # aug=AUGT.get_aug_policy_3(),
    #         # transform=AUGT.get_spatial_transform(),
    #         img_transform=DT.get_forward_transform_img(),
    #         msk_transform=DT.get_forward_transform_msk(),
    #         logger=logger
    #     )
    #
    #     #         tr_dataset = Subset(dataset, range(0    , 38       ))
    #     #         vl_dataset = Subset(dataset, range(38   , 38+25    ))
    #     #         te_dataset = Subset(dataset, range(38+25, len(dataset)))
    #     tr_dataset = Subset(dataset, range(0, 1815))
    #     vl_dataset = Subset(dataset, range(1815, 1815 + 259))
    #     te_dataset = Subset(dataset, range(1815 + 259, 1815 + 259 + 520))
    #     # We consider 1815 samples for training, 259 samples for validation and 520 samples for testing
    #     # !cat ~/deeplearning/skin/Prepare_ISIC2018.py
    #
    if config["dataset"]["class_name"] == "SyntheticDatasetFast":
        # preparing training dataset
        tr_dataset = SyntheticDatasetFast(
            mode="tr",
            data_dir=config["run"]["data_dir"],
            source_dir=config["run"]["source_dir"],
            name=config["run"]["name"],
            one_hot=False,
            image_size=config["dataset"]["input_size"],
            aug=tr_aug_transform,
            # transform=AUGT.get_spatial_transform(),
            img_transform=DT.get_forward_transform_img(),
            msk_transform=DT.get_forward_transform_msk(),
            add_boundary_mask=config["dataset"]["add_boundary_mask"],
            add_boundary_dist=config["dataset"]["add_boundary_dist"],
            logger=logger,
            data_scale=config["dataset"]["data_scale"]
        )
        vl_dataset = SyntheticDatasetFast(
            mode="vl",
            data_dir=config["run"]["data_dir"],
            source_dir=config["run"]["source_dir"],
            name=config["run"]["name"],
            one_hot=False,
            image_size=config["dataset"]["input_size"],
            # aug_empty=AUGT.get_val_test(),
            # transform=AUGT.get_spatial_transform(),
            img_transform=DT.get_forward_transform_img(),
            msk_transform=DT.get_forward_transform_msk(),
            add_boundary_mask=config["dataset"]["add_boundary_mask"],
            add_boundary_dist=config["dataset"]["add_boundary_dist"],
            logger=logger,
            data_scale=config["dataset"]["data_scale"]
        )
        # vl2real_dataset = SyntheticDatasetFast(
        #     mode="vl2real",
        #     data_dir=config["run"]["data_dir"],
        #     source_dir=config["run"]["source_dir"],
        #     name=config["run"]["name"],
        #     one_hot=False,
        #     image_size=config["dataset"]["input_size"],
        #     # aug_empty=AUGT.get_val_test(),
        #     # transform=AUGT.get_spatial_transform(),
        #     img_transform=DT.get_forward_transform_img(),
        #     msk_transform=DT.get_forward_transform_msk(),
        #     add_boundary_mask=config["dataset"]["add_boundary_mask"],
        #     add_boundary_dist=config["dataset"]["add_boundary_dist"],
        #     logger=logger,
        #     data_scale=config["dataset"]["data_scale"]
        # )
        # vl2synth_dataset = SyntheticDatasetFast(
        #     mode="vl2synth",
        #     data_dir=config["run"]["data_dir"],
        #     source_dir=config["run"]["source_dir"],
        #     name=config["run"]["name"],
        #     one_hot=False,
        #     image_size=config["dataset"]["input_size"],
        #     # aug_empty=AUGT.get_val_test(),
        #     # transform=AUGT.get_spatial_transform(),
        #     img_transform=DT.get_forward_transform_img(),
        #     msk_transform=DT.get_forward_transform_msk(),
        #     add_boundary_mask=config["dataset"]["add_boundary_mask"],
        #     add_boundary_dist=config["dataset"]["add_boundary_dist"],
        #     logger=logger,
        #     data_scale=config["dataset"]["data_scale"]
        # )
        te_dataset = SyntheticDatasetFast(
            mode="te",
            data_dir=config["run"]["data_dir"],
            source_dir=config["run"]["source_dir"],
            name=config["run"]["name"],
            one_hot=False,
            image_size=config["dataset"]["input_size"],
            # aug_empty=AUGT.get_val_test(),
            # transform=AUGT.get_spatial_transform(),
            img_transform=DT.get_forward_transform_img(),
            msk_transform=DT.get_forward_transform_msk(),
            add_boundary_mask=config["dataset"]["add_boundary_mask"],
            add_boundary_dist=config["dataset"]["add_boundary_dist"],
            logger=logger,
            data_scale=config["dataset"]["data_scale"]
        )

    else:
        message = "In the config file, `dataset>class_name` should be in: ['SyntheticDatasetFast']"
        if logger: 
            logger.exception(message)
        else:
            raise ValueError(message)

    if verbose:
        print("Synthetic:")
        print(f"├──> Length of trainig_dataset:\t   {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset: {len(vl_dataset)}")
        print(f"└──> Length of test_dataset:\t   {len(te_dataset)}")

    # prepare train dataloader
    tr_dataloader = DataLoader(tr_dataset, **config["data_loader"]["train"])

    # prepare validation dataloader
    vl_dataloader = DataLoader(vl_dataset, **config["data_loader"]["validation"])
    # vl2real_dataloader = DataLoader(vl2real_dataset, **config["data_loader"]["validation"])
    # vl2synth_dataloader = DataLoader(vl2synth_dataset, **config["data_loader"]["validation"])

    # prepare test dataloader
    te_dataloader = DataLoader(te_dataset, **config["data_loader"]["test"])

    return {
        "tr": {"dataset": tr_dataset, "loader": tr_dataloader},
        "vl": {"dataset": vl_dataset, "loader": vl_dataloader},
        # "vl2real": {"dataset": vl2real_dataset, "loader": vl2real_dataloader},
        # "vl2synth": {"dataset": vl2synth_dataset, "loader": vl2synth_dataloader},
        "te": {"dataset": te_dataset, "loader": te_dataloader},
    }
