## Setup (Processing)
Please follow the steps below to download synthetic and real datasets, train and evaluate a segmentation
model. 

- Set up home folder path:
   ```
   cd S-SYNTH
   CWD=$(pwd)
   ```

- Download synthetic (S-SYNTH) dataset and extract within `../data/synthetic_dataset/`.
  ```
  cd $CWD/code/processing
  python download_ssynth.py --name 'output_10k.zip' --saveDir '../../' --unzip
  ```

- Download real datasets HAM10K and ISIC18:

   - Download the HAM 10000 dataset from this [link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and extract the following folders inside `S-SYNTH/data/real_dataset/HAM10k/`:
     * HAM10000_images_part_1.zip
     * HAM10000_images_part_2.zip
     * HAM10000_metadata.tab
     * HAM10000_segmentations_lesion_tschandl.zip
 
   - Download the ISIC 2018 dataset from this [link](https://challenge.isic-archive.com/data/) and extract the following folders inside `S-SYNTH/data/real_dataset/ISIC2018/`:
      * ISIC2018_Task1-2_Training_Input
      * ISIC2018_Task1_Training_GroundTruth
      ```
      mkdir -p $CWD/data/real_dataset/ISIC2018/
      cd $CWD/data/real_dataset/ISIC2018/
      wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip
      unzip ISIC2018_Task1-2_Training_Input.zip
      wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip
      unzip ISIC2018_Task1_Training_GroundTruth.zip
      ```

## Training/Testing Segmentation Model
This code below is heavily based on the [DermsegDiff](https://github.com/xmindflow/DermoSegDiff) implementation of [1].

### Evaluating a Pre-trained Model:

- Download pre-trained model:
   ```
   cd $CWD/code/processing
   NAME="all_tones_real_ISIC_1.0_add_synth_0.2"
   python download_pretrained_model.py --name $NAME --saveDir '../../'
   ```

- The `test.py` script will take a pre-trained model and a text file (located at `../../data/dataset_splits/`) containing list of images to test and will create a Pandas DataFrame of the test results.  
      
   ```
   cd $CWD/code/processing
   MODEL_DIR='../../data/pretrained_models/all_tones_real_ISIC_1.0_add_synth_0.2/00/dsd_i01/n-dsd_i01_s-128_b-32_t-250_sc-linear_best.pth'
   TEST_LIST='../../data/dataset_splits/all_tones_real_ISIC_1.0_add_synth_0.2/test_images.txt'
   python -u test.py --model $MODEL_DIR  --im_test_list $TEST_LIST --saveDir '../../data/outputs/segmentation_results/' --fast
   ```

  The script accepts the following arguments:
  
     - --model (str, required): path to the pre-trained model.
     - --saveDir (str): where to save results.
     - --im_test_list (str, required): text file containing list of images to test (generated at step 3).
     - --fast (boolean): on/off flag to run the script in a "fast test run" mode on only 10 images.

    The results will be stored under `../../data/outputs/segmentation_results/` in the form of .csv files.

### Visualization:

In order to visualize the results of the inference step, use visualization.ipynb script which will plot the results based on the values stored in the DataFrames during the test step.

```
jupyter notebook visualization.ipynb
```

### Training Segmentation Model (DermoSegDiff):
- Download ```all_tones_real_ISIC_1.0_add_synth_0.2``` from hugging face and place within Github repo:
    ```
    cd $CWD/code/processing
    NAME="all_tones_real_ISIC_1.0_add_synth_0.2"
    python download_split.py --name $NAME --saveDir '../../'
    ```

- Run training: 
    ```
    cd $CWD/code/processing
    NAME="all_tones_real_ISIC_1.0_add_synth_0.2"
    RUN_ID="0"
    python src/training.py -c configs/ssynth.yaml --name $NAME --run_id $RUN_ID
    ```

- Outputs will be saved in ```../../data/outputs/training_outputs/``` (can also be modified within configs/ssynth.yaml).


### (Optional) Split the datasets to train, validation and test sets:
   - Create a list of synthetic images:
     
       ```
       cd $CWD/code/processing
       mkdir -p ../../data/synthetic_dataset/files
       find $CWD/data/synthetic_dataset/output_10k -name cropped_image.png > ../../data/synthetic_dataset/files/images_10k.txt
       ```

   - Run `data_split_ISIC.py` to generate own synthetic/ISIC data splits:
     
      ```
      cd $CWD/code/processing
      python -u data_split_ISIC.py --real_data_ratio 1.0 --synth_data_ratio 0.2 --saveDir '../../data/outputs/dataset_splits/' --add_synth 
      ```

   - Run `data_split_HAM.py` to generate own synthetic/HAM10k data splits:
     
      ```
      cd $CWD/code/processing
      python -u data_split_HAM.py --real_data_ratio 1.0 --synth_data_ratio 0.2 --saveDir '../../data/outputs/dataset_splits/' --add_synth
      ```
      
      These scripts accept the following arguments:
        - --real_data_ratio : subset of the total patient dataset to be used for training and validation sets (e.g. 0.1). Use 1.0 if you'd like to replace with or add synthetic image to the patient dataset.
        - --synth_data_ratio : ratio of the synthetic dataset to replace or added to the patient dataset (e.g. 0.5).
        - --replace_with_synth : Enable patient dataset to be replaced with synthetic dataset.
        - --add_synth : enable synthetic data to be added to the patient dataset. 
        - --saveDir : where to store output splits.

### References: 
[1] Afshin Bozorgpour, Yousef Sadegheih, Amirhossein Kazerouni, Reza Azad, Dorit Merhof. [DermoSegDiff: A Boundary-Aware Segmentation Diffusion Model for Skin Lesion Delineation](https://github.com/xmindflow/DermoSegDiff). MICCAI 2023 PRIME Workshop.
  
