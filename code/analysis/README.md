# Addditional Analysis

* Set root folder path:

   ```
   cd ssynth-dev
   CWD=$(pwd)
   ```

* Download dataset (we implemented minor parameter updates since the IJCARS'26 publication):

    ```
    cd $CWD/code/processing
    python download_ssynth.py --name '10k_None.zip' --saveDir '../../' --unzip
    ```

* Create a list of synthetic images:
    ```
    find $CWD/data/synthetic_dataset/10k_None -name image.png > $CWD/data/synthetic_dataset/files/images_10k_artifact_None.txt
    ```

2. Download HAM10K and ISIC18 datasets (follow instructions in https://github.com/DIDSR/ssynth-dev/tree/main/code/processing#setup-processing)

3. Run the ```images_masks_and_feature_analysis.ipynb``` notebook. 
    
