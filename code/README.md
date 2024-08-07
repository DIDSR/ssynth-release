## Setup

- Create and activate Python environment:
   ```
   conda create -n ssynth python=3.9
   conda activate ssynth
   pip install -r requirements.txt
   ```

- Clone repo:
   ```
   git clone https://github.com/DIDSR/ssynth-release.git
   ```

- Setup hugging face token for data download:
   -  Follow instructions on [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens) to make a token. 
   - Run and paste your token:
   ```
   huggingface-cli login
   ```

## Code Organization

Please see:
 * the ```processing/``` folder for information of how to download pre-generated images, 
set up data-splits and run the segmentation model used in the paper. 
 * the ```data_generation/``` folder for information of how to generate synthetic skin images
and associated annotations. 
