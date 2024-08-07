## Setup (Data Generation)

S-SYNTH framework involves the construction of a 3D digital object model comprising of skin tissue (epidermis, dermis,
hypodermis), blood network, hair, and a lesion. This process is implemented in [Houdini](https://www.sidefx.com/) via a
Python API. Once created, each model is processed through [Mitsuba 3](https://www.mitsuba-renderer.org/).

The S-SYNTH images can be generated using the pre-generated skin layer models and materials provided on Hugging Face (
Section 2) or new models created according to the instructions provided in Section 3. A single example of a synthetic
image and paired segmentation mask can be rendered in Section 4. A large dataset based on the S-SYNTH framework
consisting of both S-SYNTH images and their corresponding masks can be generated as described in Section 5.

<p align="center">
<img src="../../images/Pipeline.png" width="500">
</p>

## 1. Set up root folder path

   ```
   cd ssynth-release
   CWD=$(pwd)
   ```

## 2. Render an Example with Pre-generated Skin Layer Models and Optical Materials

Download pre-generated models and materials are stored in Hugging
face ([ssynth_data/data/supporting_data/materials.zip](https://huggingface.co/datasets/didsr/ssynth_data/blob/main/data/supporting_data/materials.zip)).

- Setup huggingface token for data download:
    - Follow instructions
      on [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens) to make a
      token.
    - Run and paste your token:
   ```
   huggingface-cli login
   ```

- Download and extract data:
  ```
  cd $CWD/code/data_generation
  python download_data.py --name 'materials.zip' --saveDir '../../' --unzip
  python download_data.py --name 'hdri.zip' --saveDir '../../' --unzip 
  python download_data.py --name 'params_lists.zip' --saveDir '../../' -- unzip
  python download_data.py --name 'sample_data.csv' --saveDir '../../'
  ```

- The resulting ```supporting_data/``` folder consists of 3D skin layer models, associated materials, and lightning
  conditions:
    - Epidermis models: `materials/outputModels/epidermis*`
    - Vascular models: `materials/outputModels/vascular*`
    - Dermis models: `materials/outputModels/dermis*`
    - Hair models: `materials/outputModels/hair*`
    - Lesion models: `materials/outputModels/lesion*`
    - Optical materials for each skin layer: `materials/opticalMaterials/`
    - Lighting conditions (represented via collection of High Dynamic Range Imaging (HDRI) images): ```hdri/```
    - Sample set of parameters to generate two images (csv): ```sample_data.csv```
    - (_Optional_): Lists of parameters used to generate various image sets used in the paper: ```params_lists/``` See
      Section 3 ([Generate a Dataset](#3-generate-a-dataset))

These files should be stored in `S-SYNTH/data/supporting_data/` to be used for rendering in the next steps. If you
change this directory, please update ```config.py```.

**Note**: If you want to create your own pre-generated models and materials, go to go to
the [Skin Model Generation](#4-skin-model-generation) section.

- We provide a sample rendering notebook in `render_example.ipynb` that relies on the above pre-generated data (skin
  layers, optical materials and lightning conditions). This script will render two images, specified by parameters
  in ```sample_data.csv```.

## 3. Generate a Dataset

- Each dataset is specified by a parameter list (in ```.csv``` format), where each row describes the set of parameters
  needed to render each image in the datasets used to generate results from the manuscript. We provide pre-generated
  parameter lists or "variations" on Hugging
  Face ([ssynth_data/data/supporting_data/params_lists/](https://huggingface.co/datasets/didsr/ssynth_data/tree/main/data/supporting_data/params_lists.zip))
  These files must be stored inside  `supporting_data/param_lists/` to be used for rendering a large dataset in the next
  step. The following parameter lists are available for download:
    - 10k_dataset_release.csv: 10,000 combinations of random parameters
    - blood_variation_light0_release.csv: 7260 combinations of random parameters for different blood fractions (1815
      examples for each fixed blood fraction value [0.002, 0.005, 0.02, 0.05])
    - mel_variation_light0_release.csv: 9075 combinations of random parameters for different melanosome fractions (1815
      examples for each fixed melanosome fraction [0.01, 0.11, 0.21, 0.31, 0.41])
    - lesion_regularity_light0_release.csv: 1815 combinations of random parameters used for comparing lesion shape (
      regular vs irregular) and hair (with and without hair)


- In general, the following parameters must be provided to generate each skin image and segmentation mask:
    - id_model: skin models ID (used for epidermis, vascular, dermis, hypodermis)
    - id_hairModel: hair model ID
    - id_lesion: lesion model ID
    - id_timePoint: timepoint for the growing lesion, which determines the size of the lesion
    - id_lesionMat: lesion material ID
    - id_fracBlood: blood fraction value
    - id_mel: melanosome fraction value
    - id_light: light condition ID
    - id_hairAlbedo: hair albedo index
    - offset: lesion offset (used for rendering)
    - origin_y: camera position (used for rendering)
    - lesion_scale: lesion scaling (used for rendering)
    - mi_variant: mitsuba rendering technique (we used spectral for all experiments ('cuda_spectral' or '
      scalar_spectral'), both generate same results, but first is faster)

These values will be mapped to a specific skin model and material in the rendering step (see ```./util.py```).

**Note**. Alternatively, a new parameter list for a dataset can be created using the
script ```create_parameter_list.py```, which generates a random parameter list based on the pre-generated skin layer
models available for download on Hugging Face as described in Step 2. The sample starter code creates parameter list for
a dataset with blood fraction variation (with 1815 examples for each blood fraction value) and is intended as an
example. The following parameters can be randomly selected (see `./util.py`, under `get_l_<PARAMETER>`).

    - model: 100 skin models IDs (epidermis, vascular, dermis, hypodermis) ranging from 0 to 99 
    - hairModel: 100 hair model IDs ranging from 0 to 99
    - lesion: 20 lesion model IDs ranging from 1 to 20
    - times: timepoint for the growing lesion ranging from 15 to 55 (the higher the time, the larger the lesion)
    - lesionMat: 18 lesion material IDs ranging from 1 to 18 (the lesion material names are provided in `lesionMat`)
    - fractionBlood: blood fraction value equal to 0.002, 0.005, 0.02, or 0.05
    - melanosomes: melanosome fraction value ranging from 0.01 to 0.5 in steps of 0.01
    - light: 19 light IDs ranging from 0 to 18 (the light names are provided in `exr_files` and are available as described in Section 2)
    - hairAlbedoIndex: hair albedo index equal to 0, 1, or 2 (the values of each index are provided in `l_hair_albedo`)

- Synthetic skin images and their corresponding masks with the parameters specified within the .csv file can be rendered
  using the script `render.py`

   ```
   python -u render.py --saveDir ../../data/outputs/render_output/ --variation '10k' --row_id 0 --regLesions --res 16
   ```

  The script accepts the following arguments:
    - --saveDir (str, required): directory to save outputs
    - --variation (str, required): type of variation to generate. Must match the .csv file name stored
      in `./outputs/param_lists/`
    - --row_id (str): row id of the example to be rendered
    - --regLesions: enables usage of regular lesion (ver0 or ver1 in ```supporting_data/materials/lesions_release```,
      see )
    - --res(int): number of samples per pixel (spp) (higher values results in better quality images, but longer
      rendering times)
    - --noHair: enables hair removal in all images

The three types of synthetic datasets can be reproduced via the following commands:

```
# output_10k
for row_id in {0..10000}; do 
python -u render.py --saveDir ../../data/outputs/render_output/output_10k --variation '10k' --regLesions --row_id $row_id --res 128
done

# blood variation
for row_id in {0..7259}; do 
python -u render.py --saveDir ../../data/outputs/render_output/blood_variation_selected --variation 'blood' --row_id  --res 128
done

# melanosome (skin color) variation
for row_id in {0..9074}; do 
python -u render.py --saveDir ../../data/outputs/render_output/mel_variation_selected --variation 'mel' --row_id  --res 128
done

# regularity (lesion shape) variation
for row_id in {0..1814}; do 
python -u render.py --saveDir ../../data/outputs/render_output/regularity_variation_selected/reg/ --variation 'reg' --regLesions --row_id  --res 128
python -u render.py --saveDir ../../data/outputs/render_output/regularity_variation_selected/irrreg/ --variation 'reg' --row_id  --res 128  
done

# hair (with and without) variation
for row_id in {0..1814}; do 
python -u render.py --saveDir ../../data/outputs/render_output/hair_variation_selected/ --noHair --variation 'hair' --row_id  --res 128
done

```

**Note**. Location of skin layer models, lesion models, and parameter lists is specified in  `config.py`

## 4. Skin Model Generation

### Blood Network and Multi-layer Skin Model Generation

S-SYNTH relies on Houdini for procedural generation of blood network and multi-layer skin models. The steps below were
tested on Houdini version 19.5.640. The following steps can be used to reproduce the generation of the skin layers:

- Download skin_hair.hipnc file from Hugging
  Face ([ssynth_data/data/supporting_data/skin_hair.hipnc](https://huggingface.co/datasets/didsr/ssynth_data/blob/main/data/supporting_data/skin_hair.hipnc)):
  ```commandline
    cd S-SYNTH/code/data_generation
    python download_data.py --name 'skin_hair.hipnc' --saveDir '../../'
  ```
- In Houdini, open file skin_hair.hipnc (the file may take up to 30 minutes to load (you can see timer on lower left
  corner).
- Go to Windows -> Python Source Editor​
- Edit the count condition for the while loop on line 47 to adjust the number of models you want to export. Then hit
  Apply and Accept​.
- Each set of models takes anywhere between 10-60 minutes to generate.
- Click on the Python Shell Tab. If it’s not there, click on the + icon to the right of Geometry Spreadsheet, then open
  Python Shell as a New Pane Tab Type​
- Run the code hou.session.makeSkin()​. Alternatively, you can run this script in the python command line via:
  ```
  import hou
  hou.hipFile.load("skin_hair.hiplc")
  import os
  os.chdir('~/S-SYNTH/data/supporting_data/') # change as needed
  hou.session.makeSkin()
  ```

- And you’re done with the skin model generation!​ You’ll find the output skin layers (hypodermis, dermis, epidermis,
  hair and blood network) are in the form of .obj in ```../../data/outputs/outputs/outputModels/``` together with an output text file with
  the values of each mutable parameter.​ This path can also be updated in Python Source Editor (variable modelsOut
  in ```makeSkin()``` function).

### Growing Lesion Generation

The lesion model generation code can be found in `skinGrow3DCa.py`. Depending on the size of the lesion (# of
timesteps), this step can take from a few minutes to several hours.

   ```
   LESION_ID="1"
   python -u skinGrow3DCa.py --lesion_ID $LESION_ID --saveDir '../../data/outputs/lesion_growth_output/'
   ```

- The following parameters can be modified within the code under `settings` for the `skinLesion` class to change the
  lesion shape:

    - _origProbabilities_: initial probabilities of the inward, same, and outward planes for the seed along with the
      probabilities for the neighboring points (+2 steps)
    - _stepRange_: growing step (default: (1,2))
    - _gaussianSmooth_: controls smoothness of the lesions (default: 0)
    - _probabilityChangeStd_: standard deviation of the Gaussian distribution used to update the probabilities at each
      time point (default: 0.3)
    - _probabilityCancerP_: cancer probability, controls the probability of a cancer iteration to occur, it impacts the
      irregularity of the lesion shape. A cancer iteration is a new recursive growing on the corresponding cell,
      recursiveness and limits are controlled with the parameters below. (default: 0.001)
    - _cancerIterations_: cancer iterations (number of recursive iterations of the same growing algorithm on the cancer
      cell’s location) (default: 10)
    - _maxCancerRecursion_: maximum cancer recursion (maximum number of recursions for an irregular cell to trigger
      another recursive growth) (default: 3)
    - _Niter_: number of iterations (end timepoint for the lesion to grow that determines the size of the lesion) (
      default: 60)
    - _saveIterations_: time step at which the lesion models are saved (default: 5)

The output files are in the form of .png files, which represent different cross sections of each lesion, will be stored
in `../../data/outputs/models3D/`. The lesions cross-sections can be converted to 3D models (in .obj format) for image
rendering using Mitsuba. To convert the lesions:

- Open Houdini as described
  in [Blood Network and Multi-layer Skin Model Generation](#blood-network-and-multi-layer-skin-model-generation)
  section.
- Make sure the variable anaPath in ```convertLesions``` points to the location of the directory where raw lesions are
  stored. Houdini converts lesions into the form of .obj files using the following code:

  ```
  import hou
  hou.hipFile.load("skin_hair.hiplc")
  import os
  os.chdir('~/S-SYNTH/data/supporting_data/') # change as needed
  hou.session.convertLesions()
  ```

The output .obj files will be stored in `S-SYNTH/data/outputs/lesions3D/` to be used as a lesion
layer during rendering.    
**Note**. The above steps can be run either from Terminal or from Houdini (Windows->Python Shell).

### Material Generation

Sample material generation code can be found in `material_generation.ipynb`. Materials files
contain the spectral distribution of the absorption and scattering coefficients to be read by the 
Mitsuba 3 renderer. 







