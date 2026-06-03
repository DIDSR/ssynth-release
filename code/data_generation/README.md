# OASIS: Synthetic Skin Image Framework with Artifact Simulation

OASIS is a synthetic skin simulation framework capable of simulating five types of commonly
occurring artifacts in dermoscopic skin imaging: calibration chart, ruler, dark frame, hair,
and blood vessels. It is built on top of [S-SYNTH](https://github.com/DIDSR/ssynth-release),
an open-source tool for simulating dermoscopic images.

The S-SYNTH/OASIS framework involves the construction of a 3D digital object model comprising
skin tissue (epidermis, dermis, hypodermis), a blood network, hair, and a lesion. This process
is implemented in [Houdini](https://www.sidefx.com/) via a Python API. Once created, each model
is processed through [Mitsuba 3](https://www.mitsuba-renderer.org/).

The pipeline to generate synthetic skin images with artifacts is as follows:

1. Generate skin layer models, the lesion model, and optical materials (see [Section 1](#1-skin-layer-models-lesion-model-and-optical-properties)).
2. Import the skin and lesion models into Mitsuba to assign optical materials to each component
   and configure lighting conditions.
3. Generate artifacts and add them to the scene in Mitsuba (see [Section 2](#2-generate-artifacts)).
4. Render the synthetic skin images and corresponding lesion segmentation masks
   (see [Section 3](#3-generate-a-dataset)).

<p align="center">
<img src="../../images/Pipeline.png" width="500">
</p>

## Setup: Set Up Root Folder Path
   ```
   cd ssynth-dev
   CWD=$(pwd)
   ```

## 1. Skin Layer Models, Lesion Model, and Optical Properties

You can either download and use pre-generated models and materials (Section 1.1) or generate your own (Section 1.2).
### 1.1. Use Pre-Generated Models and Optical Materials

We have released pre-generated models of skin layer components (epidermis, dermis, hypodermis,
blood network, and hair), lesion models, and their optical materials, with properties randomly
chosen from predefined ranges. The available variations are shown in the table below.

| **Parameter** | **Variations** |
|---|---|
| Each skin model component (epidermis, dermis, hypodermis, hair, blood network) | 100 |
| Lesion model | 20 |
| Melanosome fraction (i.e., epidermis material) | 50 |
| Blood fraction (i.e., dermis material) | 4 |
| Hair material | 3 |
| Hypodermis material | 1 |
| Blood network material | 1 |
| Lesion material | 20 |

These files can be downloaded from Hugging Face
([ssynth_data/data/supporting_data/materials.zip](https://huggingface.co/datasets/didsr/ssynth_data/blob/main/data/supporting_data/materials.zip)).

- Set up a Hugging Face token for data download:
  - Follow the instructions at
    [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens)
    to create a token.
  - Run and paste your token:
   ```
   huggingface-cli login
   ```

Download and extract data
   ```
   cd $CWD/code/data_generation
   python download_data.py --name 'materials/oasis/lesions_release.zip' --saveDir '../../' --oasis --unzip
   python download_data.py --name 'materials/oasis/opticalMaterials.zip' --saveDir '../../' --oasis --unzip
   python download_data.py --name 'materials/oasis/outputModels.zip' --saveDir '../../' --oasis --unzip
   python download_data.py --name 'materials/oasis/skin_layers.zip' --saveDir '../../' --oasis --unzip
   python download_data.py --name 'materials/oasis/skin_layers_bloodVessel_artifact.zip' --saveDir '../../' --oasis --unzip
   python download_data.py --name 'materials/oasis/skin_layers_hair_artifact.zip' --saveDir '../../' --oasis --unzip

   python download_data.py --name 'hdri.zip' --saveDir '../../' --unzip
   python download_data.py --name 'params_lists.zip' --saveDir '../../' --unzip
   python download_data.py --name 'sample_data.csv' --saveDir '../../'
   ```
  - The resulting `materials/oasis/` folder consists of 3D skin layer models and optical materials.
   There are three sets of skin layer models:
      - skin_layers: used for all images except those with hair or blood vessel artifacts.
      - skin_layers_hair: used for images with the hair artifact.
      - skin_layers_bloodVessel: used for images with the blood vessel artifact.

- The folder structure is as follows:
    - Epidermis models: `materials/oasis/skin_layers*/epidermis*`
    - Vascular models: `materials/oasis/skin_layers*/vascular*`
    - Dermis models: `materials/oasis/skin_layers*/dermis*`
    - Hair models: `materials/oasis/skin_layers*/hair*`
    - Lesion models: `materials/oasis/lesions/lesion*`
    - Optical materials for each skin layer: `materials/oasis/opticalMaterials/`
    - Lighting conditions (represented via collection of High Dynamic Range Imaging (HDRI) images): ```hdri/```
    - Sample set of parameters to generate two images (csv): ```sample_data.csv```
    - (_Optional_): Lists of parameters used to generate various image sets used in the paper: ```params_lists/``` See
      Section 3 ([Generate a Dataset](#3-generate-a-dataset))

These files should be stored in `data/supporting_data/` to be used for rendering in the next steps. If you
change this directory, please update ```config.py```.

### 1.2. Generate New Models and Materials
If you prefer to use the pre-generated models, download them as described in Section 1.1 and
proceed to Section 3.

#### 1.2.1. Generate New Skin Models
We use Houdini (version 19.5.640) to simulate five distinct skin components: epidermis, dermis,
hypodermis, blood network, and hair. 


- Download the appropriate Houdini file from Hugging Face:

   - `skin.hiplc`: for artifact-free skin images or images with calibration chart, ruler, or dark frame artifacts.
   - `skin_hair_artifact.hiplc`: for images with the hair artifact.
   - `skin_bloodVessel_artifact.hiplc`: for images with the blood vessel artifact.

```
cd code/data_generation
python download_data.py --name 'skin.hiplc' --saveDir '../../'
python download_data.py --name 'skin_hair_artifact.hiplc' --saveDir '../../'
python download_data.py --name 'skin_bloodVessel_artifact.hiplc' --saveDir '../../'
```

- In Houdini, open the file (the file may take up to 30 minutes to load (you can see timer on lower left corner).

- Go to Windows → Python Source Editor.

- Edit the count condition for the while loop on line 47 to adjust the number of models you want to export. Then hit Apply and Accept​.

- Each set of models takes anywhere between 10-60 minutes to generate.

- Click on the Python Shell Tab. If it’s not there, click on the + icon to the right of Geometry Spreadsheet, then open Python Shell as a New Pane Tab Type​

- Run the code `hou.session.makeSkin()​`. Alternatively, you can run this script in the python command line via:
   ```
   import hou
   hou.hipFile.load("skin.hiplc")
   import os
   os.chdir('data/supporting_data/') # change as needed
   hou.session.makeSkin()
   ```

- Output skin layers (hypodermis, dermis, epidermis, hair, and blood network) will be saved as
`.obj` files in `../../data/supporting_data/materials/skin_layers/`, along with a text file
recording the values of each mutable parameter. This path can be updated in the Python Source
Editor via the `modelsOut` variable in the `makeSkin()` function.

#### 1.2.2. Generate New Growing Lesion Models
If you prefer to use the pre-generated lesion models, download them as described in Section 1.1
and proceed to Section 3.

The lesion model generation code is in `skinGrow3DCa.py`. Depending on the lesion size (number
of timesteps), this step can take from a few minutes to several hours.

```
LESION_ID="1"
python -u skinGrow3DCa.py --lesion_ID $LESION_ID --saveDir '../../data/supporting_data/materials/lesionModels3D/'
```

The following parameters can be modified within the code under `settings` for the `skinLesion` class to change the lesion shape:

    - `origProbabilities`: initial probabilities of the inward, same, and outward planes for the seed along with the probabilities for the neighboring points (+2 steps)
    - `stepRange`: growing step (default: (1,2))
    - `gaussianSmooth`: controls smoothness of the lesions (default: 0)
    - `probabilityChangeStd`: standard deviation of the Gaussian distribution used to update the probabilities at each time point (default: 0.3)
    - `probabilityCancerP`: cancer probability, controls the probability of a cancer iteration to occur, it impacts the irregularity of the lesion shape. A cancer iteration is a new recursive growing on the corresponding cell, recursiveness and limits are controlled with the parameters below. (default: 0.001)
    - `cancerIterations`: cancer iterations (number of recursive iterations of the same growing algorithm on the cancer cell’s location) (default: 10)
    - `maxCancerRecursion`: maximum cancer recursion (maximum number of recursions for an irregular cell to trigger another recursive growth) (default: 3)
    - `Niter`: number of iterations (end timepoint for the lesion to grow that determines the size of the lesion) (default: 60)
    - `saveIterations`: time step at which the lesion models are saved (default: 5)

Output .png files representing cross-sections of each lesion will be stored in
`../../data/supporting_data/materials/lesionModels3D/`. These cross-sections can be converted
to 3D .obj models for rendering in Mitsuba:

- Open Houdini as described in section 1.2.1.
- Ensure the variable anaPath in `convertLesions` points to the location of the directory where raw lesions (`.png` files) are stored. Then run:

  ```
  import hou
  hou.hipFile.load("skin.hiplc")
  import os
  os.chdir('~/OASIS/data/supporting_data/') # change as needed
  hou.session.convertLesions()
  ```

Output .obj files will be stored in `data/supporting_data/materials/lesions/` for use
as the lesion layer during rendering.

**Note**. The above steps can be run either from Terminal or from Houdini (Windows->Python Shell).

#### 1.2.3. Generate New Materials
If you prefer to use the pre-generated materials, download them as described in Section 1.1 and
proceed to Section 3.

Sample material generation code can be found in `material_generation.ipynb`. Materials files contain the spectral distribution of the absorption and scattering coefficients to be read by the Mitsuba 3 renderer.

## 2. Generate Artifacts
We generated five different artifacts specific to dermoscopic images that challenge the skin lesion segmentation task. Three of the artifacts (calibration chart, ruler, or dark frame) were created by adding a component to the object space using Mitsuba before rendering the images. The other two artifacts (hair and blood vessels) were created by changing the mutable parameters of the skin models in Houdini, and then transferring the new models to Mitsuba to complete the remaining steps. 

### 2.1. Calibration Chart
A sphere representing the calibration chart was added as an object to the `scene` using `scene['calibration_chart']` within `util.py`, with a random size and location. This artifact will be added to the scene if `id_calChart` flag is set to 1 and takes the properties of the sphere as defined by `calChart_params`.

**Note**. For the manuscript, we kept both variables of size and location of calibration charts within pre-defined limits (radius: 2-5 mm, y-location of center: 1-5 mm from the edges, x-location of center: 2-4 mm from the edges) to prevent overlapping with the skin lesion, with their colors randomly selected from two different shades of blue or orange (see the `calChart_params` column within any of the parameter lists). In addition, this artifact was only added to the scene when the lesion covered approximately less than 50\% of the skin surface, ensuring that it did not obscure the lesion (see `render.py`).

### 2.2. Ruler
We added 15 rectangles oriented vertically, and one rectangle oriented horizontally using Mitsuba and placed them on the top of the epidermis to represent the ruler artifact in the dermoscopic images. The horizontal line was added using `scene[line_h]`, and the vertical lines were added using `scene[name]` with `name = 'line' + str(idx)`, where the idx corresponded to each of vertical lines in `util.py`. 

**Note**. The ruler's position was randomized in both the x and y directions, with the y-location and x-location to be restricted to 2-6 mm and 2-5 mm from the edges, respectively (see the `ruler_params` column within any of the parameter lists). Similarly to the calibration chart, the ruler artifact was only added in scenes with skin lesions smaller than a specified size (see `render.py`).

### 2.3. Dark Frame
The dark frame artifact was generated by placing a black torus below the sensor in Mitsuba by defining `scene['frame']` within `util.py`. 

**Note**. The radius of the torus was changed as a function of the distance between the sensor and the skin model to simulate various degrees of dark frame coverage (see `get_frame(id_origin_y)` within `util.py`). Similarly to the calibration chart and ruler, the dark frame was only added to the scenes with small lesions (see `render.py`).

### 2.4. Hair
Images with hair artifact were generated by adjusting the mutable parameters related to the hair properties in Houdini. This process involved increasing the pre-defined ranges for hair scatter seed and hair density by a factor of two and generating the hair models at greater curve angles compared to the baseline models:

| **Parameter** | **Min** | **Max** | **Unit** |
|--------------|--------------|--------------|--------------|
| Hair density value | 50 | 90 | |
| Hair scatter seed  | 0 | 20 | |
| Hair length | 4 | 15 |
| Hair bend curve angle | 0 | 60 | degrees |
| Hair bend curve random angle | 30 | 90 | degrees |

### 2.5. Blood Vessels
The blood vessel artifact was added by modifying the blood network and dermis layers in Houdini. We increased the number of random seeds, starting and ending points for both the deep and upper blood networks. To enhance the visibility of the blood vessels on the skin, we reduced the thickness of the dermis layer compared to the baseline models and limited the melanosome fraction to small values (less than 0.05 corresponding to only 5 variations instead of 50 variations used for the other images) representing lighter skin tones. The other skin components and optical properties remained the same.

| **Parameter** | **Min** | **Max** | **Unit** |
|--------------|--------------|--------------|--------------|
| Deep blood network start number of points | 40 | 60 | |
| Deep blood network start scatter seed  | 0 | 20 | |
| Deep blood network end number of points  | 500 | 900 |
| Deep blood network end scatter seed | 0 | 20 |  |
| Upper blood network number of points | 40 | 60 |  |
| Upper blood network scatter seed | 0 | 20 |  |
| Dermis thickness | 0.3 | 2.3 |  |


## 3. Generate a Dataset

### 3.1. Parameter List
Each dataset is specified by a parameter list (in `.csv` format), where each row describes the set of parameters
needed to render each image in the datasets used to generate results from the manuscript. We provide pre-generated parameter lists on Hugging Face
(`supporting_data/params_lists/`). These files must be stored inside
`supporting_data/param_lists/` to be used for rendering. The following parameter lists are
available for download:
- `oasis_all_examples.csv`: 10,000 combinations of random parameters to generate images with each of the artifacts (except blood vessel)
- `oasis_all_examples_bloodVessel.csv`: 10,000 combinations of random parameters to generate images with the blood vessel artifact

- The following parameters must be provided to generate each skin image and segmentation mask with a specific artifact:
  - `id_model`: skin models ID (used for epidermis, vascular, dermis, hypodermis)
  - `id_hairModel`: hair model ID
  - `id_lesion`: lesion model ID
  - `id_timePoint`: timepoint for the growing lesion, which determines the size of the lesion
  - `id_lesionMat`: lesion material ID
  - `id_fracBlood`: blood fraction value
  - `id_mel`: melanosome fraction value
  - `id_light`: light condition ID
  - `id_hairAlbedo`: hair albedo index
  - `mi_variant`: mitsuba rendering technique (we used spectral for all experiments ('cuda_spectral' or 'scalar_spectral'), both generate same results, but first is faster)
  - `origin_y`: camera position (used for rendering)
  - `calChart_params`: calibration chart artifact properties [x-location of center, y-location of center, radius, color], where color values of 0 and 1 correspond to orange and blue, respectively
  - `ruler_params`: ruler artifact properties [x-location of center, y-location of center]

These values will be mapped to a specific skin model and material in the rendering step (see `util.py`).

**Note.** A new parameter list can also be created using `create_parameter_list.py`, which generates a random parameter list based on the pre-generated skin layer models available on Hugging Face (see Section 1.1).

```
python create_parameter_list.py --variation "original"
```
This script accepts the following argument:
- --variation (str, required): "original" or "bloodVessel" (difference is the melanosome fraction range which is limited for blood vessel)

The parameters mentioned above can be randomly selected from pre-defined ranges (see ```./util.py```, ```under get_l_<PARAMETER>```).

### 3.2. Rendering
Synthetic skin images and their corresponding masks can be rendered using render.py. An
example command:
```
python -u render.py --saveDir ../../data/images_and_masks/10k_vesselDensity/ --row_id 0 --numRun 10 --bloodVessel --oasis
```

This script accepts the following arguments:

 - --saveDir (str, required): directory to save outputs
 - --sch (str, default: 'sge'): type of scheduler to read paras
 - --res (int, default=128): resolution of the generated images
 - --noHair (flag): enables removing hair
 - --frame (flag): enables adding dark frame artifact to the image
 - --CalChart (flag): enables adding calibration chart artifact to the image
 - --ruler (flag): enables adding ruler artifact to the image
 - --hairDense (flag): enables adding hair artifact to the image
 - --bloodVessel (flag): enables adding blood vessel artifact to the image
 - --numRun (int, default=1): number of jobs to run

**Note:** The following scripts are provided in `/scripts` folder to generate 10k images with each type of the artifacts:
    - submit_render_10k_cuda_spectral_bloodVessel.sbatch
    - submit_render_10k_cuda_spectral_calChart.sbatch
    - submit_render_10k_cuda_spectral_frame.sbatch
    - submit_render_10k_cuda_spectral_hair.sbatch
    - submit_render_10k_cuda_spectral_None.sbatch
    - submit_render_10k_cuda_spectral_ruler.sbatch
