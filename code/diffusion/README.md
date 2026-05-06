# Diffusion Model Experiments

## Set Path
    ```
    cd ssynth-release
    CWD=$(pwd)
    cd $CWD/code/diffusion
    ```
## Evaluate Model
- Download HAM data (see instructions in in ssynth-release/tree/main/code/processing/README.md)

- Evaluate model:
    ```
    jupyter notebook evaluate_diffusion_model.ipynb
    ```

## Train Model (Optional)

- Download dataset
    ```
    python download_skin_hf_dataset.py --name 'hair_ssynth_smart_train' --saveDir '../../'
    ```

- Configure: 
    ```
    accelerate config # (we trained on 4 GPUs)
    ```
- Train model (adding --multi_gpu if appropriate)
    ```
    export HF_HOME=$CWD'/data/huggingface_cache/'
    export HUGGINFACE_HUB_CACHE=$CWD'/data/huggingface_cache/'
    
    export MODEL_NAME="timbrooks/instruct-pix2pix"
    export DATASET_ID=$CWD"/data/synthetic_dataset/hf_datasets/hair_ssynth_smart_train/"
    export OUTPUT_DIR=$CWD"/data/outputs/models/"

    export HF_HUB_OFFLINE=True    # Optional: change to True only when training on a cluster with no internet
    export DISABLE_TELEMETRY=YES  # Optional: disable reporting

    accelerate launch --mixed_precision="fp16"  finetune_instruct_pix2pix.py \
     --pretrained_model_name_or_path=$MODEL_NAME \
     --dataset_name=$DATASET_ID \
     --use_ema \
     --enable_xformers_memory_efficient_attention \
     --resolution=512 --random_flip \
     --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
     --max_train_steps=15000 \
     --checkpointing_steps=5000 --checkpoints_total_limit=1 \
     --learning_rate=5e-05 --lr_warmup_steps=0 \
     --conditioning_dropout_prob=0.05 \
     --mixed_precision=fp16 \
     --seed=42 \
     --output_dir=$OUTPUT_DIR \
     --cache_dir=$HUGGINFACE_HUB_CACHE
    ```


## Create Dataset (Optional)
    - Download raw data from HuggingFace
    ```
    python ../processing/download_ssynth.py --name '10k_hairDensity.zip' --saveDir '../../' --unzip
    python ../processing/download_ssynth.py --name '10k_None.zip' --saveDir '../../' --unzip
    python ../processing/download_split.py --name 'all_tones_real_HAM_1.0_synth_only_1.0_hairDensity_10k' --saveDir '../../'
    ```
    
    - Run example scripts for creating a HuggingFace dataset from S-SYNTH data to train the model above
    ```
    jupyter notebook create_artifact_dataset.ipynb
    ```
