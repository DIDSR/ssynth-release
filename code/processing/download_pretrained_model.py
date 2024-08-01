import argparse
from huggingface_hub import hf_hub_download


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Data name", default="all_tones_real_ISIC_1.0_add_synth_0.2")
parser.add_argument("--saveDir", type=str, help="Where to save data", default="../../")
args = parser.parse_args()

# Download dataset from huggingface
print("downloading data from huggingface...")
print("saving to " + str(args.saveDir))
hf_hub_download(
    repo_id="didsr/ssynth_data",
    use_auth_token=True,
    repo_type="dataset",
    local_dir=args.saveDir,  # data will be saved here
    filename='data/pretrained_models/' + args.name + '/00/dsd_i01/n-dsd_i01_s-128_b-32_t-250_sc-linear_best.pth',
)
