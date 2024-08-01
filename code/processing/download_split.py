import argparse
from huggingface_hub import snapshot_download


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Data name", default="all_tones_real_ISIC_1.0_add_synth_0.2")
parser.add_argument("--saveDir", type=str, help="Where to save data", default="../../")
args = parser.parse_args()

# Download dataset split from huggingface
print("downloading data from huggingface...")
print("saving to " + args.saveDir + '/' + args.name)
snapshot_download(
    repo_id="didsr/ssynth_data",
    use_auth_token=True,
    repo_type="dataset",
    local_dir=args.saveDir ,  # data will be saved here
    allow_patterns='data/dataset_splits/' + args.name + '/*.txt',
)


