import argparse
from huggingface_hub import snapshot_download


parser = argparse.ArgumentParser()
parser.add_argument("--saveDir", type=str, help="Where to save data", default="../../")
parser.add_argument("--name", type=str, help="dataset_name", required=True)

args = parser.parse_args()

# Download dataset split from huggingface
print(f"downloading {args.name} dataset from huggingface...")
print("saving to " + args.saveDir + '/')

snapshot_download(
    repo_id="didsr/ssynth_data",
    use_auth_token=True,
    repo_type="dataset",
    local_dir=args.saveDir ,  # data will be saved here
    allow_patterns='data/synthetic_dataset/hf_datasets/' + args.name + '/*',
)


