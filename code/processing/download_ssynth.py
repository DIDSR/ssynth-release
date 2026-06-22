import shutil
import argparse
from huggingface_hub import hf_hub_download


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Data name", default="output_10k.zip")
parser.add_argument("--saveDir", type=str, help="Where to save data", default="../../")
parser.add_argument("--unzip", action='store_true')
args = parser.parse_args()

# Download dataset from huggingface
print("downloading data from huggingface...")
print("saving to " + str(args.saveDir))
hf_hub_download(
    repo_id="didsr/ssynth_data",
    use_auth_token=True,
    repo_type="dataset",
    local_dir=args.saveDir,  # data will be saved here
    filename='data/synthetic_dataset/' + args.name,
)

if args.unzip:
    print("unzipping...")
    shutil.unpack_archive(args.saveDir + 'data/synthetic_dataset/' + args.name,
                          args.saveDir + 'data/synthetic_dataset/', "zip")
    print("done!")
