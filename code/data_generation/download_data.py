import argparse
import shutil
import os
import requests
from huggingface_hub import hf_hub_url
from huggingface_hub import HfFolder

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Data name", default="materials.zip")
parser.add_argument("--saveDir", type=str, help="Where to save data", default="../../")
parser.add_argument("--oasis", action='store_true')
parser.add_argument("--unzip", action='store_true')
args = parser.parse_args()

print("downloading data from huggingface...")
print("saving to " + str(args.saveDir))

# Build the remote and local paths
remote_filename = 'data/supporting_data/' + args.name
filename = os.path.basename(args.name)

if args.oasis:
    local_dir = os.path.join(args.saveDir, 'data/supporting_data/materials/oasis/')
else:
    local_dir = os.path.join(args.saveDir, 'data/supporting_data/', os.path.dirname(args.name))

os.makedirs(local_dir, exist_ok=True)
local_path = os.path.join(local_dir, filename)

# Get the download URL and token
url = hf_hub_url(
    repo_id="didsr/ssynth_data-test",
    filename=remote_filename,
    repo_type="dataset"
)

token = HfFolder.get_token()
headers = {"Authorization": f"Bearer {token}"} if token else {}

# Stream download to avoid memory issues with large files
response = requests.get(url, headers=headers, stream=True)
response.raise_for_status()

with open(local_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Saved to {local_path}")

if args.unzip:
    print("unzipping...")
    shutil.unpack_archive(local_path, local_dir, "zip")
    print("done!")
