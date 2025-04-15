import json
import shutil
import sys
import pandas as pd
import glob
import yaml

with open("image_visium_match.yaml", "r") as stream:
    he_to_visium = yaml.safe_load(stream)
visium_to_he = {v:k for k,v in he_to_visium.items()}

sample = str(sys.argv[1])
org_sample = sample.replace("-", "_")
out_folder = str(sys.argv[2])

image_file = f"data/LUNG_VISIUM/gdrive/{visium_to_he[org_sample]}_cropped_90-rotated_dsmpled2.tif"
json_file = f"data/LUNG_VISIUM/ebi_downloads/{org_sample}-spatial/scalefactors_json.json"

spot_diameter_fullres = json.load(open(json_file))["spot_diameter_fullres"]
spot_diameter_fullres


img_format = image_file.split(".")[-1]
json_out_path = f"{out_folder}/data/meta/{sample}.json"
img_out_path = f"{out_folder}/data/image/{sample}.{img_format}"

# move image
shutil.copy(image_file, img_out_path)


# general meta info about sample
json_info = {"SAMPLE": sample, "spot_diameter_fullres": spot_diameter_fullres, "dot_size": 5}
json_info

with open(json_out_path, 'w') as f:
    f.write(json.dumps(json_info))
