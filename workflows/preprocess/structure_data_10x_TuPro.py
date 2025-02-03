import json
import shutil
import sys
import pandas as pd
import glob

SAMPLE = str(sys.argv[1])
org_sample_name = SAMPLE.replace('-1-', '-1_').replace('-2-', '-2_')
out_folder = str(sys.argv[2])

base_path = f"data/projects2022-digpathways_datasets/10x_tupro"
image_info_file = f"data/10x_image_info_updated.tsv"
img_path = glob.glob(f"data/tif_image/*/{org_sample_name}/{org_sample_name}.tif")[0]

df = pd.read_csv(image_info_file, sep="\t")
replicate_info = df[df.replicate == org_sample_name]
resolution = replicate_info.resolution.values[0]

tile_size = int(replicate_info.spot_diameter_fullres.values[0])
dot_size = int(replicate_info.dot_size_scanpy.values[0])

img_format = img_path.split(".")[-1]
json_out_path = f"{out_folder}/data/meta/{SAMPLE}.json"
img_out_path = f"{out_folder}/data/image/{SAMPLE}.{img_format}"

# move image
shutil.copy(img_path, img_out_path)


# general meta info about sample
json_info = {"SAMPLE": SAMPLE, "spot_diameter_fullres": tile_size, "dot_size": dot_size}
json_info

with open(json_out_path, 'w') as f:
    f.write(json.dumps(json_info))
