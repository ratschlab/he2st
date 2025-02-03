import json
import shutil
import sys
import pandas as pd
import glob

SAMPLE = str(sys.argv[1])
out_folder = str(sys.argv[2])

manual_position_file = f"data/o30773_SpaceRangerCount_v2_1_0_2023-05-18--18-23-28/manual_loupe_alignment/{SAMPLE}.json"
img_path = f"data/o30773_SpaceRangerCount_v2_1_0_2023-05-18--18-23-28/manual_loupe_alignment/{SAMPLE}.tif"

manual_position = json.load(open(manual_position_file))
manual_position = pd.DataFrame.from_dict(manual_position['oligo'])

scale_factors = manual_position.dia.values[0]

img_format = img_path.split(".")[-1]
json_out_path = f"{out_folder}/data/meta/{SAMPLE}.json"
img_out_path = f"{out_folder}/data/image/{SAMPLE}.{img_format}"

# move image
shutil.copy(img_path, img_out_path)


# general meta info about sample
json_info = {"SAMPLE": SAMPLE, "spot_diameter_fullres": scale_factors, "dot_size": 9}
json_info

with open(json_out_path, 'w') as f:
    f.write(json.dumps(json_info))
