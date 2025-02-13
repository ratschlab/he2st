import json
import shutil
import sys
import pandas as pd
import glob

SAMPLE = str(sys.argv[1])
out_folder = str(sys.argv[2])

json_path = f"data/meta/{SAMPLE}.json"
json_out_path = f"{out_folder}/data/meta/{SAMPLE}.json"
image_path = f"data/image/{SAMPLE}.tif"
image_out_path = f"{out_folder}/data/image/{SAMPLE}.tif"

# move image
shutil.copy(image_path, image_out_path)

# general meta info about sample

json_info = json.load(open(json_path))
json_info["SAMPLE"] = SAMPLE

with open(json_out_path, 'w') as f:
    f.write(json.dumps(json_info))
