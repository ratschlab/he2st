import sys
import scanpy as sc
import pandas as pd
from pickle import load


model = str(sys.argv[1])
sample = str(sys.argv[2])
out_folder = str(sys.argv[3])


model_path = f"{out_folder}/prediction/{model}_label_model.pkl"
adata_path = f"{out_folder}/prediction/{model}/data/h5ad/{sample}.h5ad"
adata_out = f"{out_folder}/prediction/{model}/data/h5ad_annotated/{sample}.h5ad"

with open(model_path, 'rb') as f:
    classifier = load(f)

adata = sc.read_h5ad(adata_path)


classes = classifier.classes_
classes = [f"{c} Probability" for c in classes]
result = pd.DataFrame(classifier.predict_proba(adata.X),
                      columns=classes, index=adata.obs.index)
result["predicted_label"] = classifier.predict(adata.X)


adata.obs = adata.obs.merge(result, left_index=True, right_index=True)

adata.write_h5ad(adata_out)
