import matplotlib.pyplot as plt
from tqdm import tqdm
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import torch
import glob
from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from scipy.stats import bootstrap
import numpy as np

from matplotlib import pyplot as plt
from gseapy import barplot, dotplot
from matplotlib_venn import venn2
from matplotlib_venn import venn3
import gseapy as gp
from sklearn.model_selection import KFold
import umap


def create_cross_validation_folds(patient_replicate_pairs,
                                  min_cv=3,
                                  random_state=2024,
                                  shuffle=True):
    folds = {}
    for i, fold in enumerate(patient_replicate_pairs.keys()):

        test = patient_replicate_pairs[fold]

        training = {k: v for k, v in patient_replicate_pairs.items() if v != test}

        kf = KFold(n_splits=min(min_cv, len(training)), shuffle=shuffle, random_state=random_state)
        training_keys = np.array(list(training.keys()))
        for j, (train_idx, test_idx) in enumerate(kf.split(training_keys)):
            validation_keys = training_keys[test_idx]
            validation = [training[k] for k in validation_keys]
            validation = [x for xs in validation for x in xs]

            internal_training_keys = training_keys[train_idx]
            internal_training = [training[k] for k in internal_training_keys]
            internal_training = [x for xs in internal_training for x in xs]

            assert len(np.intersect1d(validation, internal_training)) == 0

            folds[f"Fold_{i+1}_{j+1}"] = {"test": test, "validation": validation, "training": internal_training}

    return folds


def get_umap(emb):
    reducer = umap.UMAP()
    umap_emb = reducer.fit_transform(emb)
    return umap_emb


def normalize(x_list):
    x_scaled = scale(x_list)
    return x_scaled


def bootstrapping(x_list, n_resamples=100000):
    if len(x_list) == 1:
        # !!! case
        return [x_list.iloc[0], np.nan]
    res = bootstrap((x_list,), np.median, n_resamples=n_resamples)
    standard_error = res.standard_error
    median = np.median(res.bootstrap_distribution)
    return [median, standard_error]


def reorder_genes(adata, gene_name_idx):
    adata = adata.copy()
    adata = adata[:, adata.var.index.isin(gene_name_idx)]
    X_new = pd.DataFrame(np.zeros(shape=(len(gene_name_idx), len(adata))), index=gene_name_idx)
    X_new.loc[adata.var.index] = adata.X.T.toarray()
    return X_new.T


def load_multiple_pickles(files):
    return pd.concat([pd.read_pickle(f) for f in files])


def load_data(samples, out_folder, feature_model=None, load_image_features=True, factor=1e4, raw_counts=False):
    barcode_list = []
    image_features_emb = []
    gene_expr = []

    expr_files = [f"{out_folder}/data/inputX/{sample}.pkl" for sample in samples]

    gene_expr = load_multiple_pickles(expr_files)
    barcode_list = gene_expr.index.values

    if load_image_features:
        image_features_files = [f"{out_folder}/data/image_features/{sample}_{feature_model}.pkl" for sample in samples]
        image_features_emb = load_multiple_pickles(image_features_files)
        image_features_emb = image_features_emb.loc[barcode_list]

    data = {}
    data["barcode"] = barcode_list

    gene_expr = gene_expr.values

    if not raw_counts:
        gene_expr = (gene_expr.T / gene_expr.sum(axis=1)).T * factor
        gene_expr = np.log1p(gene_expr)

    data["y"] = gene_expr

    if load_image_features:
        data["X"] = image_features_emb.values

    return data


def plot_loss_values(train_losses, val_losses=None):
    train_losses = np.array(train_losses)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    train_idx = np.arange(0, len(train_losses))
    plt.plot(train_idx, train_losses, color="b", label="train")

    if val_losses is not None:
        val_losses = np.array(val_losses)
        val_idx = np.arange(0, len(val_losses)) * (len(train_losses) // len(val_losses) + 1)
        plt.plot(val_idx, val_losses, color="r", label="val")

    plt.legend()


def run_inference_from_dataloader(model, dataloader, device):
    model.to(device)  # same device
    model.eval()

    out = []

    with torch.no_grad():
        for X, _ in tqdm(dataloader):
            if type(X) is list:
                X = (x.to(device) for x in X)
            else:
                X = X.to(device)
            y = model(X)
            y = y.cpu().detach().numpy()

            out.extend(y)

    return np.array(out)


def run_inference(model, x, device):
    model.to(device)  # same device
    model.eval()
    with torch.no_grad():
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)
        out = model(x)
        return out.cpu().detach().numpy()


def preprocess_log_normalize(adata, normalize_target_sum=10_000):
    sc.pp.normalize_total(adata, target_sum=normalize_target_sum)
    sc.pp.log1p(adata)
    return adata


def preprocess_adata(adata, normalize_target_sum=10_000, pca_n_comps=50, run_dim_red=True):
    sc.pp.normalize_total(adata, target_sum=normalize_target_sum)
    sc.pp.log1p(adata)
    # sc.pp.scale(adata, max_value=10)
    if run_dim_red:
        sc.pp.pca(adata, n_comps=pca_n_comps)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    return adata


def extract_top_n_genes(adata, label, top_n):
    df = sc.get.rank_genes_groups_df(adata, group=label, key="wilcoxon")
    df = df[df.pvals_adj < 0.05]
    # df["pct_abs"] = abs(df.pct_nz_group - df.pct_nz_reference)
    df = df.sort_values(["logfoldchanges"], ascending=False)
    return df.names[:top_n]


def find_unique_elements(sets):
    unique_elements = []
    for i, current_set in enumerate(sets):
        other_sets = sets[:i] + sets[i + 1:]
        combined_others = set().union(*other_sets)
        unique_in_current = current_set - combined_others
        unique_elements.append(unique_in_current)
    return unique_elements


def find_genes(list_adata, list_groups, key, top_n=50):
    print(key)
    gene_groups = [set(extract_top_n_genes(adata, key, top_n)) for adata in list_adata]

    for group, unique_genes in zip(list_groups, find_unique_elements(gene_groups)):
        unique_genes = ', '.join(unique_genes)
        print(f"{group} unique: {unique_genes}")

    common_elements = ", ".join(set.intersection(*gene_groups))
    print(f"Shared genes: {common_elements}\n")
    return gene_groups


def plot_venn(list_adata, key, model, names=None, top_n=50, plot_names=False):

    gene_groups = [set(extract_top_n_genes(adata, key, top_n)) for adata in list_adata]
    if len(list_adata) == 2:
        venn = venn2(gene_groups, ('Original', model))
    elif len(list_adata) == 3:
        venn = venn3(gene_groups, ('Original', *model))
    else:
        print(f"Venn doesn't support it: {len(list_adata)}")

    if plot_names:
        common_elements = set.intersection(*gene_groups)
        if len(common_elements) > 0:
            venn.get_label_by_id('110').set_text('\n'.join(map(str, common_elements)))
            venn.get_label_by_id('110').set_fontsize(8)

    plt.title(f"{key} marker genes overlap")
    plt.show()

    return gene_groups


def plot_enrichr(adata, label, gene_sets):

    try:
        df = sc.get.rank_genes_groups_df(adata, group=label, key="wilcoxon")
        df = df[df.pvals_adj < 0.05]
        print("Signif. genes = ", len(df))
        gene_list = df.names.values.tolist()
        enr = gp.enrichr(gene_list=gene_list,
                         gene_sets=gene_sets,
                         organism='human',
                         outdir=None,  # don't write to disk
                         )

        cmap = plt.cm.get_cmap("tab20", len(gene_sets))
        dict(zip(gene_sets, [cmap(i) for i in range(len(gene_sets))]))

        ax = barplot(enr.results,
                     column="Adjusted P-value",
                     group='Gene_set',  # set group, so you could do a multi-sample/library comparsion
                     size=10,
                     top_term=10,
                     figsize=(10, 15),
                     title=f"{label}\nEnrichR",
                     color=dict(zip(gene_sets, [cmap(i) for i in range(len(gene_sets))]))
                     )
        plt.show()

    except Exception as e:
        print(e)


def compute_pearson_top_n(data, batch_key, genes_df,
                          top_n_interval=[5000, 3000, 2000,
                                          1000, 500, 400,
                                          300, 250,
                                          200, 150, 100]
                          ):
    score_top_n = []
    for batch in data[batch_key].unique():
        tab_batch = data[data[batch_key] == batch]
        for top_n in top_n_interval:

            top_n_genes = genes_df[genes_df.variances_norm_rank <= top_n]
            top_n_genes_score = tab_batch[tab_batch.gene.isin(top_n_genes.gene_name)].copy()
            top_n_genes_score["top_n"] = top_n
            score_top_n.append(top_n_genes_score)

    score_top_n = pd.concat(score_top_n)
    return score_top_n


def compute_area_under_pearson_top_n(data, batch_key, metric, n_resamples=1000):
    batch_scores = []
    for batch in tqdm(data[batch_key].unique()):
        tab_batch = data[data[batch_key] == batch]
        resample_scores = []
        for i in range(n_resamples):
            entity_scores = []
            for top_n in data.top_n.drop_duplicates().sort_values():
                scores = np.random.choice(tab_batch[tab_batch.top_n == top_n]
                                          [metric].values, size=len(tab_batch), replace=True)

                scores = np.median(scores)
                entity_scores.append([top_n, scores])

            entity_scores = np.array(entity_scores)
            resample_scores.append(auc(entity_scores[:, 0], entity_scores[:, 1]))
        resample_scores_mean = np.mean(resample_scores)
        resample_scores_std = np.std(resample_scores)
        batch_scores.append([batch, resample_scores_mean, resample_scores_std])
    batch_scores = pd.DataFrame(batch_scores, columns=[batch_key, f'auc_mean', f'auc_std'])
    return batch_scores


def select_highly_variable_genes(adata, n_top_genes,
                                 flavor="seurat_v3_paper",
                                 batch_key="sampleID",
                                 top_gene_interval=[5000, 4000, 3000, 2500, 2000,
                                                    1500, 1000, 500, 250, 200, 150,
                                                    100, 50, 25, 10, 5]):
    adata = adata.copy()
    adata.var["gene_name"] = adata.var.index.values
    tab = adata[:, adata.var.isPresentInAll].copy()

    sc.pp.highly_variable_genes(tab,
                                flavor=flavor,
                                batch_key=batch_key,
                                n_top_genes=n_top_genes)

    genes = adata.var.copy()
    genes["isPredicted"] = genes.gene_name.isin(tab[:, tab.var.highly_variable].var.gene_name)

    for top_n in top_gene_interval:
        sc.pp.highly_variable_genes(tab,
                                    flavor=flavor,
                                    batch_key=batch_key,
                                    n_top_genes=top_n)
        genes[f"top_{top_n}"] = genes.gene_name.isin(tab[:, tab.var.highly_variable].var.index)

    return genes
