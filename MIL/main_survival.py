from __future__ import print_function

import argparse
import os
from timeit import default_timer as timer
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import wandb

# internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.survival_core_utils import train
from dataset.dataset_survival import Generic_MIL_Survival_Dataset


# =========================================================
#                 H5 filename mapping helpers
# =========================================================
def build_slide_id_map_from_h5(data_dir: str):
    """
    从 h5 文件名建立映射：
        短 slide_id -> 实际 h5 basename（不含 .h5）
    """
    def list_h5_basenames(folder):
        names = []
        for fn in os.listdir(folder):
            if fn.endswith(".h5"):
                names.append(os.path.splitext(fn)[0])
        return sorted(names)

    names = sorted(set(list_h5_basenames(data_dir)))
    id_map = {}

    for full_name in names:
        parts = full_name.split("-")
        candidates = set()

        candidates.add(full_name)
        candidates.add(full_name[:12])

        if len(parts) >= 3:
            candidates.add("-".join(parts[:3]))
        if len(parts) >= 4:
            candidates.add("-".join(parts[:4]))
        if len(parts) >= 5:
            candidates.add("-".join(parts[:5]))

        for key in candidates:
            if key in id_map and id_map[key] != full_name:
                print(f"[Warn] ambiguous key '{key}' -> '{id_map[key]}' and '{full_name}', keep first one.")
                continue
            id_map[key] = full_name

    print(f"[build_slide_id_map_from_h5] total usable h5 names = {len(names)}")
    print(f"[build_slide_id_map_from_h5] total mapped keys = {len(id_map)}")
    return id_map


def normalize_dataset_slide_ids(dataset, slide_id_map):
    """
    将 dataset.slide_data['slide_id'] 从短 ID 映射成真实 h5 basename。
    """
    if not hasattr(dataset, "slide_data") or not isinstance(dataset.slide_data, pd.DataFrame):
        print("[normalize_dataset_slide_ids] dataset has no slide_data, skip.")
        return dataset

    if "slide_id" not in dataset.slide_data.columns:
        print("[normalize_dataset_slide_ids] no 'slide_id' column, skip.")
        return dataset

    sd = dataset.slide_data.copy()
    unique_full_names = list(set(slide_id_map.values()))

    resolved_ids = []
    matched = 0

    for sid in sd["slide_id"].astype(str).tolist():
        resolved = slide_id_map.get(sid, sid)

        if resolved == sid:
            hits = [v for v in unique_full_names if sid in v]
            if len(hits) == 1:
                resolved = hits[0]
            elif len(hits) > 1:
                hits = sorted(hits, key=len)
                resolved = hits[0]
                print(f"[Warn] multiple h5 matches for slide_id='{sid}', choose '{resolved}'")

        if resolved != sid:
            matched += 1
        resolved_ids.append(resolved)

    sd["slide_id"] = resolved_ids
    dataset.slide_data = sd

    print(f"[normalize_dataset_slide_ids] matched {matched}/{len(sd)} slide_id entries")
    print("[normalize_dataset_slide_ids] first 10 slide_ids:", sd["slide_id"].head(10).tolist())
    return dataset


def normalize_split_csv_ids(split_csv_path: str, slide_id_map: dict, out_path: str = None):
    """
    将 split csv 中的短 ID 映射成真实 h5 basename。
    """
    df = pd.read_csv(split_csv_path)
    unique_full_names = list(set(slide_id_map.values()))

    def resolve_one(x):
        if pd.isna(x):
            return x
        s = str(x)

        if s in slide_id_map:
            return slide_id_map[s]

        hits = [v for v in unique_full_names if s in v]
        if len(hits) == 1:
            return hits[0]
        elif len(hits) > 1:
            hits = sorted(hits, key=len)
            print(f"[Warn] split id '{s}' has multiple matches, choose '{hits[0]}'")
            return hits[0]

        return s

    if all(c in df.columns for c in ["train", "val", "test"]):
        for col in ["train", "val", "test"]:
            df[col] = df[col].apply(resolve_one)
    elif "slide_id" in df.columns:
        df["slide_id"] = df["slide_id"].apply(resolve_one)
    elif "case_id" in df.columns:
        df["case_id"] = df["case_id"].apply(resolve_one)
    else:
        print(f"[normalize_split_csv_ids] unrecognized split format: {df.columns.tolist()}")

    if out_path is None:
        out_path = split_csv_path.replace(".csv", "_normalized.csv")

    df.to_csv(out_path, index=False)
    print(f"[normalize_split_csv_ids] saved -> {out_path}")
    return out_path


# =========================================================
#                         MAIN
# =========================================================
def main(args, dataset, slide_id_map):
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    wandb.init(project=args.task)
    wandb.config.update(vars(args))

    start_fold = 0 if args.k_start == -1 else args.k_start
    end_fold = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start_fold, end_fold)

    latest_test_cindex = []
    latest_val_cindex = []

    for i in folds:
        t0 = timer()
        seed_torch(args.seed)

        results_pkl_path = os.path.join(args.results_dir, f"split_latest_val_{i}_results.pkl")
        if os.path.isfile(results_pkl_path):
            print("Skipping Split %d" % i)
            continue

        split_csv = os.path.join(args.split_dir, f"splits_{i}.csv")
        print(f"[Fold {i}] split_csv = {split_csv}")
        if not os.path.isfile(split_csv):
            raise FileNotFoundError(f"Cannot find split file: {split_csv}")

        norm_split_csv = normalize_split_csv_ids(split_csv, slide_id_map)
        split_df = pd.read_csv(norm_split_csv)
        print(f"[Fold {i}] normalized split columns = {split_df.columns.tolist()}")
        print(split_df.head())

        if hasattr(dataset, "slide_data"):
            print(f"[Fold {i}] dataset.slide_data columns = {dataset.slide_data.columns.tolist()}")
            if "slide_id" in dataset.slide_data.columns:
                print(f"[Fold {i}] dataset slide_id sample:",
                      dataset.slide_data["slide_id"].astype(str).head(10).tolist())

        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            args.backbone,
            args.patch_size,
            from_id=False,
            csv_path=norm_split_csv
        )

        if train_dataset is None or len(train_dataset) == 0:
            raise ValueError(
                f"[Fold {i}] train_dataset is empty after return_splits. "
                f"Please check split IDs and dataset.slide_data['slide_id']."
            )

        if args.k_fold:
            print("training: {}, validation: {}".format(len(train_dataset), 0 if val_dataset is None else len(val_dataset)))
            datasets = (train_dataset, val_dataset)
        else:
            print("training: {}, validation: {}, testing: {}".format(
                len(train_dataset),
                0 if val_dataset is None else len(val_dataset),
                0 if test_dataset is None else len(test_dataset)
            ))
            datasets = (train_dataset, val_dataset, test_dataset)

        if args.preloading == 'yes':
            for d in datasets:
                if d is not None:
                    d.pre_loading()

        if args.task_type == 'survival':
            if args.k_fold:
                cindex_val = train(datasets, i, args)
                latest_val_cindex.append(cindex_val)
            else:
                results, cindex_test, cindex_val = train(datasets, i, args)
                latest_val_cindex.append(cindex_val)
                latest_test_cindex.append(cindex_test)

                filename = os.path.join(args.results_dir, f"split_{i}_results.pkl")
                save_pkl(filename, results)

        print(f"[Fold {i}] done. time={timer() - t0:.1f}s")

    if args.k_fold:
        final_df = pd.DataFrame({
            "folds": folds,
            "val_cindex": latest_val_cindex
        })
    else:
        final_df = pd.DataFrame({
            "folds": folds,
            "test_cindex": latest_test_cindex,
            "val_cindex": latest_val_cindex
        })

    save_name = f"summary_partial_{start_fold}_{end_fold}.csv" if len(folds) != args.k else "summary.csv"
    out_csv = os.path.join(args.results_dir, save_name)

    if not args.k_fold and len(final_df) > 0:
        mean_test = final_df["test_cindex"].mean()
        std_test = final_df["test_cindex"].std()

    mean_val = final_df["val_cindex"].mean() if len(final_df) > 0 else np.nan
    std_val = final_df["val_cindex"].std() if len(final_df) > 0 else np.nan

    if args.k_fold:
        df_append = pd.DataFrame({
            "folds": ["mean", "std"],
            "val_cindex": [mean_val, std_val]
        })
    else:
        df_append = pd.DataFrame({
            "folds": ["mean", "std"],
            "test_cindex": [mean_test, std_test],
            "val_cindex": [mean_val, std_val]
        })

    final_df = pd.concat([final_df, df_append], ignore_index=True)
    final_df.to_csv(out_csv, index=False)

    final_df["folds"] = final_df["folds"].astype(str)
    table = wandb.Table(dataframe=final_df)
    wandb.log({"summary": table})

    if args.k_fold:
        wandb.log({"mean_val_cindex": mean_val})
    else:
        wandb.log({"mean_test_cindex": mean_test, "mean_val_cindex": mean_val})

    print("Saved:", out_csv)


# =========================================================
#                         ARGS
# =========================================================
parser = argparse.ArgumentParser(description="WSI Survival Training using saved .h5 features only")

# Generic training settings
parser.add_argument("--data_root_dir", type=str, default=None,
                    help="Data directory to WSI .h5 features")
parser.add_argument("--max_epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--label_frac", type=float, default=1.0)
parser.add_argument("--reg", type=float, default=1e-5)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--k_start", type=int, default=-1)
parser.add_argument("--k_end", type=int, default=-1)
parser.add_argument("--results_dir", default="./results")
parser.add_argument("--split_dir", type=str, default=None)
parser.add_argument("--log_data", action="store_true", default=False)
parser.add_argument("--testing", action="store_true", default=False)
parser.add_argument("--early_stopping", action="store_true", default=False)
parser.add_argument("--opt", type=str, choices=["adam", "sgd"], default="adam")
parser.add_argument("--drop_out", type=float, default=0.25)
parser.add_argument("--gc", type=int, default=32)
parser.add_argument("--bag_loss", type=str, choices=["svm", "ce", "ce_surv", "nll_surv", "cox_surv"], default="nll_surv")
parser.add_argument("--model_type", type=str,
                    choices=["mean_mil", "max_mil", "att_mil", "trans_mil", "s4model", "mamba_mil"],
                    default="mamba_mil")
parser.add_argument("--mode", type=str, choices=["path", "omic", "pathomic", "cluster"], default="path")
parser.add_argument("--apply_sig", action="store_true", default=False)
parser.add_argument("--apply_sigfeats", action="store_true", default=False)
parser.add_argument("--fusion", type=str, choices=["None", "concat", "bilinear"], default="None")
parser.add_argument("--exp_code", type=str)
parser.add_argument("--weighted_sample", action="store_true", default=False)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--no_inst_cluster", action="store_true", default=False)
parser.add_argument("--alpha_surv", type=float, default=0.0)
parser.add_argument("--reg_type", type=str, choices=["None", "omic", "pathomic"], default="None")
parser.add_argument("--lambda_reg", type=float, default=1e-4)
parser.add_argument("--inst_loss", type=str, choices=["svm", "ce", None], default=None)
parser.add_argument("--subtyping", action="store_true", default=False)
parser.add_argument("--bag_weight", type=float, default=0.7)
parser.add_argument("--B", type=int, default=8)
parser.add_argument("--backbone", type=str, default="resnet50")
parser.add_argument("--patch_size", type=str, default="")
parser.add_argument("--preloading", type=str, default="no")
parser.add_argument("--in_dim", type=int, default=1024)
parser.add_argument("--k_fold", action="store_true", default=False)

parser.add_argument("--csv_path", type=str, required=True,
                    help="Path to survival csv file")
parser.add_argument("--label_col", type=str, required=True,
                    help="Label column name in csv, e.g. survival_months or futime")

# mambamil
parser.add_argument("--mambamil_rate", type=int, default=10)
parser.add_argument("--mambamil_layer", type=int, default=2)
parser.add_argument("--mambamil_type", type=str, default="SRMamba", choices=["Mamba", "BiMamba", "SRMamba"])

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is:", device)


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

print("Experiment Name:", args.exp_code)
print("csv_path:", args.csv_path)
print("label_col:", args.label_col)

settings = {
    "num_splits": args.k,
    "k_start": args.k_start,
    "k_end": args.k_end,
    "task": args.task,
    "max_epochs": args.max_epochs,
    "results_dir": args.results_dir,
    "lr": args.lr,
    "experiment": args.exp_code,
    "reg": args.reg,
    "label_frac": args.label_frac,
    "bag_loss": args.bag_loss,
    "seed": args.seed,
    "model_type": args.model_type,
    "use_drop_out": args.drop_out,
    "weighted_sample": args.weighted_sample,
    "opt": args.opt,
    "csv_path": args.csv_path,
    "label_col": args.label_col,
}

print("\nLoad Dataset")

if "survival" in args.task:
    args.n_classes = 300
    dataset = Generic_MIL_Survival_Dataset(
        csv_path=args.csv_path,
        mode=args.mode,
        apply_sig=args.apply_sig,
        data_dir=args.data_root_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        patient_strat=False,
        n_bins=4,
        label_col=args.label_col,
        ignore=[]
    )

    slide_id_map = build_slide_id_map_from_h5(args.data_root_dir)
    dataset = normalize_dataset_slide_ids(dataset, slide_id_map)

    args.task_type = "survival"

    if hasattr(dataset, "slide_data"):
        print("slide_data columns:", list(dataset.slide_data.columns))
        print(dataset.slide_data.head())
else:
    raise NotImplementedError("Only survival task is implemented in this script.")

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + "_s{}".format(args.seed))
os.makedirs(args.results_dir, exist_ok=True)

if args.split_dir is None:
    args.split_dir = os.path.join("splits", args.task + "_{}".format(int(args.label_frac * 100)))

print("split_dir:", args.split_dir)
if not os.path.isdir(args.split_dir):
    raise FileNotFoundError(f"split_dir does not exist: {args.split_dir}")

expected_split0 = os.path.join(args.split_dir, "splits_0.csv")
print("expect first split file:", expected_split0)
if not os.path.isfile(expected_split0):
    raise FileNotFoundError(f"Cannot find split file: {expected_split0}")

settings.update({"split_dir": args.split_dir})

with open(args.results_dir + "/experiment.txt", "w") as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print("{}: {}".format(key, val))

if __name__ == "__main__":
    t0 = timer()
    main(args, dataset, slide_id_map)
    t1 = timer()
    print("finished!")
    print("end script")
    print("Script Time: %f seconds" % (t1 - t0))
