import os

# ------------------ limit CPU threads ------------------
cpu_num = 1
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)

Z_95 = 1.96  # 95% CI factor (approx)
import argparse
import numpy as np
import h5py

# ------------------ utils ------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def parse_comma_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]

def load_done_set(done_file: str):
    if not os.path.exists(done_file):
        return set()
    with open(done_file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def append_done(done_file: str, item: str):
    with open(done_file, "a", encoding="utf-8") as f:
        f.write(item + "\n")

# ------------------ core math ------------------
def load_goldenbank(goldenbank_path: str):
    """
    goldenbank/<encode>/<dataset>_goldenbank.h5 must contain:
      - cluster_embedding: (K,D)
      - dist: (K,D) or (D,)
    Return: centroids (K,D), dist, D
    """
    with h5py.File(goldenbank_path, "r") as f:
        if "cluster_embedding" not in f or "dist" not in f:
            raise KeyError(f"{goldenbank_path} must contain 'cluster_embedding' and 'dist'")
        centroids = f["cluster_embedding"][:]  # (K,D)
        dist = f["dist"][:]                   # (K,D) or (D,)

    centroids = np.asarray(centroids, dtype=np.float32)
    dist = np.asarray(dist, dtype=np.float32)

    if centroids.ndim != 2:
        raise ValueError(f"cluster_embedding must be 2D (K,D), got {centroids.shape}")

    K, D = centroids.shape
    if K <= 0 or D <= 0:
        raise ValueError(f"Invalid cluster_embedding shape: {centroids.shape}")

    if dist.ndim == 1:
        if dist.shape[0] != D:
            raise ValueError(f"dist must be (D,) if 1D. Expected {D}, got {dist.shape}")
    elif dist.ndim == 2:
        if dist.shape != centroids.shape:
            raise ValueError(
                f"dist must match cluster_embedding when 2D. Expected {centroids.shape}, got {dist.shape}"
            )
    else:
        raise ValueError(f"dist must be 1D or 2D, got ndim={dist.ndim}")

    return centroids, dist, D

def nearest_centroid_indices(x: np.ndarray, centroids: np.ndarray, chunk_size: int = 2048):
    """
    x: (N,D), centroids: (K,D)
    return idx: (N,) nearest centroid index by squared L2 distance
    """
    x = np.asarray(x, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)

    if x.ndim != 2 or centroids.ndim != 2 or x.shape[1] != centroids.shape[1]:
        raise ValueError(f"Dim mismatch: x {x.shape}, centroids {centroids.shape}")

    K = centroids.shape[0]
    c_norm = np.sum(centroids * centroids, axis=1)  # (K,)
    idx_all = np.empty((x.shape[0],), dtype=np.int64)

    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        xb = x[start:end]  # (B,D)

        x_norm = np.sum(xb * xb, axis=1, keepdims=True)  # (B,1)
        dots = xb @ centroids.T  # (B,K)
        d2 = x_norm + c_norm.reshape(1, K) - 2.0 * dots  # (B,K)

        idx_all[start:end] = np.argmin(d2, axis=1)

    return idx_all

def clip_to_ci(features: np.ndarray, centroids: np.ndarray, dist: np.ndarray, idx: np.ndarray):
    """
    Clip each feature vector to its nearest centroid's 95% CI:
      lower = mean - 1.96*std
      upper = mean + 1.96*std
    features: (N,D)
    centroids: (K,D)
    dist: (K,D) or (D,)
    idx: (N,)
    """
    features = np.asarray(features, dtype=np.float32)
    means = centroids[idx]  # (N,D)
    D = centroids.shape[1]

    if dist.ndim == 1:
        std = dist.reshape(1, D)  # (1,D), broadcast
    else:
        std = dist[idx]  # (N,D)

    lower = means - Z_95 * std
    upper = means + Z_95 * std
    return np.clip(features, lower, upper)

# ------------------ H5 IO ------------------
def copy_h5_with_replaced_features(src_h5: str, dst_h5: str, new_features: np.ndarray):
    """
    Copy all content from src_h5 to dst_h5, but replace dataset 'features' with new_features.
    Preserves groups/datasets/attrs. Uses gzip compression for datasets.
    """
    def _copy_item(src_group, dst_group, name):
        obj = src_group[name]
        if isinstance(obj, h5py.Dataset):
            if name == "features":
                if "features" in dst_group:
                    del dst_group["features"]
                dst_group.create_dataset("features", data=new_features, compression="gzip")
            else:
                dst_group.create_dataset(name, data=obj[...], compression="gzip")
        elif isinstance(obj, h5py.Group):
            grp = dst_group.require_group(name)
            for k in obj.keys():
                _copy_item(obj, grp, k)
        else:
            raise TypeError(f"Unsupported HDF5 object: {type(obj)}")

    ensure_dir(os.path.dirname(dst_h5))

    with h5py.File(src_h5, "r") as fsrc, h5py.File(dst_h5, "w") as fdst:
        # copy root attrs
        for k, v in fsrc.attrs.items():
            fdst.attrs[k] = v

        # copy all content, replacing features if present
        for k in fsrc.keys():
            _copy_item(fsrc, fdst, k)

        # if src has no features, create it
        if "features" not in fsrc:
            fdst.create_dataset("features", data=new_features, compression="gzip")

# ------------------ pipeline ------------------
def process_dataset_encode(
    dataset: str,
    encode: str,
    embedding_root: str,
    goldenbank_root: str,
    output_root: str,
    chunk_size: int = 2048,
    resume: bool = True,
    overwrite: bool = False,
    progress_root: str | None = None
):
    """
    Input:
      embedding_root/<dataset>/<encode>/*.h5
        each has dataset "features": (N,D)
    Goldenbank:
      goldenbank_root/<encode>/<dataset>_goldenbank.h5
        has cluster_embedding (K,D), dist ((K,D) or (D,))
    Output:
      output_root/<dataset>/<encode>/*.h5
        same as input but features clipped to 95% CI of nearest centroid
    Resume:
      records completed slide filenames in progress log.
    """
    in_dir = os.path.join(embedding_root, dataset, encode)
    if not os.path.isdir(in_dir):
        print(f"[WARN] Input dir not found, skip: {in_dir}")
        return

    goldenbank_path = os.path.join(goldenbank_root, encode, f"{dataset}_goldenbank.h5")
    if not os.path.exists(goldenbank_path):
        print(f"[WARN] Goldenbank not found, skip: {goldenbank_path}")
        return

    centroids, dist, D = load_goldenbank(goldenbank_path)

    out_dir = os.path.join(output_root, dataset, encode)
    ensure_dir(out_dir)

    h5_files = sorted([fn for fn in os.listdir(in_dir) if fn.endswith(".h5")])
    if not h5_files:
        print(f"[INFO] No .h5 files in {in_dir}")
        return

    done_set = set()
    done_file = None
    if resume:
        if progress_root is None:
            progress_root = os.path.join(output_root, "_progress")
        ensure_dir(progress_root)
        done_file = os.path.join(progress_root, f"{dataset}__{encode}.done.txt")
        done_set = load_done_set(done_file)

    for fn in h5_files:
        src_path = os.path.join(in_dir, fn)
        dst_path = os.path.join(out_dir, fn)

        # resume skip
        if resume and fn in done_set:
            continue

        # if not overwrite, skip existing output (and mark done to speed later)
        if (not overwrite) and os.path.exists(dst_path):
            if resume and done_file is not None:
                append_done(done_file, fn)
                done_set.add(fn)
            continue

        try:
            with h5py.File(src_path, "r") as f:
                if "features" not in f:
                    print(f"[WARN] skip (no 'features'): {src_path}")
                    if resume and done_file is not None:
                        append_done(done_file, fn)
                        done_set.add(fn)
                    continue
                features = f["features"][:]

            features = np.asarray(features, dtype=np.float32)

            # allow 1D (D,) -> (1,D)
            if features.ndim == 1:
                features = features.reshape(1, -1)

            if features.ndim != 2:
                print(f"[WARN] skip (features not 2D): {src_path}, got {features.shape}")
                continue

            if features.shape[1] != D:
                print(f"[WARN] skip (dim mismatch): {src_path}, features D={features.shape[1]} != goldenbank D={D}")
                continue

            idx = nearest_centroid_indices(features, centroids, chunk_size=chunk_size)
            corrected = clip_to_ci(features, centroids, dist, idx)

            copy_h5_with_replaced_features(src_path, dst_path, corrected)

            if resume and done_file is not None:
                append_done(done_file, fn)
                done_set.add(fn)

            print(f"[OK] ds={dataset} en={encode} slide={fn}  N={corrected.shape[0]} D={D}")

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Interrupted by user. Progress saved. Rerun with --resume to continue.")
            raise
        except Exception as e:
            print(f"[ERROR] failed on {src_path}: {e}")
            # don't mark done, so it can retry next run
            continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BRACS,TCGA-BRCA",
                    help="Comma-separated dataset names, e.g. BCNB,BRACS")
    ap.add_argument("--encode", default="CONCH",
                    help="Comma-separated encode names, e.g. UNI,CONCH")
    ap.add_argument("--embedding_root", default="/home/baizhiwang/prov-data/embedding",
                    help="root of embedding folder")
    ap.add_argument("--goldenbank_root", default="/home/baizhiwang/Badge4/Uncertainty/ablation/goldenbank/ratio=0.5",
                    help="root of goldenbank folder")
    ap.add_argument("--output_root", default="/home/baizhiwang/Badge4/Uncertainty/ablation/dataset/ratio=0.5",
                    help="output root folder")
    ap.add_argument("--chunk_size", type=int, default=2048,
                    help="chunk size for nearest centroid search")

    # resume / overwrite
    ap.add_argument("--resume", action="store_true",
                    help="Resume from progress log (recommended)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output h5 files")
    ap.add_argument("--progress_root", default="",
                    help="Where to store progress logs (default: output_root/_progress)")

    args = ap.parse_args()

    datasets = parse_comma_list(args.dataset)
    encodes = parse_comma_list(args.encode)

    progress_root = args.progress_root.strip() if args.progress_root.strip() else None

    for ds in datasets:
        for en in encodes:
            print(f"\n=== Processing ds={ds}  en={en} ===")
            process_dataset_encode(
                dataset=ds,
                encode=en,
                embedding_root=args.embedding_root,
                goldenbank_root=args.goldenbank_root,
                output_root=args.output_root,
                chunk_size=args.chunk_size,
                resume=args.resume,
                overwrite=args.overwrite,
                progress_root=progress_root
            )

if __name__ == "__main__":
    main()
