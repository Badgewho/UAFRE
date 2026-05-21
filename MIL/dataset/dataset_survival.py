from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re
from glob import glob

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path='dataset_csv/ccrcc_clean.csv',
        mode='omic',
        apply_sig=False,
        shuffle=False,
        seed=7,
        print_info=True,
        n_bins=4,
        ignore=[],
        patient_strat=False,
        label_col=None,
        filter_dict={},
        eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None

        slide_data = pd.read_csv(csv_path, low_memory=False)

        if shuffle:
            np.random.seed(seed)
            slide_data = slide_data.sample(frac=1, random_state=seed).reset_index(drop=True)

        if 'case_id' not in slide_data.columns:
            slide_data.index = slide_data.index.astype(str).str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns, f"label_col={label_col} not in columns={slide_data.columns.tolist()}"
        self.label_col = label_col

        slide_data[self.label_col] = pd.to_numeric(slide_data[self.label_col], errors='coerce')
        if 'censorship' not in slide_data.columns:
            raise KeyError(f"'censorship' column not found in csv: {csv_path}")
        slide_data['censorship'] = pd.to_numeric(slide_data['censorship'], errors='coerce')

        before = len(slide_data)
        slide_data = slide_data.dropna(subset=[self.label_col, 'censorship']).copy()
        slide_data['censorship'] = slide_data['censorship'].astype(int)
        after = len(slide_data)
        print(f"[dataset] cleaned rows: {before} -> {after}")

        if 'slide_id' not in slide_data.columns:
            slide_data['slide_id'] = slide_data['case_id'].astype(str)

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1].copy()

        if len(uncensored_df) == 0:
            raise ValueError("No uncensored samples found. Cannot build survival bins.")

        disc_labels, q_bins = pd.qcut(
            uncensored_df[self.label_col],
            q=n_bins,
            retbins=True,
            labels=False,
            duplicates='drop'
        )

        q_bins[-1] = slide_data[self.label_col].max() + eps
        q_bins[0] = slide_data[self.label_col].min() - eps

        disc_labels, q_bins = pd.cut(
            patients_df[self.label_col],
            bins=q_bins,
            retbins=True,
            labels=False,
            right=False,
            include_lowest=True
        )

        disc_labels = pd.Series(disc_labels, index=patients_df.index)
        if disc_labels.isna().any():
            bad_rows = patients_df.loc[disc_labels.isna()]
            print("[dataset] bad rows causing NaN disc_labels:")
            print(bad_rows.head(20))
            raise ValueError("disc_labels contains NaN after discretization.")

        patients_df.insert(2, 'label', disc_labels.astype(int).values)

        patient_dict = {}
        slide_data_indexed = slide_data.set_index('case_id')
        representative_slide_ids = []

        for patient in patients_df['case_id']:
            slide_ids = slide_data_indexed.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array([slide_ids])
                representative_slide_ids.append(str(slide_ids[0]))
            elif isinstance(slide_ids, pd.Series):
                slide_ids = slide_ids.astype(str).values
                representative_slide_ids.append(str(slide_ids[0]))
            else:
                slide_ids = np.array(slide_ids).reshape(-1).astype(str)
                representative_slide_ids.append(str(slide_ids[0]))

            patient_dict.update({patient: slide_ids})

        self.patient_dict = patient_dict

        slide_data = patients_df.copy()
        slide_data.reset_index(drop=True, inplace=True)
        slide_data['slide_id'] = representative_slide_ids

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.label_dict = label_dict

        for i in slide_data.index:
            disc_key = int(slide_data.loc[i, 'label'])
            slide_data.at[i, 'disc_label'] = disc_key
            censorship = int(slide_data.loc[i, 'censorship'])
            key = (disc_key, censorship)
            if key not in label_dict:
                raise KeyError(f"Unexpected key={key}, label_dict keys={list(label_dict.keys())}")
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes = len(self.label_dict)

        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {
            'case_id': patients_df['case_id'].values,
            'label': patients_df['label'].values
        }

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.mode = mode
        self.cls_ids_prep()

        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./dataset_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]]
            patient_labels.append(label)

        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def get_split_from_df(self, backbone, patch_size, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].astype(str).isin(split.astype(str).tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(
                df_slice,
                metadata=self.metadata,
                mode=self.mode,
                signatures=self.signatures,
                data_dir=self.data_dir,
                label_col=self.label_col,
                patient_dict=self.patient_dict,
                num_classes=self.num_classes
            )
            split.set_backbone(backbone)
            split.set_patch_size(patch_size)
        else:
            split = None

        return split

    def return_splits(self, backbone, patch_size='', from_id: bool=True, csv_path: str=None):
        print("[return_splits] from_id =", from_id)
        print("[return_splits] csv_path =", csv_path)

        train_split, val_split, test_split = None, None, None

        if csv_path is not None:
            all_splits = pd.read_csv(csv_path)
            print("[return_splits] split columns:", all_splits.columns.tolist())
            print(all_splits.head())

            if all(c in all_splits.columns for c in ["train", "val", "test"]):
                train_ids = [x for x in all_splits["train"].dropna().astype(str).tolist() if x.strip() != ""]
                val_ids = [x for x in all_splits["val"].dropna().astype(str).tolist() if x.strip() != ""]
                test_ids = [x for x in all_splits["test"].dropna().astype(str).tolist() if x.strip() != ""]

            elif "slide_id" in all_splits.columns and "split" in all_splits.columns:
                train_ids = all_splits.loc[all_splits["split"] == "train", "slide_id"].dropna().astype(str).tolist()
                val_ids = all_splits.loc[all_splits["split"] == "val", "slide_id"].dropna().astype(str).tolist()
                test_ids = all_splits.loc[all_splits["split"] == "test", "slide_id"].dropna().astype(str).tolist()

            elif "case_id" in all_splits.columns and "split" in all_splits.columns:
                train_ids = all_splits.loc[all_splits["split"] == "train", "case_id"].dropna().astype(str).tolist()
                val_ids = all_splits.loc[all_splits["split"] == "val", "case_id"].dropna().astype(str).tolist()
                test_ids = all_splits.loc[all_splits["split"] == "test", "case_id"].dropna().astype(str).tolist()

            else:
                raise ValueError(f"Unrecognized split csv format: columns={all_splits.columns.tolist()}")

            print("[return_splits] parsed ids lens:", len(train_ids), len(val_ids), len(test_ids))

            self.train_ids = train_ids
            self.val_ids = val_ids
            self.test_ids = test_ids

            if len(train_ids) > 0:
                mask = self.slide_data["slide_id"].astype(str).isin(train_ids)
                train_data = self.slide_data[mask].reset_index(drop=True)
                train_split = Generic_Split(
                    train_data,
                    metadata=self.metadata,
                    mode=self.mode,
                    signatures=self.signatures,
                    data_dir=self.data_dir,
                    label_col=self.label_col,
                    patient_dict=self.patient_dict,
                    num_classes=self.num_classes
                )
                train_split.set_backbone(backbone)
                train_split.set_patch_size(patch_size)

            if len(val_ids) > 0:
                mask = self.slide_data["slide_id"].astype(str).isin(val_ids)
                val_data = self.slide_data[mask].reset_index(drop=True)
                val_split = Generic_Split(
                    val_data,
                    metadata=self.metadata,
                    mode=self.mode,
                    signatures=self.signatures,
                    data_dir=self.data_dir,
                    label_col=self.label_col,
                    patient_dict=self.patient_dict,
                    num_classes=self.num_classes
                )
                val_split.set_backbone(backbone)
                val_split.set_patch_size(patch_size)

            if len(test_ids) > 0:
                mask = self.slide_data["slide_id"].astype(str).isin(test_ids)
                test_data = self.slide_data[mask].reset_index(drop=True)
                test_split = Generic_Split(
                    test_data,
                    metadata=self.metadata,
                    mode=self.mode,
                    signatures=self.signatures,
                    data_dir=self.data_dir,
                    label_col=self.label_col,
                    patient_dict=self.patient_dict,
                    num_classes=self.num_classes
                )
                test_split.set_backbone(backbone)
                test_split.set_patch_size(patch_size)

            print("[return_splits] actual matched split lens:",
                  0 if train_split is None else len(train_split),
                  0 if val_split is None else len(val_split),
                  0 if test_split is None else len(test_split))

            return train_split, val_split, test_split

        self.train_ids = [] if self.train_ids is None else self.train_ids
        self.val_ids = [] if self.val_ids is None else self.val_ids
        self.test_ids = [] if self.test_ids is None else self.test_ids

        print("[return_splits] fallback to internal ids:",
              len(self.train_ids), len(self.val_ids), len(self.test_ids))

        if len(self.train_ids) > 0:
            train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
            train_split = Generic_Split(
                train_data,
                metadata=self.metadata,
                mode=self.mode,
                signatures=self.signatures,
                data_dir=self.data_dir,
                label_col=self.label_col,
                patient_dict=self.patient_dict,
                num_classes=self.num_classes
            )
            train_split.set_backbone(backbone)
            train_split.set_patch_size(patch_size)

        if len(self.val_ids) > 0:
            val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
            val_split = Generic_Split(
                val_data,
                metadata=self.metadata,
                mode=self.mode,
                signatures=self.signatures,
                data_dir=self.data_dir,
                label_col=self.label_col,
                patient_dict=self.patient_dict,
                num_classes=self.num_classes
            )
            val_split.set_backbone(backbone)
            val_split.set_patch_size(patch_size)

        if len(self.test_ids) > 0:
            test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
            test_split = Generic_Split(
                test_data,
                metadata=self.metadata,
                mode=self.mode,
                signatures=self.signatures,
                data_dir=self.data_dir,
                label_col=self.label_col,
                patient_dict=self.patient_dict,
                num_classes=self.num_classes
            )
            test_split.set_backbone(backbone)
            test_split.set_patch_size(patch_size)

        return train_split, val_split, test_split

    def create_splits(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0, custom_test_ids=None):
        settings = {
            'n_splits': k,
            'val_num': val_num,
            'test_num': test_num,
            'label_frac': label_frac,
            'seed': self.seed,
            'custom_test_ids': custom_test_ids
        }

        if self.patient_strat:
            settings.update({'cls_ids': self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids': self.slide_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)

    def set_splits(self, start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)
        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))]

            for split in range(len(ids)):
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def test_split_gen(self, return_descriptor=False):
        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index=index, columns=columns)
            df = df.reset_index(drop=True)

        count = len(self.train_ids)
        print('\nnumber of training samples: {}'.format(count))
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]

        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1)
        df.to_csv(filename, index=False)


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False

        self._h5_name_cache = None
        self._h5_resolved_cache = {}
        self._last_global_h5_path = None

        if self.data_dir is not None and self.mode == 'path' and not isinstance(self.data_dir, dict):
            self._normalize_patient_dict_slide_ids()

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def _build_h5_name_cache(self, data_dir):
        if data_dir is None:
            self._h5_name_cache = []
            return

        if self._h5_name_cache is not None:
            return

        h5_files = sorted(glob(os.path.join(data_dir, "*.h5")))
        self._h5_name_cache = [os.path.splitext(os.path.basename(p))[0] for p in h5_files]
        print(f"[H5 cache] built from {data_dir}, num_h5={len(self._h5_name_cache)}")

    def _resolve_h5_path(self, data_dir, slide_id: str):
        sid = str(slide_id).rstrip('.svs')
        cache_key = (data_dir, sid)

        if cache_key in self._h5_resolved_cache:
            return self._h5_resolved_cache[cache_key]

        self._build_h5_name_cache(data_dir)

        exact_path = os.path.join(data_dir, f"{sid}.h5")
        if os.path.exists(exact_path):
            self._h5_resolved_cache[cache_key] = exact_path
            self._last_global_h5_path = exact_path
            return exact_path

        names = self._h5_name_cache if self._h5_name_cache is not None else []

        prefix_hits = [name for name in names if name.startswith(sid)]
        if len(prefix_hits) == 1:
            path = os.path.join(data_dir, prefix_hits[0] + ".h5")
            self._h5_resolved_cache[cache_key] = path
            self._last_global_h5_path = path
            return path
        elif len(prefix_hits) > 1:
            prefix_hits = sorted(prefix_hits, key=len)
            path = os.path.join(data_dir, prefix_hits[0] + ".h5")
            print(f"[Warn] multiple prefix matches for '{sid}', choose '{prefix_hits[0]}'")
            self._h5_resolved_cache[cache_key] = path
            self._last_global_h5_path = path
            return path

        contains_hits = [name for name in names if sid in name]
        if len(contains_hits) == 1:
            path = os.path.join(data_dir, contains_hits[0] + ".h5")
            self._h5_resolved_cache[cache_key] = path
            self._last_global_h5_path = path
            return path
        elif len(contains_hits) > 1:
            contains_hits = sorted(contains_hits, key=len)
            path = os.path.join(data_dir, contains_hits[0] + ".h5")
            print(f"[Warn] multiple contains matches for '{sid}', choose '{contains_hits[0]}'")
            self._h5_resolved_cache[cache_key] = path
            self._last_global_h5_path = path
            return path

        raise FileNotFoundError(
            f"Cannot resolve h5 for slide_id='{sid}' under data_dir='{data_dir}'"
        )

    def _resolve_h5_path_with_fallback(self, data_dir, slide_id: str, previous_h5_path: str = None):
        try:
            path = self._resolve_h5_path(data_dir, slide_id)
            return path
        except FileNotFoundError:
            if previous_h5_path is not None:
                print(f"[Warn] unresolved slide_id='{slide_id}', fallback to previous h5 '{os.path.basename(previous_h5_path)}'")
                return previous_h5_path
            if self._last_global_h5_path is not None:
                print(f"[Warn] unresolved slide_id='{slide_id}', fallback to global previous h5 '{os.path.basename(self._last_global_h5_path)}'")
                return self._last_global_h5_path
            raise

    def _normalize_patient_dict_slide_ids(self):
        self._build_h5_name_cache(self.data_dir)
        new_patient_dict = {}

        for case_id, slide_ids in self.patient_dict.items():
            resolved = []
            last_local_path = None
            for sid in np.array(slide_ids).reshape(-1):
                try:
                    path = self._resolve_h5_path(self.data_dir, sid)
                    resolved_name = os.path.splitext(os.path.basename(path))[0]
                    resolved.append(resolved_name)
                    last_local_path = path
                except FileNotFoundError:
                    if last_local_path is not None:
                        resolved_name = os.path.splitext(os.path.basename(last_local_path))[0]
                        print(f"[Warn] patient_dict unresolved slide_id='{sid}' for case_id='{case_id}', fallback to previous '{resolved_name}'")
                        resolved.append(resolved_name)
                    elif self._last_global_h5_path is not None:
                        resolved_name = os.path.splitext(os.path.basename(self._last_global_h5_path))[0]
                        print(f"[Warn] patient_dict unresolved slide_id='{sid}' for case_id='{case_id}', fallback to global previous '{resolved_name}'")
                        resolved.append(resolved_name)
                    else:
                        print(f"[Warn] patient_dict unresolved slide_id='{sid}' for case_id='{case_id}', keep original")
                        resolved.append(str(sid))
            new_patient_dict[case_id] = np.array(resolved)

        self.patient_dict = new_patient_dict
        print("[patient_dict] normalized with real h5 basenames")

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    previous_h5_path = None

                    for slide_id in slide_ids:
                        wsi_path = self._resolve_h5_path_with_fallback(data_dir, slide_id, previous_h5_path)

                        with h5py.File(wsi_path, 'r') as f:
                            features = f['features'][:]
                            wsi_bag = torch.tensor(features, dtype=torch.float32)

                        path_features.append(wsi_bag)
                        previous_h5_path = wsi_path
                        self._last_global_h5_path = wsi_path

                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, torch.zeros((1, 1)), label, event_time, c)

                elif self.mode == 'cluster':
                    path_features = []
                    cluster_ids = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(str(slide_id).rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                        cluster_ids.extend(self.fname2ids[str(slide_id)[:-4] + '.pt'])
                    path_features = torch.cat(path_features, dim=0)
                    cluster_ids = torch.Tensor(cluster_ids)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, cluster_ids, genomic_features, label, event_time, c)

                elif self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1, 1)), genomic_features, label, event_time, c)

                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(str(slide_id).rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features, label, event_time, c)

                elif self.mode == 'coattn':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(str(slide_id).rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c)

                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.signatures = signatures

        self._h5_name_cache = None
        self._h5_resolved_cache = {}
        self._last_global_h5_path = None

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)

    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)

    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed

    def set_backbone(self, backbone):
        print('Setting Backbone:', backbone)
        self.backbone = backbone

    def set_patch_size(self, size):
        print('Setting Patchsize:', size)
        self.patch_size = size

    def pre_loading(self, thread=8):
        self.cache_flag = True
        ids = list(range(len(self)))
        from multiprocessing.pool import ThreadPool
        exe = ThreadPool(thread)
        exe.map(self.__getitem__, ids)
