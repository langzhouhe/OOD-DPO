import logging
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data_loader import EnergyDPODataLoader
from utils import process_drugood_data, process_good_data

logger = logging.getLogger(__name__)


class SupervisedFeatureDataset(Dataset):
    """Simple dataset of precomputed features and labels."""

    def __init__(self, features, labels):
        self.features = features if isinstance(features, torch.Tensor) else torch.stack(features)
        # labels may be scalars or lists; turn into tensor but keep shape
        if isinstance(labels, torch.Tensor):
            lbl = labels
        else:
            try:
                lbl = torch.tensor(labels)
            except Exception:
                lbl = torch.tensor(labels, dtype=torch.float32)

        # Build masks for multi-task to ignore NaNs; also replace NaNs in labels with 0
        self.masks = None
        if lbl.dtype.is_floating_point and lbl.dim() > 1:
            masks = ~torch.isnan(lbl)
            lbl = torch.where(masks, lbl, torch.zeros_like(lbl))
            self.masks = masks

        self.labels = lbl
        assert len(self.features) == len(self.labels)
        logger.info(
            f"SupervisedFeatureDataset: {len(self.features)} samples, feature dim: {self.features.shape[1] if self.features.dim()>1 else 'unknown'}"
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {"features": self.features[idx], "labels": self.labels[idx]}
        if self.masks is not None:
            item["masks"] = self.masks[idx]
        return item


def _to_jsonable(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        return v.item() if v.numel() == 1 else v.view(-1).tolist()
    if isinstance(v, np.ndarray):
        return v.item() if v.size == 1 else v.reshape(-1).tolist()
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()
                out.append(x.item() if x.numel() == 1 else x.view(-1).tolist())
            elif isinstance(x, np.ndarray):
                out.append(x.item() if x.size == 1 else x.reshape(-1).tolist())
            else:
                out.append(x)
        return out
    return v


class SupervisedBaselineDataLoader(EnergyDPODataLoader):
    """
    Extends EnergyDPODataLoader: reuses splits/features and adds real labels for supervised training.
    """

    def __init__(self, args):
        logger.info("=== Initializing SupervisedBaselineDataLoader ===")
        logger.info("Inheriting from EnergyDPODataLoader for consistent data splits...")
        super().__init__(args)

        self._raw_labels = {}
        self._raw_smiles_with_labels = {}
        self.final_labels = {}

        logger.info("Loading classification labels for supervised training...")
        self._load_labels_from_cache_or_recompute()

        logger.info("SupervisedBaselineDataLoader initialization complete.")
        logger.info(f"Available label splits: {list(self.final_labels.keys())}")

    # ----- Hook to also load labels when parent loads raw data -----
    def _load_raw_data(self):
        super()._load_raw_data()
        self._load_raw_labels()

    # ----- Raw labels extraction -----
    def _load_raw_labels(self):
        dataset_lower = self.dataset_name.lower()

        if "drugood" in dataset_lower or "lbap" in dataset_lower:
            # Load DrugOOD JSON and extract cls_label
            if not self.data_file:
                self.data_file = f"{self.data_path}/{self.drugood_subset}.json"
            with open(self.data_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            if "split" not in raw_data:
                raise ValueError("Invalid DrugOOD data format: missing 'split'")
            split_data = raw_data["split"]

            def extract_labels_and_smiles(items):
                smiles, labels = [], []
                for it in items:
                    s = it.get("smiles")
                    if s is None:
                        continue
                    smiles.append(s)
                    if "cls_label" in it:
                        labels.append(it["cls_label"])
                return smiles, labels

            train_smiles, train_labels = extract_labels_and_smiles(split_data.get("train", []))
            if split_data.get("iid_val"):
                val_smiles, val_labels = extract_labels_and_smiles(split_data.get("iid_val", []))
            else:
                # split from train
                val_size = min(len(train_smiles) // 4, 1000) if len(train_smiles) >= 100 else len(train_smiles) // 2
                val_smiles, val_labels = train_smiles[:val_size], train_labels[:val_size]
                train_smiles, train_labels = train_smiles[val_size:], train_labels[val_size:]
            if split_data.get("iid_test"):
                test_smiles, test_labels = extract_labels_and_smiles(split_data.get("iid_test", []))
            else:
                test_smiles, test_labels = val_smiles, val_labels

            if not train_labels:
                raise ValueError("No cls_label found in DrugOOD training data.")

            self._raw_labels = {"train_id": train_labels, "val_id": val_labels, "test_id": test_labels}
            self._raw_smiles_with_labels = {"train_id": train_smiles, "val_id": val_smiles, "test_id": test_smiles}

            try:
                unique_labels = set(train_labels)
                logger.info(f"Found cls_label in DrugOOD data: {len(unique_labels)} classes")
            except TypeError:
                logger.info("DrugOOD labels include non-hashables; skipping unique count")

        elif "good" in dataset_lower:
            # Use processed GOOD data (utils.process_good_data already returns labels)
            data_dict = getattr(self, "_good_data_dict", None)
            if not data_dict:
                data_dict = process_good_data(
                    dataset_name=self.dataset_name,
                    domain=self.good_domain,
                    shift=self.good_shift,
                    data_path=self.data_path,
                    max_samples=self.max_samples,
                    seed=self.data_seed,
                    validate_smiles_flag=False,
                )
                self._good_data_dict = data_dict

            train_smiles = data_dict.get("train_id_smiles", [])
            val_smiles = data_dict.get("val_id_smiles", [])
            test_smiles = data_dict.get("test_id_smiles", val_smiles)

            train_labels = data_dict.get("train_id_labels", [])
            val_labels = data_dict.get("val_id_labels", [])
            test_labels = data_dict.get("test_id_labels", val_labels)

            if not train_labels:
                raise ValueError("No labels found for GOOD training data.")

            # Normalize to JSON-safe labels
            def ser(l):
                return _to_jsonable(l)

            train_labels = [ser(l) for l in train_labels]
            val_labels = [ser(l) for l in val_labels] if val_labels else []
            test_labels = [ser(l) for l in test_labels] if test_labels else []

            # Single-task vs multitask detection
            is_multitask = any(isinstance(l, (list, tuple)) for l in train_labels)
            if not is_multitask:
                # Unique values for classification/regression detection
                try:
                    unique_values = set(train_labels)
                except TypeError:
                    unique_values = set([tuple(l) if isinstance(l, (list, tuple)) else l for l in train_labels])
                import numpy as _np
                is_reg = len(unique_values) > 20 or any(
                    isinstance(v, (float, _np.floating)) and not float(v).is_integer() for v in unique_values
                )
                if is_reg:
                    logger.info("GOOD dataset appears regression; converting to binary via median split")
                    med = sorted(train_labels)[len(train_labels) // 2]
                    def binarize(seq):
                        return [1 if float(x) >= float(med) else 0 for x in seq]
                    train_labels = binarize(train_labels)
                    val_labels = binarize(val_labels) if val_labels else []
                    test_labels = binarize(test_labels) if test_labels else []
                    logger.info(
                        f"Converted to binary classification: {sum(train_labels)}/{len(train_labels)} positive"
                    )
            else:
                logger.info(
                    f"Detected multi-task GOOD labels: samples={len(train_labels)}, tasks={len(train_labels[0]) if train_labels and isinstance(train_labels[0], (list, tuple)) else 'unknown'}"
                )

            self._raw_labels = {"train_id": train_labels, "val_id": val_labels, "test_id": test_labels}
            self._raw_smiles_with_labels = {"train_id": train_smiles, "val_id": val_smiles, "test_id": test_smiles}

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        logger.info(
            "Raw labels loaded. Found %d train_id, %d val_id, %d test_id labels.",
            len(self._raw_labels["train_id"]),
            len(self._raw_labels["val_id"]),
            len(self._raw_labels["test_id"]),
        )

    # ----- Cache orchestration -----
    def _load_labels_from_cache_or_recompute(self):
        if not hasattr(self, "final_smiles") or not self.final_smiles:
            raise RuntimeError("SMILES data not loaded yet.")

        labels_cache = self._get_labels_cache_path()
        if self._load_labels_cache(labels_cache):
            return

        if not self._raw_labels:
            logger.info("Labels cache miss - loading raw labels and selecting corresponding labels...")
            self._load_raw_labels()

        self._select_final_labels()
        self._save_labels_cache(labels_cache)

    def _get_labels_cache_path(self):
        if "drugood" in self.dataset_name.lower() or "lbap" in self.dataset_name.lower():
            base = self.drugood_subset if self.drugood_subset else self.dataset_name
        else:
            base = self.dataset_name
        if not base:
            base = "unknown_dataset"
        base = base.replace("/", "_").replace("-", "_")
        if "good" in base:
            filename = f"{base}_{self.good_domain}_{self.good_shift}_seed{self.data_seed}"
        else:
            filename = f"{base}_seed{self.data_seed}"
        if self.max_samples:
            filename += f"_debug{self.max_samples}"
        filename += "_labels.json"
        return self.cache_dir / filename

    def _load_labels_cache(self, cache_file: Path):
        if not cache_file.exists() or self.force_recompute:
            return False
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cache_meta = data.get("metadata", {})
            current_meta = self._get_splits_cache_metadata()
            for k, v in current_meta.items():
                if cache_meta.get(k) != v:
                    logger.info(
                        f"Labels cache parameter mismatch: {k} = {cache_meta.get(k)} != {v}"
                    )
                    return False
            self.final_labels = data["labels"]
            total = sum(len(lst) for lst in self.final_labels.values())
            logger.info(f"Loaded labels cache with {total} total labels from: {cache_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load labels cache {cache_file}: {e}")
            return False

    def _save_labels_cache(self, cache_file: Path):
        labels_jsonable = {k: [_to_jsonable(l) for l in v] for k, v in self.final_labels.items()}
        payload = {
            "metadata": self._get_splits_cache_metadata(),
            "labels": labels_jsonable,
            "timestamp": self._get_current_timestamp(),
        }
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            kb = cache_file.stat().st_size / 1024
            total = sum(len(lst) for lst in self.final_labels.values())
            logger.info(f"Labels cache saved: {cache_file} ({kb:.1f} KB, {total} labels)")
        except Exception as e:
            logger.error(f"Failed to save labels cache {cache_file}: {e}")

    def _get_current_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()

    # ----- Final label selection aligning to final_smiles -----
    def _select_final_labels(self):
        global_map = {}
        for split_key in ["train_id", "val_id", "test_id"]:
            smiles_list = self._raw_smiles_with_labels.get(split_key, [])
            labels_list = self._raw_labels.get(split_key, [])
            if len(smiles_list) == len(labels_list):
                for s, l in zip(smiles_list, labels_list):
                    global_map[s] = l
                if "good" in self.dataset_name.lower():
                    try:
                        from rdkit import Chem
                        for s, l in zip(smiles_list, labels_list):
                            mol = Chem.MolFromSmiles(s)
                            if mol is not None:
                                can = Chem.MolToSmiles(mol, canonical=True)
                                if can != s:
                                    global_map[can] = l
                    except Exception:
                        pass

        for split in ["train_id", "val_id", "test_id"]:
            final_smiles = self.final_smiles.get(split, [])
            if not final_smiles:
                self.final_labels[split] = []
                continue

            raw_smiles = self._raw_smiles_with_labels.get(split, [])
            raw_labels = self._raw_labels.get(split, [])
            if len(raw_smiles) != len(raw_labels):
                raise ValueError(f"Internal error: SMILES and labels count mismatch in {split}")
            smile2label = dict(zip(raw_smiles, raw_labels))

            # Also add canonical mapping per split for GOOD
            if "good" in self.dataset_name.lower():
                try:
                    from rdkit import Chem
                    can_map = {}
                    for s, l in zip(raw_smiles, raw_labels):
                        mol = Chem.MolFromSmiles(s)
                        if mol is not None:
                            can = Chem.MolToSmiles(mol, canonical=True)
                            if can != s:
                                can_map[can] = l
                    smile2label.update(can_map)
                except Exception:
                    pass

            selected = []
            fallback = 0
            missing = 0
            for s in final_smiles:
                if s in smile2label:
                    val = smile2label[s]
                elif s in global_map:
                    val = global_map[s]
                    fallback += 1
                else:
                    if "good" in self.dataset_name.lower():
                        try:
                            from rdkit import Chem
                            mol = Chem.MolFromSmiles(s)
                            if mol is not None:
                                can = Chem.MolToSmiles(mol, canonical=True)
                                if can in smile2label:
                                    val = smile2label[can]
                                    fallback += 1
                                elif can in global_map:
                                    val = global_map[can]
                                    fallback += 1
                                else:
                                    missing += 1
                                    logger.warning(f"SMILES not found in labeled data: {s}")
                                    continue
                            else:
                                missing += 1
                                logger.warning(f"Invalid SMILES: {s}")
                                continue
                        except Exception:
                            missing += 1
                            logger.warning(f"SMILES not found in labeled data: {s}")
                            continue
                    else:
                        missing += 1
                        logger.warning(f"SMILES not found in labeled data: {s}")
                        continue

                # Ensure JSON-safe
                selected.append(_to_jsonable(val))

            if fallback > 0:
                logger.warning(
                    f"Used global/canonical mapping for {fallback} SMILES in {split} due to split mismatch."
                )
            if missing > 0:
                raise ValueError(
                    f"Cannot find cls_label for {missing} SMILES in {split}. All supervised SMILES must have labels."
                )

            self.final_labels[split] = selected

            # Log distribution only for single-task
            if selected and not any(isinstance(l, (list, tuple)) for l in selected):
                uniq = set(selected)
                counts = {u: selected.count(u) for u in uniq}
                logger.info(f"Selected {len(selected)} labels for {split}. Distribution: {counts}")
            elif selected:
                logger.info(f"Selected {len(selected)} multi-task labels for {split} (distribution skipped)")

        # final consistency
        for split in ["train_id", "val_id", "test_id"]:
            if len(self.final_smiles.get(split, [])) != len(self.final_labels.get(split, [])):
                raise ValueError(
                    f"Final count mismatch in {split}: {len(self.final_smiles.get(split, []))} SMILES vs {len(self.final_labels.get(split, []))} labels"
                )

    # ----- Feature loaders -----
    def _get_features_for_smiles(self, smiles_list):
        features = []
        missing = []
        for s in smiles_list:
            if s in self.feature_cache:
                f = self.feature_cache[s]
                if isinstance(f, np.ndarray):
                    f = torch.from_numpy(f)
                elif not isinstance(f, torch.Tensor):
                    logger.warning(f"Unexpected feature type for {s}: {type(f)}")
                    f = torch.tensor(f)
                features.append(f)
            else:
                missing.append(s)
        if missing:
            logger.error(f"Missing features for {len(missing)} molecules: {missing[:5]}...")
            raise ValueError(f"Missing {len(missing)} molecule features")
        return torch.stack(features)

    def get_training_loader(self, batch_size, num_workers=4):
        if not self.enable_cache:
            raise ValueError(
                "Supervised baseline training requires pre-computed features. Enable --precompute_features"
            )
        feats = self._get_features_for_smiles(self.final_smiles["train_id"])
        labels = self.final_labels["train_id"]
        ds = SupervisedFeatureDataset(feats, labels)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def get_validation_loader(self, batch_size, num_workers=4):
        if not self.enable_cache:
            raise ValueError(
                "Supervised baseline validation requires pre-computed features. Enable --precompute_features"
            )
        feats = self._get_features_for_smiles(self.final_smiles["val_id"])
        labels = self.final_labels["val_id"]
        ds = SupervisedFeatureDataset(feats, labels)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def get_test_loaders(self, batch_size, num_workers=4):
        if not self.enable_cache:
            raise ValueError(
                "Supervised baseline testing requires pre-computed features. Enable --precompute_features"
            )
        id_smiles = self.final_smiles.get("test_id", []) or self.final_smiles.get("val_id", [])
        ood_smiles = self.final_smiles.get("test_ood", []) or self.final_smiles.get("val_ood", [])
        if not id_smiles or not ood_smiles:
            raise ValueError("Neither test nor validation data available for testing")
        id_feats = self._get_features_for_smiles(id_smiles)
        ood_feats = self._get_features_for_smiles(ood_smiles)
        id_ds = TensorDataset(id_feats)
        ood_ds = TensorDataset(ood_feats)
        id_dl = DataLoader(
            id_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
        )
        ood_dl = DataLoader(
            ood_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
        )
        logger.info(
            f"Created test loaders: ID={len(id_ds)} samples, OOD={len(ood_ds)} samples, batch_size={batch_size}"
        )
        return id_dl, ood_dl

    def get_dataloaders(self):
        logger.warning(
            "Using SupervisedBaselineDataLoader.get_dataloaders() - prefer training/validation/test specific methods."
        )
        return super().get_dataloaders()

    def print_label_summary(self):
        logger.info("=" * 50)
        logger.info("Supervised Baseline Label Summary:")
        for split in ["train_id", "val_id", "test_id"]:
            labels = self.final_labels.get(split, [])
            if labels:
                if any(isinstance(l, (list, tuple)) for l in labels):
                    logger.info(f"  - {split:<10}: {len(labels):>5} samples, multi-task labels")
                else:
                    uniq = set(labels)
                    logger.info(f"  - {split:<10}: {len(labels):>5} samples, {len(uniq)} classes")
            else:
                logger.info(f"  - {split:<10}: 0 samples")
        logger.info("=" * 50)
