#!/usr/bin/env python3
"""
Train lightweight KRAS G12C QSAR baselines from public ChEMBL activity data.

This script intentionally lives outside the PDF/docking workflow. It creates a
small target-specific activity model that can later score EASIE analogs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVR, SVR


CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
KRAS_TARGET_CHEMBL_ID = "CHEMBL2189121"
DEFAULT_ENDPOINT = "IC50"
DEFAULT_MUTATION = "G12C"
DEFAULT_RANDOM_SEED = 714


def try_import_rdkit() -> tuple[Any, Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return None, None, None
    return Chem, AllChem, MurckoScaffold


Chem, AllChem, MurckoScaffold = try_import_rdkit()
HAS_RDKIT = Chem is not None and AllChem is not None


def try_import_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except Exception:
        return None
    return lgb


lgb = try_import_lightgbm()
HAS_LIGHTGBM = lgb is not None


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def chembl_get_json(
    endpoint: str,
    params: dict[str, Any],
    *,
    timeout: int,
    retries: int,
    base_delay: float,
) -> dict[str, Any]:
    url = f"{CHEMBL_BASE_URL}/{endpoint}.json"
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(base_delay * (2**attempt))
    raise RuntimeError(f"ChEMBL request failed for {endpoint} with params={params}: {last_error}")


def fetch_all_chembl(
    endpoint: str,
    params: dict[str, Any],
    result_key: str,
    *,
    timeout: int,
    retries: int,
    base_delay: float,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        request_params = dict(params)
        request_params.update({"limit": limit, "offset": offset})
        data = chembl_get_json(
            endpoint,
            request_params,
            timeout=timeout,
            retries=retries,
            base_delay=base_delay,
        )
        batch = data.get(result_key, []) or []
        rows.extend(batch)
        total = int(data.get("page_meta", {}).get("total_count") or len(rows))
        if not batch or len(rows) >= total:
            return rows
        offset += limit


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_mutation_assays(
    mutation: str,
    cache_dir: Path,
    *,
    force: bool,
    timeout: int,
    retries: int,
    base_delay: float,
) -> list[dict[str, Any]]:
    cache_path = cache_dir / f"chembl_assays_{mutation.lower()}.json"
    if cache_path.exists() and not force:
        return read_json(cache_path)
    assays = fetch_all_chembl(
        "assay",
        {
            "target_chembl_id": KRAS_TARGET_CHEMBL_ID,
            "description__icontains": mutation,
        },
        "assays",
        timeout=timeout,
        retries=retries,
        base_delay=base_delay,
    )
    write_json(cache_path, assays)
    return assays


def fetch_activity_for_assay(
    assay_chembl_id: str,
    *,
    timeout: int,
    retries: int,
    base_delay: float,
) -> list[dict[str, Any]]:
    return fetch_all_chembl(
        "activity",
        {"assay_chembl_id": assay_chembl_id},
        "activities",
        timeout=timeout,
        retries=retries,
        base_delay=base_delay,
    )


def fetch_mutation_activities(
    assays: list[dict[str, Any]],
    mutation: str,
    cache_dir: Path,
    *,
    force: bool,
    timeout: int,
    retries: int,
    base_delay: float,
    workers: int,
) -> list[dict[str, Any]]:
    cache_path = cache_dir / f"chembl_activities_{mutation.lower()}.json"
    if cache_path.exists() and not force:
        return read_json(cache_path)

    assay_ids = [str(item.get("assay_chembl_id", "")) for item in assays if item.get("assay_chembl_id")]
    by_assay = {str(item.get("assay_chembl_id")): item for item in assays}
    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                fetch_activity_for_assay,
                assay_id,
                timeout=timeout,
                retries=retries,
                base_delay=base_delay,
            ): assay_id
            for assay_id in assay_ids
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            assay_id = futures[future]
            try:
                rows = future.result()
            except Exception as exc:
                errors.append({"assay_chembl_id": assay_id, "error": str(exc)})
                continue
            assay = by_assay.get(assay_id, {})
            for row in rows:
                row["assay_description"] = assay.get("description", "")
                row["assay_type"] = assay.get("assay_type", "")
                row["assay_document_chembl_id"] = assay.get("document_chembl_id", "")
            all_rows.extend(rows)
            if index % 50 == 0:
                print(f"Fetched {index}/{len(assay_ids)} assays; activity rows={len(all_rows)}", flush=True)

    if errors:
        write_json(cache_dir / f"chembl_activity_fetch_errors_{mutation.lower()}.json", errors)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in all_rows:
        activity_id = str(row.get("activity_id", ""))
        key = activity_id or json.dumps(row, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    write_json(cache_path, deduped)
    return deduped


def fetch_molecules(
    molecule_ids: list[str],
    cache_dir: Path,
    *,
    force: bool,
    timeout: int,
    retries: int,
    base_delay: float,
) -> dict[str, dict[str, Any]]:
    cache_path = cache_dir / "chembl_molecules.json"
    if cache_path.exists() and not force:
        cached = read_json(cache_path)
        return {str(key): value for key, value in cached.items()}

    molecules: dict[str, dict[str, Any]] = {}
    for ids in chunked(sorted(set(molecule_ids)), 100):
        data = fetch_all_chembl(
            "molecule",
            {"molecule_chembl_id__in": ",".join(ids)},
            "molecules",
            timeout=timeout,
            retries=retries,
            base_delay=base_delay,
        )
        for item in data:
            molecule_id = str(item.get("molecule_chembl_id", ""))
            if molecule_id:
                molecules[molecule_id] = item
    write_json(cache_path, molecules)
    return molecules


def normalize_relation(value: Any) -> str:
    relation = str(value or "").strip()
    return relation or "="


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def molecule_smiles(molecule: dict[str, Any]) -> str:
    structures = molecule.get("molecule_structures") or {}
    for key in ("canonical_smiles", "standard_inchi"):
        value = str(structures.get(key, "") or "").strip()
        if value:
            return value
    return ""


def canonicalize_smiles(smiles: str) -> str:
    smiles = str(smiles or "").strip()
    if not smiles:
        return ""
    if not HAS_RDKIT:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def warhead_flags(smiles: str) -> dict[str, int]:
    text = str(smiles or "")
    flags = {
        "warhead_acrylamide_like": 0,
        "warhead_chloroacetamide_like": 0,
        "warhead_vinyl_sulfonamide_like": 0,
        "warhead_cyanoacrylamide_like": 0,
    }
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return flags
        smarts = {
            "warhead_acrylamide_like": "[CX3](=[OX1])[NX3][C,c][C,c].[C,c]=[C,c]",
            "warhead_chloroacetamide_like": "Cl[CH2][CX3](=[OX1])[NX3]",
            "warhead_vinyl_sulfonamide_like": "S(=O)(=O)[NX3][C,c]=[C,c]",
            "warhead_cyanoacrylamide_like": "N#C[C,c]=[C,c][CX3](=[OX1])[NX3]",
        }
        for key, pattern in smarts.items():
            query = Chem.MolFromSmarts(pattern)
            flags[key] = int(query is not None and mol.HasSubstructMatch(query))
        return flags
    lowered = text.lower()
    flags["warhead_chloroacetamide_like"] = int("cl" in lowered and "c(=o)n" in lowered)
    flags["warhead_acrylamide_like"] = int("c=c" in lowered and "c(=o)n" in lowered)
    flags["warhead_vinyl_sulfonamide_like"] = int("s(=o)(=o)" in lowered and "c=c" in lowered)
    flags["warhead_cyanoacrylamide_like"] = int("n#c" in lowered and "c=c" in lowered and "c(=o)n" in lowered)
    return flags


def build_raw_activity_table(
    activities: list[dict[str, Any]],
    molecules: dict[str, dict[str, Any]],
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in activities:
        molecule_id = str(item.get("molecule_chembl_id", "") or "")
        molecule = molecules.get(molecule_id, {})
        rows.append(
            {
                "activity_id": item.get("activity_id", ""),
                "molecule_chembl_id": molecule_id,
                "canonical_smiles": molecule_smiles(molecule),
                "assay_chembl_id": item.get("assay_chembl_id", ""),
                "assay_description": item.get("assay_description", ""),
                "assay_type": item.get("assay_type", ""),
                "document_chembl_id": item.get("document_chembl_id") or item.get("assay_document_chembl_id", ""),
                "standard_type": item.get("standard_type", ""),
                "standard_relation": normalize_relation(item.get("standard_relation")),
                "standard_value": item.get("standard_value", ""),
                "standard_units": item.get("standard_units", ""),
                "pchembl_value": item.get("pchembl_value", ""),
                "data_validity_comment": item.get("data_validity_comment", ""),
                "activity_comment": item.get("activity_comment", ""),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def clean_activity_dataset(
    raw_df: pd.DataFrame,
    *,
    endpoint: str,
    exact_only: bool,
    min_pic50: float,
    max_pic50: float,
) -> pd.DataFrame:
    df = raw_df.copy()
    df["standard_value_float"] = df["standard_value"].map(safe_float)
    df["standard_units_norm"] = df["standard_units"].fillna("").astype(str).str.lower()
    df["standard_relation_norm"] = df["standard_relation"].map(normalize_relation)
    df["canonical_smiles_norm"] = df["canonical_smiles"].map(canonicalize_smiles)
    mask = (
        (df["standard_type"].astype(str) == endpoint)
        & (df["standard_units_norm"] == "nm")
        & df["standard_value_float"].notna()
        & (df["standard_value_float"] > 0)
        & (df["canonical_smiles_norm"] != "")
    )
    if exact_only:
        mask &= df["standard_relation_norm"].eq("=")
    df = df.loc[mask].copy()
    df["p_activity"] = 9.0 - np.log10(df["standard_value_float"].astype(float))
    df = df[(df["p_activity"] >= min_pic50) & (df["p_activity"] <= max_pic50)].copy()

    records: list[dict[str, Any]] = []
    for smiles, group in df.groupby("canonical_smiles_norm", sort=False):
        values = group["p_activity"].astype(float).to_numpy()
        assay_ids = sorted(set(group["assay_chembl_id"].astype(str)))
        document_ids = sorted(set(group["document_chembl_id"].astype(str)))
        row = {
            "canonical_smiles": smiles,
            "p_activity": float(np.median(values)),
            "p_activity_mean": float(np.mean(values)),
            "p_activity_std": float(np.std(values)) if len(values) > 1 else 0.0,
            "n_records": int(len(group)),
            "n_molecules": int(group["molecule_chembl_id"].nunique()),
            "molecule_chembl_ids": ";".join(sorted(set(group["molecule_chembl_id"].astype(str)))),
            "activity_ids": ";".join(str(x) for x in group["activity_id"].tolist()),
            "assay_chembl_ids": ";".join(assay_ids),
            "document_chembl_ids": ";".join(document_ids),
            "primary_assay_chembl_id": assay_ids[0] if assay_ids else "",
            "primary_document_chembl_id": document_ids[0] if document_ids else "",
            "min_standard_value_nm": float(group["standard_value_float"].min()),
            "max_standard_value_nm": float(group["standard_value_float"].max()),
        }
        row.update(warhead_flags(smiles))
        records.append(row)
    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values(["p_activity", "canonical_smiles"], ascending=[False, True]).reset_index(drop=True)


class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self.radius = radius
        self.n_bits = n_bits

    def fit(self, X: Any, y: Any = None) -> "MorganFingerprintTransformer":
        if not HAS_RDKIT:
            raise RuntimeError("RDKit is required for Morgan fingerprints.")
        return self

    def transform(self, X: Any) -> sparse.csr_matrix:
        if not HAS_RDKIT:
            raise RuntimeError("RDKit is required for Morgan fingerprints.")
        try:
            from rdkit.Chem import rdFingerprintGenerator

            generator = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        except Exception:
            generator = None
        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []
        smiles_values = pd.Series(X).astype(str).tolist()
        for row_index, smiles in enumerate(smiles_values):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp = (
                generator.GetFingerprint(mol)
                if generator is not None
                else AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            )
            on_bits = list(fp.GetOnBits())
            rows.extend([row_index] * len(on_bits))
            cols.extend(on_bits)
            data.extend([1] * len(on_bits))
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(smiles_values), self.n_bits), dtype=np.float32)


class TanimotoKNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        k: int = 5,
        radius: int = 2,
        n_bits: int = 2048,
        weight_power: float = 2.0,
        min_similarity: float = 0.0,
    ):
        self.k = k
        self.radius = radius
        self.n_bits = n_bits
        self.weight_power = weight_power
        self.min_similarity = min_similarity

    def _generator(self) -> Any:
        try:
            from rdkit.Chem import rdFingerprintGenerator

            return rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        except Exception:
            return None

    def _fingerprint(self, smiles: str, generator: Any) -> Any:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        if generator is not None:
            return generator.GetFingerprint(mol)
        return AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)

    def fit(self, X: Any, y: Any) -> "TanimotoKNNRegressor":
        if not HAS_RDKIT:
            raise RuntimeError("RDKit is required for Tanimoto nearest-neighbor regression.")
        smiles_values = pd.Series(X).astype(str).tolist()
        y_values = np.asarray(y, dtype=float)
        generator = self._generator()
        fps: list[Any] = []
        labels: list[float] = []
        for smiles, label in zip(smiles_values, y_values):
            fp = self._fingerprint(smiles, generator)
            if fp is None:
                continue
            fps.append(fp)
            labels.append(float(label))
        if not fps:
            raise RuntimeError("No valid fingerprints for Tanimoto nearest-neighbor regression.")
        self.fps_ = fps
        self.y_ = np.asarray(labels, dtype=float)
        self.mean_ = float(np.mean(self.y_))
        self.generator_ = generator
        return self

    def predict(self, X: Any) -> np.ndarray:
        from rdkit import DataStructs

        preds: list[float] = []
        k = max(1, int(self.k))
        for smiles in pd.Series(X).astype(str).tolist():
            fp = self._fingerprint(smiles, self.generator_)
            if fp is None:
                preds.append(self.mean_)
                continue
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, self.fps_), dtype=float)
            if sims.size == 0 or float(np.max(sims)) < float(self.min_similarity):
                preds.append(self.mean_)
                continue
            top_k = min(k, sims.size)
            top_idx = np.argpartition(sims, -top_k)[-top_k:]
            top_sims = sims[top_idx]
            top_y = self.y_[top_idx]
            weights = np.power(np.clip(top_sims, 0.0, 1.0), float(self.weight_power))
            if float(np.sum(weights)) <= 0:
                preds.append(float(np.mean(top_y)))
            else:
                preds.append(float(np.average(top_y, weights=weights)))
        return np.asarray(preds, dtype=float)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("generator_", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "fps_" in state:
            self.generator_ = self._generator()


class AverageRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators: list[tuple[str, Any]], weights: Optional[list[float]] = None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X: Any, y: Any) -> "AverageRegressor":
        self.fitted_estimators_: list[tuple[str, Any]] = []
        for name, estimator in self.estimators:
            fitted = clone(estimator)
            fitted.fit(X, y)
            self.fitted_estimators_.append((name, fitted))
        if self.weights is None:
            self.weights_ = np.ones(len(self.fitted_estimators_), dtype=float)
        else:
            self.weights_ = np.asarray(self.weights, dtype=float)
            if len(self.weights_) != len(self.fitted_estimators_):
                raise RuntimeError("AverageRegressor weights length must match estimators length.")
        return self

    def predict(self, X: Any) -> np.ndarray:
        preds = np.vstack([estimator.predict(X) for _name, estimator in self.fitted_estimators_])
        return np.average(preds, axis=0, weights=self.weights_)


class RDKitDescriptorTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any = None) -> "RDKitDescriptorTransformer":
        if not HAS_RDKIT:
            raise RuntimeError("RDKit is required for molecular descriptors.")
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not HAS_RDKIT:
            raise RuntimeError("RDKit is required for molecular descriptors.")
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

        rows: list[list[float]] = []
        for smiles in pd.Series(X).astype(str).tolist():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                rows.append([0.0] * 18)
                continue
            rows.append(
                [
                    float(Descriptors.MolWt(mol)),
                    float(Descriptors.ExactMolWt(mol)),
                    float(Crippen.MolLogP(mol)),
                    float(rdMolDescriptors.CalcTPSA(mol)),
                    float(Lipinski.NumHDonors(mol)),
                    float(Lipinski.NumHAcceptors(mol)),
                    float(Lipinski.NumRotatableBonds(mol)),
                    float(rdMolDescriptors.CalcNumRings(mol)),
                    float(rdMolDescriptors.CalcNumAromaticRings(mol)),
                    float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
                    float(rdMolDescriptors.CalcFractionCSP3(mol)),
                    float(mol.GetNumHeavyAtoms()),
                    float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
                    float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
                    float(rdMolDescriptors.CalcNumAmideBonds(mol)),
                    float(Descriptors.BertzCT(mol)),
                    float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
                    float(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
                ]
            )
        return np.asarray(rows, dtype=np.float32)


def register_pickle_module_alias() -> None:
    """
    Keep joblib artifacts loadable when this file is executed as a script.

    Without this, custom transformers/functions may be pickled as __main__.*,
    which breaks loading from predict_g12c_activity.py or rank_g12c_candidates.py.
    """
    module = sys.modules[__name__]
    sys.modules.setdefault("train_g12c_qsar", module)
    for obj in (
        MorganFingerprintTransformer,
        TanimotoKNNRegressor,
        AverageRegressor,
        RDKitDescriptorTransformer,
        smiles_identity,
        to_dense_matrix,
    ):
        obj.__module__ = "train_g12c_qsar"


def smiles_identity(X: Any) -> np.ndarray:
    return pd.Series(X).astype(str).to_numpy()


def to_dense_matrix(X: Any) -> np.ndarray:
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def lightgbm_regressor(random_seed: int) -> Any:
    if not HAS_LIGHTGBM:
        raise RuntimeError("LightGBM is not installed.")
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=900,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=random_seed,
        n_jobs=-1,
        verbosity=-1,
    )


def build_model_pipelines(random_seed: int) -> dict[str, Pipeline]:
    text_features = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=12000, lowercase=False)
    models: dict[str, Pipeline] = {
        "char_tfidf_ridge": Pipeline(
            [
                ("identity", FunctionTransformer(smiles_identity, validate=False)),
                ("tfidf", text_features),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "char_tfidf_linear_svr": Pipeline(
            [
                ("identity", FunctionTransformer(smiles_identity, validate=False)),
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=12000, lowercase=False)),
                ("model", LinearSVR(C=1.0, epsilon=0.05, random_state=random_seed, max_iter=8000)),
            ]
        ),
        "char_tfidf_svd_random_forest": Pipeline(
            [
                ("identity", FunctionTransformer(smiles_identity, validate=False)),
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=12000, lowercase=False)),
                ("svd", TruncatedSVD(n_components=192, random_state=random_seed)),
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_seed)),
            ]
        ),
        "char_tfidf_svd_extra_trees": Pipeline(
            [
                ("identity", FunctionTransformer(smiles_identity, validate=False)),
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=12000, lowercase=False)),
                ("svd", TruncatedSVD(n_components=192, random_state=random_seed)),
                ("scaler", StandardScaler()),
                ("model", ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=random_seed)),
            ]
        ),
        "char_tfidf_svd_hist_gradient_boosting": Pipeline(
            [
                ("identity", FunctionTransformer(smiles_identity, validate=False)),
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=12000, lowercase=False)),
                ("svd", TruncatedSVD(n_components=192, random_state=random_seed)),
                ("scaler", StandardScaler()),
                ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=0.05, random_state=random_seed)),
            ]
        ),
    }

    if HAS_LIGHTGBM:
        models["char_tfidf_svd_lightgbm"] = Pipeline(
            [
                ("identity", FunctionTransformer(smiles_identity, validate=False)),
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=12000, lowercase=False)),
                ("svd", TruncatedSVD(n_components=256, random_state=random_seed)),
                ("model", lightgbm_regressor(random_seed)),
            ]
        )

    if HAS_RDKIT:
        models.update(
            {
                "morgan_ridge": Pipeline(
                    [
                        ("fp", MorganFingerprintTransformer(radius=2, n_bits=2048)),
                        ("model", Ridge(alpha=1.0)),
                    ]
                ),
                "morgan_elastic_net": Pipeline(
                    [
                        ("fp", MorganFingerprintTransformer(radius=2, n_bits=2048)),
                        ("model", ElasticNet(alpha=0.002, l1_ratio=0.05, random_state=random_seed, max_iter=8000)),
                    ]
                ),
                "morgan_linear_svr": Pipeline(
                    [
                        ("fp", MorganFingerprintTransformer(radius=2, n_bits=2048)),
                        ("model", LinearSVR(C=0.5, epsilon=0.05, random_state=random_seed, max_iter=8000)),
                    ]
                ),
                "morgan_random_forest": Pipeline(
                    [
                        ("fp", MorganFingerprintTransformer(radius=2, n_bits=2048)),
                        ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=random_seed)),
                    ]
                ),
                "morgan_extra_trees": Pipeline(
                    [
                        ("fp", MorganFingerprintTransformer(radius=2, n_bits=2048)),
                        ("model", ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, n_jobs=-1, random_state=random_seed)),
                    ]
                ),
                "morgan_hist_gradient_boosting": Pipeline(
                    [
                        ("fp", MorganFingerprintTransformer(radius=2, n_bits=2048)),
                        ("dense", FunctionTransformer(to_dense_matrix, accept_sparse=True)),
                        ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=0.05, random_state=random_seed)),
                    ]
                ),
                "rdkit_desc_ridge": Pipeline(
                    [
                        ("desc", RDKitDescriptorTransformer()),
                        ("scale", StandardScaler()),
                        ("model", Ridge(alpha=1.0)),
                    ]
                ),
                "rdkit_desc_elastic_net": Pipeline(
                    [
                        ("desc", RDKitDescriptorTransformer()),
                        ("scale", StandardScaler()),
                        ("model", ElasticNet(alpha=0.01, l1_ratio=0.15, random_state=random_seed, max_iter=8000)),
                    ]
                ),
                "rdkit_desc_rbf_svr": Pipeline(
                    [
                        ("desc", RDKitDescriptorTransformer()),
                        ("scale", StandardScaler()),
                        ("model", SVR(C=4.0, epsilon=0.08, gamma="scale")),
                    ]
                ),
                "rdkit_desc_random_forest": Pipeline(
                    [
                        ("desc", RDKitDescriptorTransformer()),
                        ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=random_seed)),
                    ]
                ),
                "rdkit_desc_extra_trees": Pipeline(
                    [
                        ("desc", RDKitDescriptorTransformer()),
                        ("model", ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=random_seed)),
                    ]
                ),
                "rdkit_desc_hist_gradient_boosting": Pipeline(
                    [
                        ("desc", RDKitDescriptorTransformer()),
                        ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, l2_regularization=0.05, random_state=random_seed)),
                    ]
                ),
                "morgan_plus_desc_random_forest": Pipeline(
                    [
                        (
                            "features",
                            FeatureUnion(
                                [
                                    ("fp", MorganFingerprintTransformer(radius=2, n_bits=2048)),
                                    ("desc", Pipeline([("desc", RDKitDescriptorTransformer()), ("scale", StandardScaler())])),
                                ]
                            ),
                        ),
                        ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=random_seed)),
                    ]
                ),
                "morgan_plus_desc_extra_trees": Pipeline(
                    [
                        (
                            "features",
                            FeatureUnion(
                                [
                                    ("fp", MorganFingerprintTransformer(radius=2, n_bits=2048)),
                                    ("desc", Pipeline([("desc", RDKitDescriptorTransformer()), ("scale", StandardScaler())])),
                                ]
                            ),
                        ),
                        ("model", ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, n_jobs=-1, random_state=random_seed)),
                    ]
                ),
            }
        )
        if HAS_LIGHTGBM:
            models.update(
                {
                    "morgan_tanimoto_knn_k3": TanimotoKNNRegressor(k=3, weight_power=2.0),
                    "morgan_tanimoto_knn_k5": TanimotoKNNRegressor(k=5, weight_power=2.0),
                    "morgan_tanimoto_knn_k10": TanimotoKNNRegressor(k=10, weight_power=2.0),
                    "morgan_lightgbm": Pipeline(
                        [
                            ("fp", MorganFingerprintTransformer(radius=2, n_bits=2048)),
                            ("model", lightgbm_regressor(random_seed)),
                        ]
                    ),
                    "morgan_plus_desc_lightgbm": Pipeline(
                        [
                            (
                                "features",
                                FeatureUnion(
                                    [
                                        ("fp", MorganFingerprintTransformer(radius=2, n_bits=2048)),
                                        ("desc", Pipeline([("desc", RDKitDescriptorTransformer()), ("scale", StandardScaler())])),
                                    ]
                                ),
                            ),
                            ("model", lightgbm_regressor(random_seed)),
                        ]
                    ),
                    "rdkit_desc_lightgbm": Pipeline(
                        [
                            ("desc", RDKitDescriptorTransformer()),
                            ("model", lightgbm_regressor(random_seed)),
                        ]
                    ),
                }
            )
        else:
            models.update(
                {
                    "morgan_tanimoto_knn_k3": TanimotoKNNRegressor(k=3, weight_power=2.0),
                    "morgan_tanimoto_knn_k5": TanimotoKNNRegressor(k=5, weight_power=2.0),
                    "morgan_tanimoto_knn_k10": TanimotoKNNRegressor(k=10, weight_power=2.0),
                }
            )
        models["ensemble_nn_ridge_rf"] = AverageRegressor(
            [
                ("nn5", TanimotoKNNRegressor(k=5, weight_power=2.0)),
                ("char_ridge", models["char_tfidf_ridge"]),
                ("morgan_rf", models["morgan_random_forest"]),
            ]
        )
        if HAS_LIGHTGBM:
            models["ensemble_nn_ridge_lgbm"] = AverageRegressor(
                [
                    ("nn5", TanimotoKNNRegressor(k=5, weight_power=2.0)),
                    ("char_ridge", models["char_tfidf_ridge"]),
                    ("morgan_lgbm", models["morgan_lightgbm"]),
                ]
            )
    return models


def optuna_supported_models() -> list[str]:
    names = [
        "char_tfidf_ridge",
        "char_tfidf_linear_svr",
        "char_tfidf_svd_extra_trees",
        "char_tfidf_svd_hist_gradient_boosting",
    ]
    if HAS_RDKIT:
        names.extend(
            [
                "morgan_random_forest",
                "morgan_extra_trees",
                "morgan_tanimoto_knn",
                "rdkit_desc_elastic_net",
                "rdkit_desc_random_forest",
                "rdkit_desc_extra_trees",
                "morgan_plus_desc_random_forest",
                "morgan_plus_desc_extra_trees",
            ]
        )
    if HAS_LIGHTGBM:
        names.append("char_tfidf_svd_lightgbm")
        if HAS_RDKIT:
            names.extend(["morgan_lightgbm", "rdkit_desc_lightgbm", "morgan_plus_desc_lightgbm"])
    return names


def default_optuna_models() -> list[str]:
    names = ["char_tfidf_ridge", "char_tfidf_linear_svr"]
    if HAS_RDKIT:
        names.append("rdkit_desc_elastic_net")
    if HAS_LIGHTGBM:
        names.append("morgan_plus_desc_lightgbm" if HAS_RDKIT else "char_tfidf_svd_lightgbm")
    return names


def parse_csv_names(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_optuna_model_names(value: str) -> list[str]:
    text = str(value or "default").strip().lower()
    if text in {"", "default"}:
        return default_optuna_models()
    if text == "all":
        return optuna_supported_models()
    requested = parse_csv_names(value)
    supported = set(optuna_supported_models())
    unknown = [name for name in requested if name not in supported]
    if unknown:
        raise RuntimeError(
            "Unsupported or unavailable Optuna model(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(optuna_supported_models())
        )
    return requested


def filter_models(models: dict[str, Any], value: str) -> dict[str, Any]:
    requested = parse_csv_names(value)
    if not requested:
        return models
    unknown = [name for name in requested if name not in models]
    if unknown:
        raise RuntimeError(
            "Unknown model-filter value(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(sorted(models))
        )
    return {name: models[name] for name in requested}


def optuna_tfidf_vectorizer(trial: Any) -> TfidfVectorizer:
    ngram_upper = trial.suggest_int("ngram_upper", 3, 6)
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, ngram_upper),
        min_df=trial.suggest_int("min_df", 1, 4),
        max_features=trial.suggest_int("max_features", 4000, 30000, step=2000),
        lowercase=False,
    )


def optuna_lightgbm_regressor(trial: Any, random_seed: int) -> Any:
    if not HAS_LIGHTGBM:
        raise RuntimeError("LightGBM is not installed.")
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=trial.suggest_int("n_estimators", 200, 1400, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.008, 0.12, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 95, step=8),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
        subsample=trial.suggest_float("subsample", 0.60, 1.00),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.45, 1.00),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=random_seed,
        n_jobs=-1,
        verbosity=-1,
    )


def optuna_tree_regressor(trial: Any, random_seed: int, *, extra_trees: bool) -> Any:
    estimator_class = ExtraTreesRegressor if extra_trees else RandomForestRegressor
    return estimator_class(
        n_estimators=trial.suggest_int("n_estimators", 150, 800, step=50),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
        max_features=trial.suggest_categorical("tree_max_features", ["sqrt", "log2", 0.50, 0.75, 1.0]),
        n_jobs=-1,
        random_state=random_seed,
    )


def build_optuna_model(model_name: str, trial: Any, random_seed: int) -> Any:
    if model_name == "char_tfidf_ridge":
        return Pipeline(
            [
                ("identity", FunctionTransformer(smiles_identity, validate=False)),
                ("tfidf", optuna_tfidf_vectorizer(trial)),
                ("model", Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100.0, log=True))),
            ]
        )
    if model_name == "char_tfidf_linear_svr":
        return Pipeline(
            [
                ("identity", FunctionTransformer(smiles_identity, validate=False)),
                ("tfidf", optuna_tfidf_vectorizer(trial)),
                (
                    "model",
                    LinearSVR(
                        C=trial.suggest_float("C", 0.03, 20.0, log=True),
                        epsilon=trial.suggest_float("epsilon", 0.005, 0.25, log=True),
                        random_state=random_seed,
                        max_iter=12000,
                    ),
                ),
            ]
        )
    if model_name in {"char_tfidf_svd_extra_trees", "char_tfidf_svd_hist_gradient_boosting", "char_tfidf_svd_lightgbm"}:
        steps: list[tuple[str, Any]] = [
            ("identity", FunctionTransformer(smiles_identity, validate=False)),
            ("tfidf", optuna_tfidf_vectorizer(trial)),
            ("svd", TruncatedSVD(n_components=trial.suggest_int("n_components", 64, 384, step=32), random_state=random_seed)),
        ]
        if model_name == "char_tfidf_svd_extra_trees":
            steps.extend(
                [
                    ("scaler", StandardScaler()),
                    ("model", optuna_tree_regressor(trial, random_seed, extra_trees=True)),
                ]
            )
        elif model_name == "char_tfidf_svd_hist_gradient_boosting":
            steps.extend(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=trial.suggest_int("max_iter", 120, 700, step=40),
                            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                            l2_regularization=trial.suggest_float("l2_regularization", 1e-5, 1.0, log=True),
                            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
                            random_state=random_seed,
                        ),
                    ),
                ]
            )
        else:
            steps.append(("model", optuna_lightgbm_regressor(trial, random_seed)))
        return Pipeline(steps)

    if not HAS_RDKIT:
        raise RuntimeError(f"{model_name} requires RDKit.")

    if model_name == "morgan_tanimoto_knn":
        return TanimotoKNNRegressor(
            k=trial.suggest_int("k", 3, 20),
            radius=trial.suggest_int("radius", 2, 3),
            n_bits=trial.suggest_categorical("n_bits", [1024, 2048]),
            weight_power=trial.suggest_float("weight_power", 0.75, 4.0),
            min_similarity=trial.suggest_float("min_similarity", 0.0, 0.35),
        )

    morgan_steps: list[tuple[str, Any]]
    if model_name.startswith("morgan_plus_desc_"):
        morgan_steps = [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "fp",
                            MorganFingerprintTransformer(
                                radius=trial.suggest_int("radius", 2, 3),
                                n_bits=trial.suggest_categorical("n_bits", [1024, 2048]),
                            ),
                        ),
                        ("desc", Pipeline([("desc", RDKitDescriptorTransformer()), ("scale", StandardScaler())])),
                    ]
                ),
            )
        ]
    elif model_name.startswith("morgan_"):
        morgan_steps = [
            (
                "fp",
                MorganFingerprintTransformer(
                    radius=trial.suggest_int("radius", 2, 3),
                    n_bits=trial.suggest_categorical("n_bits", [1024, 2048]),
                ),
            )
        ]
    elif model_name.startswith("rdkit_desc_"):
        morgan_steps = [("desc", RDKitDescriptorTransformer())]
        if "elastic_net" in model_name:
            morgan_steps.append(("scale", StandardScaler()))
    else:
        raise RuntimeError(f"Unsupported Optuna model: {model_name}")

    if model_name.endswith("_elastic_net"):
        morgan_steps.append(
            (
                "model",
                ElasticNet(
                    alpha=trial.suggest_float("alpha", 1e-4, 0.5, log=True),
                    l1_ratio=trial.suggest_float("l1_ratio", 0.0, 0.95),
                    random_state=random_seed,
                    max_iter=12000,
                ),
            )
        )
    elif model_name.endswith("_random_forest"):
        morgan_steps.append(("model", optuna_tree_regressor(trial, random_seed, extra_trees=False)))
    elif model_name.endswith("_extra_trees"):
        morgan_steps.append(("model", optuna_tree_regressor(trial, random_seed, extra_trees=True)))
    elif model_name.endswith("_lightgbm"):
        morgan_steps.append(("model", optuna_lightgbm_regressor(trial, random_seed)))
    else:
        raise RuntimeError(f"Unsupported Optuna model: {model_name}")
    return Pipeline(morgan_steps)


def murcko_scaffold(smiles: str) -> str:
    if not HAS_RDKIT:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=False) if scaffold is not None else ""


def scaffold_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not HAS_RDKIT:
        raise RuntimeError("RDKit is required for scaffold split.")
    scaffolds: dict[str, list[int]] = {}
    for index, smiles in enumerate(df["canonical_smiles"].astype(str)):
        scaffold = murcko_scaffold(smiles)
        if not scaffold:
            scaffold = hashlib.sha1(smiles.encode("utf-8")).hexdigest()
        scaffolds.setdefault(scaffold, []).append(index)
    groups = sorted(scaffolds.values(), key=lambda values: (-len(values), values[0]))
    rng = np.random.default_rng(random_seed)
    same_size_blocks: dict[int, list[list[int]]] = {}
    for group in groups:
        same_size_blocks.setdefault(len(group), []).append(group)
    shuffled: list[list[int]] = []
    for _size, block in same_size_blocks.items():
        rng.shuffle(block)
        shuffled.extend(block)
    shuffled.sort(key=lambda values: -len(values))
    target_test_count = max(1, int(round(len(df) * test_size)))
    test_indices: list[int] = []
    train_indices: list[int] = []
    for group in shuffled:
        if len(test_indices) < target_test_count:
            test_indices.extend(group)
        else:
            train_indices.extend(group)
    return np.array(train_indices, dtype=int), np.array(test_indices, dtype=int)


def grouped_train_test_split(
    df: pd.DataFrame,
    group_column: str,
    *,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = df[group_column].fillna("").astype(str).to_numpy()
    groups = np.asarray(
        [
            group if group else f"missing_group_{index}"
            for index, group in enumerate(groups)
        ]
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


@dataclass
class SplitData:
    name: str
    train_index: np.ndarray
    test_index: np.ndarray


def make_splits(df: pd.DataFrame, *, test_size: float, random_seed: int) -> list[SplitData]:
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_seed)
    splits = [SplitData("random", np.array(train_idx), np.array(test_idx))]
    if HAS_RDKIT:
        scaffold_train, scaffold_test = scaffold_train_test_split(df, test_size=test_size, random_seed=random_seed)
        splits.append(SplitData("scaffold", scaffold_train, scaffold_test))
    for split_name, group_column in (
        ("document", "primary_document_chembl_id"),
        ("assay", "primary_assay_chembl_id"),
    ):
        if group_column in df.columns and df[group_column].fillna("").astype(str).nunique() > 1:
            group_train, group_test = grouped_train_test_split(
                df,
                group_column,
                test_size=test_size,
                random_seed=random_seed,
            )
            splits.append(SplitData(split_name, group_train, group_test))
    return splits


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rho = spearmanr(y_true, y_pred).statistic
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman": float(0.0 if np.isnan(rho) else rho),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    models: dict[str, Pipeline],
    splits: list[SplitData],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    X = df["canonical_smiles"].astype(str)
    y = df["p_activity"].astype(float).to_numpy()
    metric_rows: list[dict[str, Any]] = []
    fitted_by_key: dict[str, Pipeline] = {}
    random_predictions: Optional[pd.DataFrame] = None
    all_predictions: list[pd.DataFrame] = []

    for split in splits:
        for name, pipeline in models.items():
            key = f"{name}__{split.name}"
            print(f"Training {name} on {split.name} split", flush=True)
            model = clone(pipeline)
            model.fit(X.iloc[split.train_index], y[split.train_index])
            pred = model.predict(X.iloc[split.test_index])
            metrics = regression_metrics(y[split.test_index], pred)
            metric_rows.append(
                {
                    "model": name,
                    "split": split.name,
                    "n_train": int(len(split.train_index)),
                    "n_test": int(len(split.test_index)),
                    **metrics,
                }
            )
            fitted_by_key[key] = model
            pred_df = df.iloc[split.test_index].copy()
            pred_df["prediction_model"] = name
            pred_df["split"] = split.name
            pred_df["predicted_p_activity"] = pred
            pred_df["prediction_error"] = pred_df["predicted_p_activity"] - pred_df["p_activity"]
            pred_df["abs_prediction_error"] = pred_df["prediction_error"].abs()
            all_predictions.append(pred_df)
            if split.name == "random":
                random_predictions = pred_df if random_predictions is None else pd.concat([random_predictions, pred_df], ignore_index=True)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["split", "rmse", "mae"], ascending=[True, True, True])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    all_predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    all_predictions_df.to_csv(output_dir / "predictions_all_splits.csv", index=False)
    predictions_df = random_predictions if random_predictions is not None else pd.DataFrame()
    predictions_df.to_csv(output_dir / "predictions_random_split.csv", index=False)
    return metrics_df, fitted_by_key, predictions_df


def select_best_model(metrics_df: pd.DataFrame) -> tuple[str, str]:
    available_splits = set(metrics_df["split"])
    preferred_split = next(
        (split for split in ("document", "scaffold", "assay", "random") if split in available_splits),
        str(metrics_df["split"].iloc[0]),
    )
    subset = metrics_df[metrics_df["split"] == preferred_split].sort_values(["rmse", "mae", "spearman"], ascending=[True, True, False])
    best = subset.iloc[0]
    return str(best["model"]), str(best["split"])


def fit_best_on_full_dataset(df: pd.DataFrame, pipeline: Pipeline) -> Pipeline:
    X = df["canonical_smiles"].astype(str)
    y = df["p_activity"].astype(float).to_numpy()
    model = clone(pipeline)
    model.fit(X, y)
    return model


def select_optuna_split(splits: list[SplitData], requested: str) -> SplitData:
    by_name = {split.name: split for split in splits}
    requested_norm = str(requested or "auto").strip().lower()
    if requested_norm and requested_norm != "auto":
        if requested_norm not in by_name:
            raise RuntimeError(
                f"Optuna split {requested!r} is not available. Available splits: {', '.join(by_name)}"
            )
        return by_name[requested_norm]
    for split_name in ("document", "scaffold", "random", "assay"):
        if split_name in by_name:
            return by_name[split_name]
    return splits[0]


def run_optuna_search(
    df: pd.DataFrame,
    *,
    model_names: list[str],
    split: SplitData,
    output_dir: Path,
    random_seed: int,
    n_trials: int,
    timeout_seconds: int,
    storage: str,
    study_name_prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed. Use the sandbox venv with .venv-optuna/bin/python "
            "or install it with: python -m pip install optuna"
        ) from exc

    X = df["canonical_smiles"].astype(str)
    y = df["p_activity"].astype(float).to_numpy()
    tuned_models: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []

    for model_name in model_names:
        print(f"Optuna tuning {model_name} on {split.name} split for {n_trials} trials", flush=True)

        def objective(trial: Any) -> float:
            try:
                model = build_optuna_model(model_name, trial, random_seed)
                model.fit(X.iloc[split.train_index], y[split.train_index])
                pred = model.predict(X.iloc[split.test_index])
                metrics = regression_metrics(y[split.test_index], pred)
                for key, value in metrics.items():
                    trial.set_user_attr(key, value)
                return float(metrics["rmse"])
            except Exception as exc:
                trial.set_user_attr("error", str(exc)[:1000])
                return float("inf")

        study_name = f"{study_name_prefix}_{model_name}" if study_name_prefix else None
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage or None,
            load_if_exists=bool(storage),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds or None,
            show_progress_bar=False,
        )
        trials_path = output_dir / f"optuna_{model_name}_trials.csv"
        study.trials_dataframe().to_csv(trials_path, index=False)
        if not np.isfinite(float(study.best_value)):
            raise RuntimeError(f"All Optuna trials failed for {model_name}; see {trials_path}")
        tuned_name = f"optuna_{model_name}"
        tuned_models[tuned_name] = build_optuna_model(
            model_name,
            optuna.trial.FixedTrial(study.best_params),
            random_seed,
        )
        summary = {
            "base_model": model_name,
            "tuned_model": tuned_name,
            "split": split.name,
            "n_trials": len(study.trials),
            "best_rmse": float(study.best_value),
            "best_params": study.best_params,
            "trials_csv": str(trials_path),
            "study_name": study.study_name,
            "storage": storage,
        }
        summaries.append(summary)
        print(f"Best Optuna {model_name} RMSE={study.best_value:.4f}", flush=True)

    write_json(output_dir / "optuna_summary.json", summaries)
    return tuned_models, summaries


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KRAS G12C QSAR baselines from ChEMBL.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Run output directory. Defaults to runs/activity_model/g12c_<timestamp>.",
    )
    parser.add_argument("--cache-dir", default="runs/activity_model/cache", help="ChEMBL API cache directory.")
    parser.add_argument("--mutation", default=DEFAULT_MUTATION)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, choices=["IC50", "Ki", "Kd", "EC50"])
    parser.add_argument("--include-inequalities", action="store_true", help="Include < and > values as if exact. Default keeps exact '=' only.")
    parser.add_argument("--min-pactivity", type=float, default=3.0)
    parser.add_argument("--max-pactivity", type=float, default=11.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--base-delay", type=float, default=1.0)
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--model-filter", default="", help="Comma-separated baseline model names to train/evaluate. Default uses all available baselines.")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Run Optuna tuning before final model comparison when > 0.")
    parser.add_argument("--optuna-models", default="default", help="Comma-separated models to tune, 'default', or 'all'.")
    parser.add_argument("--optuna-split", default="auto", help="Validation split for Optuna: auto, random, scaffold, document, or assay.")
    parser.add_argument("--optuna-timeout", type=int, default=0, help="Optional total seconds per Optuna study.")
    parser.add_argument(
        "--optuna-storage",
        default="",
        help="Optional Optuna storage URL, e.g. sqlite:///runs/activity_model/cache/optuna.db.",
    )
    parser.add_argument("--optuna-study-name", default="", help="Optional Optuna study-name prefix. Model name is appended.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    register_pickle_module_alias()
    task_root = Path(__file__).resolve().parents[2]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else task_root / "runs" / "activity_model" / f"{args.mutation.lower()}_{now_stamp()}"
    )
    output_dir = ensure_dir(output_dir)
    cache_dir = ensure_dir(Path(args.cache_dir))

    print(f"Output directory: {output_dir}")
    print(f"RDKit available: {HAS_RDKIT}")
    print(f"LightGBM available: {HAS_LIGHTGBM}")

    assays = fetch_mutation_assays(
        args.mutation,
        cache_dir,
        force=args.force_fetch,
        timeout=args.timeout,
        retries=args.retries,
        base_delay=args.base_delay,
    )
    print(f"Fetched/cached assays: {len(assays)}")

    activities = fetch_mutation_activities(
        assays,
        args.mutation,
        cache_dir,
        force=args.force_fetch,
        timeout=args.timeout,
        retries=args.retries,
        base_delay=args.base_delay,
        workers=args.workers,
    )
    print(f"Fetched/cached activity rows: {len(activities)}")

    molecule_ids = sorted({str(item.get("molecule_chembl_id", "")) for item in activities if item.get("molecule_chembl_id")})
    molecules = fetch_molecules(
        molecule_ids,
        cache_dir,
        force=args.force_fetch,
        timeout=args.timeout,
        retries=args.retries,
        base_delay=args.base_delay,
    )
    print(f"Fetched/cached molecules: {len(molecules)}")

    raw_df = build_raw_activity_table(activities, molecules, output_dir / f"raw_{args.mutation.lower()}_chembl_activities.csv")
    dataset = clean_activity_dataset(
        raw_df,
        endpoint=args.endpoint,
        exact_only=not args.include_inequalities,
        min_pic50=args.min_pactivity,
        max_pic50=args.max_pactivity,
    )
    dataset_path = output_dir / f"{args.mutation.lower()}_{args.endpoint.lower()}_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    if len(dataset) < 50:
        raise RuntimeError(f"Cleaned dataset too small for modeling: {len(dataset)} rows")
    print(f"Cleaned dataset rows: {len(dataset)}")

    models = filter_models(build_model_pipelines(args.random_seed), args.model_filter)
    splits = make_splits(dataset, test_size=args.test_size, random_seed=args.random_seed)
    optuna_summary: list[dict[str, Any]] = []
    if args.optuna_trials > 0:
        optuna_model_names = parse_optuna_model_names(args.optuna_models)
        optuna_split = select_optuna_split(splits, args.optuna_split)
        tuned_models, optuna_summary = run_optuna_search(
            dataset,
            model_names=optuna_model_names,
            split=optuna_split,
            output_dir=output_dir,
            random_seed=args.random_seed,
            n_trials=args.optuna_trials,
            timeout_seconds=args.optuna_timeout,
            storage=args.optuna_storage,
            study_name_prefix=args.optuna_study_name,
        )
        models.update(tuned_models)
    metrics_df, _fitted_by_key, _predictions = train_and_evaluate(dataset, models, splits, output_dir)
    best_model_name, best_split = select_best_model(metrics_df)
    best_model = fit_best_on_full_dataset(dataset, models[best_model_name])
    joblib.dump(best_model, output_dir / "best_model.joblib")

    metadata = {
        "mutation": args.mutation,
        "target_chembl_id": KRAS_TARGET_CHEMBL_ID,
        "endpoint": args.endpoint,
        "exact_only": not args.include_inequalities,
        "rdkit_available": HAS_RDKIT,
        "lightgbm_available": HAS_LIGHTGBM,
        "n_assays": len(assays),
        "n_activity_rows_raw": len(activities),
        "n_molecules_raw": len(molecule_ids),
        "n_dataset_rows": len(dataset),
        "best_model": best_model_name,
        "best_selection_split": best_split,
        "best_metric_row": metrics_df[(metrics_df["model"] == best_model_name) & (metrics_df["split"] == best_split)].iloc[0].to_dict(),
        "model_filter": args.model_filter,
        "optuna": {
            "enabled": args.optuna_trials > 0,
            "trials_per_study": args.optuna_trials,
            "models": parse_optuna_model_names(args.optuna_models) if args.optuna_trials > 0 else [],
            "split": optuna_summary[0]["split"] if optuna_summary else "",
            "timeout_seconds": args.optuna_timeout,
            "storage": args.optuna_storage,
            "summary_json": str(output_dir / "optuna_summary.json") if optuna_summary else "",
            "studies": optuna_summary,
        },
        "outputs": {
            "dataset_csv": str(dataset_path),
            "metrics_csv": str(output_dir / "metrics.csv"),
            "best_model": str(output_dir / "best_model.joblib"),
            "predictions_random_split_csv": str(output_dir / "predictions_random_split.csv"),
            "predictions_all_splits_csv": str(output_dir / "predictions_all_splits.csv"),
        },
        "data_source": {
            "chembl_api_base_url": CHEMBL_BASE_URL,
            "assay_filter": {
                "target_chembl_id": KRAS_TARGET_CHEMBL_ID,
                "description__icontains": args.mutation,
            },
        },
    }
    write_json(output_dir / "best_model_metadata.json", metadata)

    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_model_name} selected on {best_split} split")
    print(f"Wrote {output_dir / 'best_model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
