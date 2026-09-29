"""Prepare case data outside the pinned checkout; every source URL is configurable.

    python tasks/researchgym/scripts/prepare_case_data.py tse --data-root DIR
    python tasks/researchgym/scripts/prepare_case_data.py cl  --data-root DIR [--cifar-url URL]
    python tasks/researchgym/scripts/prepare_case_data.py cmr --data-root DIR --upstream-root ~/ResearchGym

The evaluator links DIR/data, DIR/model, DIR/dataset, DIR/weights and
DIR/ldm_configs into each workspace (see resources/cases/catalog.json). Run
cmr corruption generation in the case environment (it needs Pillow, NumPy and
imagecorruptions). The TSE state classifiers in DIR/model are trained with the
upstream real/main.py and are not produced here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT.parents[1]))

CIFAR_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
CORRUPTIONS = ("gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise", "defocus_blur", "glass_blur",
               "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform",
               "pixelate", "jpeg_compression")


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-fsSL", "--retry", "8", "--retry-delay", "5", "--retry-all-errors",
                    "--connect-timeout", "30", "-o", str(target), url], check=True)
    return target


def extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as bundle:
        try:
            bundle.extractall(destination, filter="data")
        except TypeError:  # Python without tarfile extraction filters
            bundle.extractall(destination)


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_tse(args) -> dict:
    """Five real datasets with the file names datasets/*.py expect."""
    data = args.data_root / "data"
    data.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        for file_id, archive, target in (("7495583", "PAM", "PAM"), ("7495586", "Epilepsy", "epilepsy"),
                                         ("7495589", "Boiler", "boiler")):
            if (data / target).is_dir():
                continue
            path = download(f"{args.dataverse_url}/{file_id}", work / f"{archive}.tar.gz")
            extract(path, work)
            shutil.move(str(work / archive), str(data / target))
        names = ("Wafer", "FreezerRegularTrain")
        if not all((data / n / f"{n}_TRAIN.txt").exists() for n in names):
            archive = download(args.ucr_url, work / "ucr.zip")
            with zipfile.ZipFile(archive) as bundle:
                for name in names:
                    (data / name).mkdir(parents=True, exist_ok=True)
                    for split in ("TRAIN", "TEST"):
                        member = f"UCRArchive_2018/{name}/{name}_{split}.tsv"
                        (data / name / f"{name}_{split}.txt").write_bytes(
                            bundle.read(member, pwd=args.ucr_password.encode()))
    return {"data": str(data), "datasets": sorted(p.name for p in data.iterdir())}


def prepare_cl(args) -> dict:
    """The official CIFAR-100 archive, verified by its published md5."""
    data = args.data_root / "data"
    data.mkdir(parents=True, exist_ok=True)
    archive = data / "cifar-100-python.tar.gz"
    if not archive.exists():
        download(args.cifar_url, archive)
    if digest(archive, "md5") != CIFAR_MD5:
        raise SystemExit(f"{archive} is not the official CIFAR-100 archive (md5 mismatch); use a mirror of the "
                         "original file rather than a re-encoded copy")
    if not (data / "cifar-100-python").is_dir():
        extract(archive, data)
    return {"data": str(data), "archive_md5": CIFAR_MD5}


def prepare_cmr(args) -> dict:
    """Severity-5 COCO-C queries, a seed-0 1000-image subset, and per-corruption configs."""
    from tasks.researchgym.core.cases import load_case
    from tasks.researchgym.core.source import UpstreamCase

    upstream = UpstreamCase(args.upstream_root, load_case("cross_modal_retrieval"))
    dataset = args.data_root / "dataset"
    annotations = json.loads((dataset / "coco/coco_annotations/coco_karpathy_test.json").read_text())
    if args.generate_corruptions:
        _generate_corruptions(dataset, annotations, args.workers)
    rng = random.Random(0)
    keep = set(rng.sample(range(len(annotations)), args.subset))
    subset = [row for i, row in enumerate(annotations) if i in keep]
    subset_dir = dataset / f"coco_sub{args.subset}/coco_annotations"
    subset_dir.mkdir(parents=True, exist_ok=True)
    subset_path = subset_dir / "coco_karpathy_test.json"
    subset_path.write_text(json.dumps(subset))
    configs = args.data_root / "ldm_configs" / f"coco_image_c_sub{args.subset}"
    configs.mkdir(parents=True, exist_ok=True)
    written = {}
    for corruption in CORRUPTIONS:
        relative = f"configs/QS/i2t/coco_image_c/COCO_IP_{corruption}_5.yaml"
        source = upstream.root / relative
        if digest(source, "sha256") != upstream.pinned["files"][relative]:
            raise SystemExit(f"{relative} does not match the pinned ResearchGym source")
        text = source.read_text()
        if "ann_root: ./dataset/coco/coco_annotations/" not in text:
            raise SystemExit(f"unexpected ann_root in {relative}")
        text = text.replace("ann_root: ./dataset/coco/coco_annotations/",
                            f"ann_root: ./dataset/coco_sub{args.subset}/coco_annotations/")
        (configs / f"COCO_IP_{corruption}_5.yaml").write_text(text)
        written[corruption] = digest(configs / f"COCO_IP_{corruption}_5.yaml", "sha256")
    manifest = {"subset_size": len(subset), "subset_seed": 0, "subset_sha256": digest(subset_path, "sha256"),
                "configs": written, "source_commit": upstream.commit}
    (configs / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _generate_corruptions(dataset: Path, annotations, workers: int) -> None:
    from functools import partial
    from multiprocessing import Pool

    clean = dataset / "coco/images/val2014"
    link = dataset / "coco-IP/coco_image/val2014"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(clean)
    names = sorted({row["image"].split("/")[-1] for row in annotations})
    for corruption in CORRUPTIONS:
        out = dataset / f"coco-IP/COCO_IP_{corruption}_5/val2014"
        out.mkdir(parents=True, exist_ok=True)
        with Pool(workers) as pool:
            pool.map(partial(_corrupt, clean=clean, out=out, corruption=corruption), names, chunksize=16)


def _corrupt(name, *, clean, out, corruption):
    import numpy as np
    from imagecorruptions import corrupt
    from PIL import Image

    target = out / name
    if not target.exists():
        image = np.array(Image.open(clean / name).convert("RGB"))
        Image.fromarray(corrupt(image, corruption_name=corruption, severity=5)).save(target, quality=95)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="case", required=True)
    tse = sub.add_parser("tse")
    tse.add_argument("--dataverse-url", default="https://dataverse.harvard.edu/api/access/datafile")
    tse.add_argument("--ucr-url", default="https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip")
    tse.add_argument("--ucr-password", default="someone", help="the archive's published password")
    cl = sub.add_parser("cl")
    cl.add_argument("--cifar-url", default="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz")
    cmr = sub.add_parser("cmr")
    cmr.add_argument("--upstream-root", type=Path, required=True)
    cmr.add_argument("--subset", type=int, default=1000)
    cmr.add_argument("--generate-corruptions", action="store_true")
    cmr.add_argument("--workers", type=int, default=16)
    for command in (tse, cl, cmr):
        command.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    result = {"tse": prepare_tse, "cl": prepare_cl, "cmr": prepare_cmr}[args.case](args)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
