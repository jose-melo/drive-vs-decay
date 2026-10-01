from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CIFAR10_ZENODO_URL = (
    "https://zenodo.org/records/10089977/files/cifar-10-python.tar.gz?download=1"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="Directory passed as root_path to the repo dataset helpers.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["cifar10", "cifar100", "stl10"],
        default=["cifar10", "stl10"],
        help="Datasets to download using torchvision(download=True).",
    )
    return parser.parse_args()

def _build_transform(mean: tuple[float, float, float], std: tuple[float, float, float]):
    import torchvision

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])

def _download_cifar10(root: str) -> None:
    from torchvision.datasets import CIFAR10

    from src.datasets.cifar10 import (
        CIFAR10_MEAN,
        CIFAR10_STD,
        make_cifar10,
        make_cifar10_linear_probe,
    )

    CIFAR10.url = CIFAR10_ZENODO_URL
    transform = _build_transform(CIFAR10_MEAN, CIFAR10_STD)
    make_cifar10(
        transform=transform,
        batch_size=1,
        num_workers=0,
        root_path=root,
        training=True,
        drop_last=False,
    )
    make_cifar10_linear_probe(
        batch_size=1,
        num_workers=0,
        root_path=root,
    )

def _download_cifar100(root: str) -> None:
    from src.datasets.cifar100 import make_cifar100, make_cifar100_linear_probe

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform = _build_transform(mean, std)
    make_cifar100(
        transform=transform,
        batch_size=1,
        num_workers=0,
        root_path=root,
        training=True,
        drop_last=False,
    )
    make_cifar100_linear_probe(
        batch_size=1,
        num_workers=0,
        root_path=root,
    )

def _download_stl10(root: str) -> None:
    from src.datasets.stl10 import (
        STL10_MEAN,
        STL10_STD,
        make_stl10,
        make_stl10_linear_probe,
    )

    transform = _build_transform(STL10_MEAN, STL10_STD)
    make_stl10(
        transform=transform,
        batch_size=1,
        num_workers=0,
        root_path=root,
        training=True,
        drop_last=False,
    )
    make_stl10_linear_probe(
        batch_size=1,
        num_workers=0,
        root_path=root,
    )

DOWNLOADERS = {
    "cifar10": _download_cifar10,
    "cifar100": _download_cifar100,
    "stl10": _download_stl10,
}

def main() -> int:
    args = parse_args()
    if not args.root.is_absolute():
        args.root = (REPO_ROOT / args.root).resolve()
    args.root.mkdir(parents=True, exist_ok=True)

    try:
        import torchvision
    except ModuleNotFoundError:
        return 2

    for dataset in args.datasets:
        DOWNLOADERS[dataset](str(args.root))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
