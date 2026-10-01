"""Export ImageNet-100 (Tian et al., 2020) from Hugging Face to ImageFolder, short side 160 px."""
import argparse
import os

from datasets import load_dataset
from PIL import Image


def export(split, out_dir, short_side):
    ds = load_dataset('clane9/imagenet-100', split=split)
    for i, ex in enumerate(ds):
        img = ex['image'].convert('RGB')
        w, h = img.size
        s = short_side / min(w, h)
        if s < 1:
            img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
        d = os.path.join(out_dir, str(ex['label']))
        os.makedirs(d, exist_ok=True)
        img.save(os.path.join(d, f'{i:08d}.jpg'), quality=95)
    print(f'{split}: {len(ds)} images -> {out_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default='data/imagenet100')
    ap.add_argument('--short_side', type=int, default=160)
    args = ap.parse_args()
    export('train', os.path.join(args.out, 'train'), args.short_side)
    export('validation', os.path.join(args.out, 'val'), args.short_side)


if __name__ == '__main__':
    main()
