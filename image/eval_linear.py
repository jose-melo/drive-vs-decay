"""Linear probe of a saved checkpoint, same protocol as the in-loop probe."""

import argparse
import glob
import json
import os

import torch
import yaml

from src.helper import init_model
from src.mu_utils import linear_probe
from src.datasets.cifar100 import make_cifar100_linear_probe
from src.datasets.cifar10 import make_cifar10_linear_probe
from src.datasets.stl10 import make_stl10_linear_probe
from src.datasets.imagenet100 import make_imagenet100_linear_probe

_PROBE_MAKERS = {
    'cifar100': make_cifar100_linear_probe,
    'cifar10': make_cifar10_linear_probe,
    'stl10': make_stl10_linear_probe,
    'imagenet100': make_imagenet100_linear_probe,
    'imagenet1k': make_imagenet100_linear_probe,  # same loader as ImageNet-100
}


def main():
    torch.manual_seed(0)
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--root_path', default=None)
    ap.add_argument('--probe_epochs', type=int, default=None,
                    help='default: the evaluation block of the run params.yaml, '
                         'as in the in-loop probe')
    ap.add_argument('--probe_lr', type=float, default=None)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    folder = os.path.dirname(os.path.abspath(args.checkpoint))
    params_files = glob.glob(os.path.join(folder, '*_params.yaml'))
    assert len(params_files) == 1, f'expected 1 params.yaml in {folder}, got {params_files}'
    with open(params_files[0]) as f:
        params = yaml.safe_load(f)

    model_name = params['meta']['model_name']
    pred_depth = params['meta']['pred_depth']
    pred_emb_dim = params['meta']['pred_emb_dim']
    crop_size = params['data']['crop_size']
    patch_size = params['mask']['patch_size']
    dataset = params['data'].get('dataset', 'cifar100')
    eval_cfg = params.get('evaluation', {})
    num_classes = eval_cfg.get('num_classes', 1000)
    probe_epochs = (args.probe_epochs if args.probe_epochs is not None
                    else eval_cfg.get('linear_probe_epochs', 20))
    probe_lr = (args.probe_lr if args.probe_lr is not None
                else eval_cfg.get('linear_probe_lr', 0.1))
    root_path = args.root_path or params['data']['root_path']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder, _ = init_model(
        device=device, patch_size=patch_size, crop_size=crop_size,
        pred_depth=pred_depth, pred_emb_dim=pred_emb_dim,
        model_name=model_name, predictor_type='standard')

    ck = torch.load(args.checkpoint, map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in ck['target_encoder'].items()}
    msg = encoder.load_state_dict(state)
    print(f'loaded target_encoder from {args.checkpoint} (epoch {ck.get("epoch")}): {msg}')

    make_probe = _PROBE_MAKERS[dataset]
    train_loader, test_loader = make_probe(
        batch_size=256, num_workers=args.num_workers, root_path=root_path,
        crop_size=crop_size)

    test_acc, train_acc = linear_probe(
        encoder=encoder, train_loader=train_loader, test_loader=test_loader,
        device=device, num_classes=num_classes, num_epochs=probe_epochs,
        lr=probe_lr)

    result = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'epoch': ck.get('epoch'),
        'test_acc': test_acc,
        'train_acc': train_acc,
        'protocol': 'in-loop-identical (extract_features + standardize + '
                    f'SGD-cosine, epochs={probe_epochs}, lr={probe_lr})',
    }
    out = args.out or args.checkpoint.replace('.pth.tar', '_offline_probe.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
