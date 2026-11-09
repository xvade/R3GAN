"""Score PNGs with the R3GAN discriminator. Expects <indir>/<class_name>/*.png (as written by gen_grid.py).
Prints mean D logit per class with the correct label, and with a wrong (shifted) label as a control."""
import argparse, glob, os
import numpy as np, PIL.Image, torch
import dnnlib, legacy, ref_ops  # noqa: F401

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
p = argparse.ArgumentParser()
p.add_argument('--network', required=True)
p.add_argument('--indir', required=True)
a = p.parse_args()

dev = torch.device('cuda')
with dnnlib.util.open_url(a.network) as f:
    D = legacy.load_network_pkl(f)['D'].to(dev).eval()

def score(x, cls):
    c = torch.nn.functional.one_hot(torch.full((len(x),), cls), len(CLASSES)).float().to(dev)
    with torch.no_grad():
        return D(x, c).float().cpu().numpy()

print(f'{"class":<12}{"n":>4}{"D(correct)":>12}{"D(wrong)":>12}')
for i, name in enumerate(CLASSES):
    files = sorted(glob.glob(os.path.join(a.indir, name, '*.png')))
    if not files: continue
    x = torch.from_numpy(np.stack([np.array(PIL.Image.open(f).convert('RGB')) for f in files]))
    x = (x.permute(0, 3, 1, 2).float() / 127.5 - 1).to(dev)
    print(f'{name:<12}{len(files):>4}{score(x, i).mean():>12.3f}{score(x, (i + 5) % 10).mean():>12.3f}')
