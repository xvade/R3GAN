"""Generate a class-conditional sample grid (rows = CIFAR-10 classes) from an R3GAN pickle."""
import argparse, os
import numpy as np, PIL.Image, torch
import dnnlib, legacy, ref_ops

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

p = argparse.ArgumentParser()
p.add_argument('--network', required=True)
p.add_argument('--outdir', required=True)
p.add_argument('--per-class', type=int, default=10)
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()

dev = torch.device('cuda')
with dnnlib.util.open_url(a.network) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(dev).eval()
os.makedirs(a.outdir, exist_ok=True)

rows = []
with torch.no_grad():
    for c in range(G.c_dim or 1):
        z = torch.from_numpy(np.random.RandomState(a.seed + c).randn(a.per_class, G.z_dim)).to(dev)
        lbl = torch.zeros([a.per_class, G.c_dim], device=dev)
        if G.c_dim: lbl[:, c] = 1
        img = (G(z, lbl).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        for i, im in enumerate(img):
            d = os.path.join(a.outdir, CLASSES[c] if G.c_dim else 'uncond'); os.makedirs(d, exist_ok=True)
            PIL.Image.fromarray(im, 'RGB').save(f'{d}/{i:03d}.png')
        rows.append(np.concatenate(img, axis=1))
PIL.Image.fromarray(np.concatenate(rows, axis=0), 'RGB').save(f'{a.outdir}/grid.png')
print('wrote', f'{a.outdir}/grid.png')
