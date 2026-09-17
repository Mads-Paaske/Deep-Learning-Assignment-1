import numpy as np, random, json
from common import *
d = np.load('/Users/mp/.medmnist/bloodmnist.npz')
ti, tl = d['train_images'], d['train_labels'].flatten()
random.seed(42)
fig, axes = plt.subplots(5, 8, figsize=(7.2, 4.8))
for c in range(8):
    idx = random.sample([i for i, l in enumerate(tl) if l == c], 5)
    for j, ix in enumerate(idx):
        axes[j, c].imshow(ti[ix]); axes[j, c].axis('off')
        if j == 0: axes[j, c].set_title(CLASSES[c], fontsize=6)
fig.tight_layout(); fig.savefig(FIG+'class-examples.png'); plt.close(fig)

# per-split class distribution, which the report is missing
rows = {}
for sp in ['train', 'val', 'test']:
    lab = d[f'{sp}_labels'].flatten()
    rows[sp] = [int((lab == c).sum()) for c in range(8)]
json.dump(rows, open('class_dist.json', 'w'), indent=1)
print(f"{'class':16s} {'train':>7s} {'val':>6s} {'test':>6s}")
for c in range(8):
    print(f"{CLASSES[c]:16s} {rows['train'][c]:7d} {rows['val'][c]:6d} {rows['test'][c]:6d}")
print(f"{'TOTAL':16s} {sum(rows['train']):7d} {sum(rows['val']):6d} {sum(rows['test']):6d}")
