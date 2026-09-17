import numpy as np
from common import *
Xtr, ytr, Xva, yva, _, _ = load()
d = np.load('/Users/mp/.medmnist/bloodmnist.npz')
np.random.seed(0)
r = np.arange(d['train_images'].shape[0]); np.random.shuffle(r)
tr_img = d['train_images'][r][:5000]; va_img = d['val_images'][:500]

def l1(a, B): return np.sum(np.abs(a - B), axis=1)
rows = [3, 11, 27, 44, 61]
fig, axes = plt.subplots(len(rows), 6, figsize=(5.4, 4.6))
for ri, i in enumerate(rows):
    near = np.argsort(l1(Xva[i], Xtr))[:5]
    axes[ri,0].imshow(va_img[i]); axes[ri,0].axis('off')
    axes[ri,0].set_ylabel(CLASSES[yva[i]], fontsize=5)
    axes[ri,0].set_title(f'query: {CLASSES[yva[i]]}', fontsize=5)
    for j, n in enumerate(near):
        axes[ri,j+1].imshow(tr_img[n]); axes[ri,j+1].axis('off')
        ok = '=' if ytr[n] == yva[i] else 'x'
        axes[ri,j+1].set_title(f'{ok} {CLASSES[ytr[n]]}', fontsize=5)
fig.suptitle('Validation image (left) and its 5 nearest training images, L1 distance', fontsize=7)
fig.tight_layout(); fig.savefig(FIG+'knn-neighbours.png'); plt.close(fig)
print("saved knn-neighbours.png")
