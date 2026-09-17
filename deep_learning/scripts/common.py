import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = '/Users/mp/Documents/deep_learning/figures/'
CLASSES = ['basophil','eosinophil','erythroblast','immature gran.',
           'lymphocyte','monocyte','neutrophil','platelet']

plt.rcParams.update({
    'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'figure.dpi': 150, 'savefig.bbox': 'tight',
})

def load(n_train=5000, n_val=500, n_test=1000, as_float=True):
    d = np.load('/Users/mp/.medmnist/bloodmnist.npz')
    np.random.seed(0)
    r = np.arange(d['train_images'].shape[0]); np.random.shuffle(r)
    Xtr = d['train_images'][r][:n_train]; ytr = d['train_labels'][r].flatten()[:n_train]
    Xva = d['val_images'][:n_val];        yva = d['val_labels'].flatten()[:n_val]
    Xte = d['test_images'][:n_test];      yte = d['test_labels'].flatten()[:n_test]
    flat = lambda a: a.reshape(a.shape[0], -1).astype(np.float32 if as_float else a.dtype)
    return flat(Xtr), ytr, flat(Xva), yva, flat(Xte), yte

def confusion_fig(y_true, y_pred, title, path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(8))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cmn, cmap='Greys', vmin=0, vmax=1)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(CLASSES, fontsize=6)
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(title, fontsize=8)
    ax.grid(False)
    for i in range(8):
        for j in range(8):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=5.5,
                        color='white' if cmn[i, j] > 0.5 else 'black')
    fig.savefig(path); plt.close(fig)
    return cm
