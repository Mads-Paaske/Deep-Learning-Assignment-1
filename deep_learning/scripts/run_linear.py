import numpy as np, json
from sklearn.metrics import balanced_accuracy_score
from common import *

d = np.load('/Users/mp/.medmnist/bloodmnist.npz')
np.random.seed(0)
r = np.arange(d['train_images'].shape[0]); np.random.shuffle(r)
X_train = d['train_images'][r][:5000].reshape(5000,-1)
y_train = d['train_labels'][r].flatten()[:5000]
X_val = d['val_images'][:500].reshape(500,-1);   y_val = d['val_labels'].flatten()[:500]
X_test = d['test_images'][:1000].reshape(1000,-1); y_test = d['test_labels'].flatten()[:1000]

mean_image = np.mean(X_train.astype(np.float32), axis=0)
bias = lambda A: np.hstack([A.astype(np.float32)-mean_image, np.ones((A.shape[0],1),np.float32)])
X_train, X_val, X_test = bias(X_train), bias(X_val), bias(X_test)
num_classes = 8

def svm_loss(W, X, y, reg=0):
    n = X.shape[0]; s = X.dot(W)
    ccs = s[np.arange(n), y].reshape(-1,1)
    m = np.maximum(0, s - ccs + 1); m[np.arange(n), y] = 0
    loss = np.sum(m)/n + reg*np.sum(W*W)
    b = (m > 0).astype(np.float64); b[np.arange(n), y] = -np.sum(b, axis=1)
    return loss, X.T.dot(b)/n + 2*reg*W

def softmax_loss(W, X, y, reg=0):
    n = X.shape[0]; s = X.dot(W); s -= np.max(s, axis=1, keepdims=True)
    e = np.exp(s); p = e/np.sum(e, axis=1, keepdims=True)
    loss = np.sum(-np.log(p[np.arange(n), y]))/n + reg*np.sum(W*W)
    ds = p.copy(); ds[np.arange(n), y] -= 1; ds /= n
    return loss, X.T.dot(ds) + 2*reg*W

class LinearClassifier:
    def __init__(self, input_dim, num_classes, loss_type='softmax'):
        self.W = np.random.randn(input_dim, num_classes)*0.0001; self.loss_type = loss_type
    def loss(self, Xb, yb, reg=0):
        return (svm_loss if self.loss_type=='svm' else softmax_loss)(self.W, Xb, yb, reg)
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200):
        nb = max(X.shape[0]//batch_size, 1); hist=[]
        for it in range(num_iters):
            idx = np.random.permutation(X.shape[0]); tot=0
            for b in np.array_split(idx, nb):
                l, g = self.loss(X[b], y[b], reg); self.W -= learning_rate*g; tot += l
            hist.append(tot/nb)
        return hist
    def predict(self, X): return np.argmax(X.dot(self.W), axis=1)

LRS = [1e-7, 5e-7, 1e-6]; REGS = [1e3, 1e4, 5e4]
out = {}
for lt in ['svm','softmax']:
    best, best_acc, results = None, -1, {}
    for lr in LRS:
        for reg in REGS:
            c = LinearClassifier(X_train.shape[1], num_classes, lt)
            h = c.train(X_train, y_train, lr, reg, num_iters=30, batch_size=200)
            tr = np.mean(c.predict(X_train)==y_train); va = np.mean(c.predict(X_val)==y_val)
            results[(lr,reg)] = (tr, va)
            if va > best_acc: best_acc, best, best_cfg, best_hist = va, c, (lr,reg), h
    pred = best.predict(X_test)
    ta = np.mean(pred==y_test)*100; tb = balanced_accuracy_score(y_test, pred)*100
    print(f"{lt.upper():8s} best lr={best_cfg[0]:.0e} reg={best_cfg[1]:.0e} "
          f"val={best_acc*100:.1f}% test={ta:.1f}% balanced={tb:.1f}%", flush=True)
    confusion_fig(y_test, pred, f'Linear ({lt}) on the test set', FIG+f'linear-confusion-{lt}.png')
    out[lt] = {'lr':best_cfg[0],'reg':best_cfg[1],'val':best_acc*100,'test':ta,'bal':tb,
               'hist':best_hist,'grid':{f'{k[0]:.0e}|{k[1]:.0e}':[v[0]*100,v[1]*100] for k,v in results.items()},
               'W':best.W}

# tuning grid figure
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, lt in zip(axes, ['svm','softmax']):
    for j, reg in enumerate(REGS):
        ys = [out[lt]['grid'][f'{lr:.0e}|{reg:.0e}'][1] for lr in LRS]
        ax.plot(LRS, ys, marker='o', ms=3, lw=1.2, label=f'reg={reg:.0e}')
    ax.set_xscale('log'); ax.set_xlabel('learning rate'); ax.set_title(lt, fontsize=8)
axes[0].set_ylabel('validation accuracy (%)'); axes[1].legend(frameon=False, fontsize=6)
fig.savefig(FIG+'linear-tuning.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.0, 2.8))
for lt in ['svm','softmax']: ax.plot(out[lt]['hist'], lw=1.3, label=lt)
ax.set_xlabel('epoch'); ax.set_ylabel('training loss'); ax.legend(frameon=False)
fig.savefig(FIG+'loss-curves.png'); plt.close(fig)

# learned weights, best SVM vs random
def wgrid(W, path, title):
    w = W[:-1,:].reshape(28,28,3,8); lo,hi = w.min(), w.max()
    fig, axes = plt.subplots(1, 8, figsize=(7.2, 1.25))
    for i, ax in enumerate(axes):
        ax.imshow((255.0*(w[:,:,:,i]-lo)/(hi-lo)).astype('uint8'))
        ax.axis('off'); ax.set_title(CLASSES[i], fontsize=5)
    fig.suptitle(title, fontsize=7, y=1.12); fig.savefig(path); plt.close(fig)
wgrid(out['svm']['W'], FIG+'weights-trained.png', 'Learned SVM weights, one per class')
wgrid(np.random.randn(2353,8)*0.001, FIG+'weights-random.png', 'Randomly initialized weights')

json.dump({k:{kk:vv for kk,vv in v.items() if kk!='W'} for k,v in out.items()},
          open('linear_results.json','w'), indent=1, default=str)
