import numpy as np, json
from sklearn.metrics import balanced_accuracy_score
from common import *

Xtr, ytr, Xva, yva, Xte, yte = load()
print("shapes:", Xtr.shape, Xva.shape, Xte.shape, "dtype:", Xtr.dtype, flush=True)

def l2(A, B):
    a = np.sum(A**2, axis=1).reshape(-1,1); b = np.sum(B**2, axis=1).reshape(1,-1)
    return np.sqrt(np.maximum(a - 2*(A @ B.T) + b, 0))
def l1(A, B):
    out = np.empty((A.shape[0], B.shape[0]), dtype=np.float32)
    for i in range(A.shape[0]): out[i] = np.sum(np.abs(A[i] - B), axis=1)
    return out
def vote(dists, ylab, k):
    idx = np.argpartition(dists, k, axis=1)[:, :k]
    out = np.empty(dists.shape[0], dtype=int)
    for i in range(dists.shape[0]):
        order = idx[i][np.argsort(dists[i][idx[i]])]
        v, c = np.unique(ylab[order], return_counts=True); out[i] = v[np.argmax(c)]
    return out

KS = [1,3,5,7,13,21,33,45,55,67,79,91,111,131,149,167,193,201]
res = {}
for name, fn in [('L1', l1), ('L2', l2)]:
    D = fn(Xva, Xtr)
    res[name] = [ (vote(D, ytr, k) == yva).mean()*100 for k in KS ]
    print(name, "done", flush=True)

best = max(((m,k,a) for m in res for k,a in zip(KS,res[m])), key=lambda t: t[2])
bm, bk, bacc = best
print(f"BEST: {bm}, k={bk}, val={bacc:.1f}%", flush=True)

fig, ax = plt.subplots(figsize=(5.0, 3.0))
for m in res: ax.plot(KS, res[m], marker='o', ms=3, lw=1.2, label=m)
ax.axvline(bk, color='0.6', ls=':', lw=1)
ax.set_xscale('log'); ax.set_xlabel('k (number of neighbors, log scale)')
ax.set_ylabel('validation accuracy (%)'); ax.legend(frameon=False, title='distance')
fig.savefig(FIG+'knn-tuning.png'); plt.close(fig)

D_te = (l1 if bm=='L1' else l2)(Xte, Xtr)
pred = vote(D_te, ytr, bk)
test_acc = (pred == yte).mean()*100
test_bal = balanced_accuracy_score(yte, pred)*100
print(f"TEST: acc={test_acc:.1f}%  balanced={test_bal:.1f}%", flush=True)
cm = confusion_fig(yte, pred, f'kNN (k={bk}, {bm}) on the test set', FIG+'knn-confusion.png')

per = cm.diagonal() / np.maximum(cm.sum(axis=1), 1) * 100
for c, p, n in zip(CLASSES, per, cm.sum(axis=1)): print(f"  {c:16s} {p:5.1f}%  (n={n})")

json.dump({'ks':KS,'res':res,'best':{'metric':bm,'k':bk,'val':bacc},
           'test_acc':test_acc,'test_bal':test_bal,
           'per_class':dict(zip(CLASSES, per.round(1).tolist()))},
          open('knn_results.json','w'), indent=1)
