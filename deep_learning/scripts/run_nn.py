import numpy as np, json, time
from sklearn.metrics import balanced_accuracy_score
from common import *

d = np.load('/Users/mp/.medmnist/bloodmnist.npz')
np.random.seed(0)
r = np.arange(d['train_images'].shape[0]); np.random.shuffle(r)
X_train = d['train_images'][r][:5000].reshape(5000,-1)
y_train = d['train_labels'][r].flatten()[:5000]
X_val  = d['val_images'][:500].reshape(500,-1);    y_val  = d['val_labels'].flatten()[:500]
X_test = d['test_images'][:1000].reshape(1000,-1); y_test = d['test_labels'].flatten()[:1000]
mean_image = np.mean(X_train.astype(np.float32), axis=0)
prep = lambda A: (A.astype(np.float32) - mean_image) / 255.
X_train, X_val, X_test = prep(X_train), prep(X_val), prep(X_test)

class FullyConnectedNN:
    def __init__(self, layers, reg_strength=0.01, loss='softmax', seed=42):
        np.random.seed(seed); self.layers=layers; self.reg_strength=reg_strength; self.loss_type=loss
        self.params={}
        for i in range(1,len(layers)):
            self.params['W'+str(i)]=np.random.randn(layers[i-1],layers[i])*0.01
            self.params['b'+str(i)]=np.zeros((1,layers[i]))
    def relu(self,Z): return np.maximum(0,Z)
    def relu_derivative(self,Z): return Z>0
    def softmax(self,Z):
        e=np.exp(Z-np.max(Z,axis=1,keepdims=True)); return e/np.sum(e,axis=1,keepdims=True)
    def forward(self,X):
        cache={'A0':X}; A=X
        for i in range(1,len(self.layers)):
            Z=np.dot(A,self.params['W'+str(i)])+self.params['b'+str(i)]
            A=self.relu(Z) if i!=len(self.layers)-1 else Z
            cache['Z'+str(i)]=Z; cache['A'+str(i)]=A
        return A,cache
    def loss_value(self,A,y):
        p=self.softmax(A); n=y.shape[0]
        reg=sum(np.sum(self.params['W'+str(i)]**2) for i in range(1,len(self.layers)))
        return np.sum(-np.log(np.maximum(p[np.arange(n),y],1e-12)))/n + self.reg_strength*reg/2
    def backward(self,cache,y):
        grads={}; m=y.shape[0]
        dA=self.softmax(cache['A'+str(len(self.layers)-1)]); dA[range(m),y]-=1; dA/=m
        for i in reversed(range(1,len(self.layers))):
            dZ=dA; A_prev=cache['A'+str(i-1)]
            grads['W'+str(i)]=np.dot(A_prev.T,dZ)+self.reg_strength*self.params['W'+str(i)]
            grads['b'+str(i)]=np.sum(dZ,axis=0,keepdims=True)
            if i>1: dA=np.dot(dZ,self.params['W'+str(i)].T)*self.relu_derivative(cache['Z'+str(i-1)])
        return grads
    def fit(self,X,y,epochs=100,batch_size=64,learning_rate=0.01):
        hist=[]
        for ep in range(epochs):
            p=np.random.permutation(X.shape[0]); Xs,ys=X[p],y[p]; tot=0;nb=0
            for i in range(0,X.shape[0],batch_size):
                Xb,yb=Xs[i:i+batch_size],ys[i:i+batch_size]
                A,cache=self.forward(Xb); tot+=self.loss_value(A,yb); nb+=1
                g=self.backward(cache,yb)
                for j in range(1,len(self.layers)):
                    self.params['W'+str(j)]-=learning_rate*g['W'+str(j)]
                    self.params['b'+str(j)]-=learning_rate*g['b'+str(j)]
            hist.append(tot/nb)
        return hist
    def predict(self,X): return np.argmax(self.forward(X)[0],axis=1)

HID=[200,500]; LRS=[0.01,0.05,0.1]; EPOCHS=60
best=None; best_va=-1; grid={}
for h in HID:
    for lr in LRS:
        t=time.time()
        net=FullyConnectedNN([X_train.shape[1],h,8], reg_strength=0.001, loss='softmax')
        hist=net.fit(X_train,y_train,epochs=EPOCHS,batch_size=64,learning_rate=lr)
        tr=np.mean(net.predict(X_train)==y_train)*100; va=np.mean(net.predict(X_val)==y_val)*100
        grid[(h,lr)]=(tr,va)
        print(f"hidden={h:4d} lr={lr:<5} train={tr:5.1f}% val={va:5.1f}%  ({time.time()-t:.0f}s)",flush=True)
        if va>best_va: best_va,best,best_cfg,best_hist=va,net,(h,lr),hist

pred=best.predict(X_test)
ta=np.mean(pred==y_test)*100; tb=balanced_accuracy_score(y_test,pred)*100
print(f"\nBEST hidden={best_cfg[0]} lr={best_cfg[1]} val={best_va:.1f}% test={ta:.1f}% balanced={tb:.1f}%",flush=True)
confusion_fig(y_test,pred,f'Neural network ({best_cfg[0]} hidden, lr={best_cfg[1]}) on the test set',FIG+'nn-confusion.png')

fig,ax=plt.subplots(figsize=(5.0,2.9))
for h in HID:
    ax.plot(LRS,[grid[(h,lr)][1] for lr in LRS],marker='o',ms=3,lw=1.2,label=f'{h} hidden nodes')
ax.set_xscale('log'); ax.set_xlabel('learning rate'); ax.set_ylabel('validation accuracy (%)')
ax.legend(frameon=False); fig.savefig(FIG+'nn-tuning.png'); plt.close(fig)

fig,ax=plt.subplots(figsize=(5.0,2.8))
ax.plot(best_hist,lw=1.3); ax.set_xlabel('epoch'); ax.set_ylabel('training loss')
fig.savefig(FIG+'nn-loss.png'); plt.close(fig)

W1=best.params['W1']; w=W1.reshape(28,28,3,-1); lo,hi=w.min(),w.max()
fig,axes=plt.subplots(2,8,figsize=(7.2,2.0))
for i,ax in enumerate(axes.flat):
    ax.imshow((255.0*(w[:,:,:,i]-lo)/(hi-lo)).astype('uint8')); ax.axis('off')
fig.suptitle('First-layer weights, 16 of the hidden nodes',fontsize=7,y=1.04)
fig.savefig(FIG+'nn-weights.png'); plt.close(fig)

json.dump({'grid':{f'{k[0]}|{k[1]}':list(v) for k,v in grid.items()},
           'best':{'hidden':best_cfg[0],'lr':best_cfg[1],'val':best_va,'test':ta,'bal':tb},
           'train_acc':grid[best_cfg][0]}, open('nn_results.json','w'), indent=1)
