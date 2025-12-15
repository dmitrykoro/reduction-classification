import os
import argparse
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
H, W = 112, 92

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.dec = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.dec(self.enc(x))

def get_data(path):
    paths, labs = [], []
    if os.path.exists(path):
        dirs = sorted([d for d in os.listdir(path) if d.startswith('s')], key=lambda x: int(x[1:]))
        for d in dirs:
            sid = int(d[1:]) - 1
            p = os.path.join(path, d)
            fs = sorted([f for f in os.listdir(p) if f.endswith('.pgm')])
            for f in fs:
                paths.append(os.path.join(p, f))
                labs.append(sid)
    return paths, np.array(labs)

def prep_data(paths):
    x = np.zeros((H * W, len(paths)))
    for i, p in enumerate(paths):
        img = Image.open(p).convert('L')
        x[:, i] = np.array(img).ravel()
    return x

"""
For data augmentation:
Rotate the immages on different axis. (1--->3)
"""
def augment(x_in):

    out = []
    for i in range(x_in.shape[1]):
        flat = x_in[:, i]
        img = flat.reshape(H, W)
        pil = Image.fromarray(img)
        out.append(flat)
        out.append(np.array(pil.rotate(5, resample=Image.NEAREST)).ravel())
        out.append(np.array(pil.rotate(-5, resample=Image.NEAREST)).ravel())
    return np.array(out).T

def train_pca(x_tr, x_te, name):
    mu = x_tr.mean(axis=1, keepdims=True)
    xc = x_tr - mu
    c = xc.T @ xc / (x_tr.shape[1] - 1)
    v, vec = np.linalg.eigh(c)
    idx = v.argsort()[::-1]
    vec = vec[:, idx]
    u = xc @ vec[:, :30]
    u = u / np.linalg.norm(u, axis=0)
    
    # Plot Loss
    plt.figure()
    plt.plot(v[idx[:50]]) # Top 50
    plt.title(f'{name} Eigenvalues (Variance)')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.savefig(f'figures/{name}_loss.pdf')
    plt.close()
    
    # Thresh
    z = u.T @ xc
    rec = u @ z + mu
    err = np.linalg.norm(x_tr - rec, axis=0)
    th = np.mean(err) + 2 * np.std(err)
    
    # Save_checkpoint for inferencing
    with open(f'checkpoints/{name}.pkl', 'wb') as f:
        pickle.dump({'u': u, 'mu': mu, 'th': th}, f)
        
    # Test on test set
    xc_te = x_te - mu
    z_te = u.T @ xc_te
    rec_te = u @ z_te + mu
    err_te = np.linalg.norm(x_te - rec_te, axis=0)
    corr = err_te < th
    return corr, np.mean(corr) * 100

# SVM model
def train_svm(x_tr, x_te, name):
    x_tr_n = x_tr.T / 255.0
    x_te_n = x_te.T / 255.0
    clf = OneClassSVM(kernel='linear', nu=0.1).fit(x_tr_n)
    
    # Plot Decision Function Scores
    scores = clf.decision_function(x_tr_n)
    plt.figure()
    plt.hist(scores, bins=50)
    plt.title(f'{name} Training Scores Distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.savefig(f'figures/{name}_loss.pdf')
    plt.close()
    
    with open(f'checkpoints/{name}.pkl', 'wb') as f:
        pickle.dump(clf, f)
        
    pred = clf.predict(x_te_n)
    corr = pred == 1
    return corr, np.mean(corr) * 100

"""
CNN training configs
net: model predifned earlier
opt: optimizer used
crit: loss function MSE
epoch: 100
lr: 0.002
"""
def train_cnn(x_tr, x_te, name):

    xt = torch.tensor(x_tr.T.reshape(-1, 1, H, W) / 255.0, dtype=torch.float32)
    xe = torch.tensor(x_te.T.reshape(-1, 1, H, W) / 255.0, dtype=torch.float32)
    
    net = Net()
    opt = optim.Adam(net.parameters(), lr=0.002)
    crit = nn.MSELoss()
    
    losses = []
    loop = tqdm(range(100), desc=f"Training {name}")
    
    for _ in loop:
        opt.zero_grad()
        out = net(xt)
        loss = crit(out, xt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())
        
    # Plot training loss
    plt.figure()
    plt.plot(losses)
    plt.title(f'{name} Training Loss')
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.savefig(f'figures/{name}_loss.pdf')
    plt.close()
        
    with torch.no_grad():
        rec_tr = net(xt)
        diff = (xt - rec_tr).view(xt.size(0), -1)
        err = torch.norm(diff, dim=1).numpy()
        th = np.mean(err) + 2 * np.std(err)
        
        rec_te = net(xe)
        diff_te = (xe - rec_te).view(xe.size(0), -1)
        err_te = torch.norm(diff_te, dim=1).numpy()
        corr = err_te < th
        
    torch.save({'state': net.state_dict(), 'th': th}, f'checkpoints/{name}.pth')
    return corr, np.mean(corr) * 100

#Visualize the images after training
def plot(imgs, res, accs, keys):
    n = len(imgs)
    m = len(keys)
    cols = 10
    rows = int(np.ceil(n / cols))
    w_tot = cols * m + (m - 1)
    
    fig = plt.figure(figsize=(10 * m, 14))
    gs = fig.add_gridspec(rows, w_tot, wspace=0.0, hspace=0.05)
    eff = [path_effects.withStroke(linewidth=3, foreground='black')]
    
    for i in range(n):
        r, c = i // cols, i % cols
        for k_idx, k in enumerate(keys):
            off = k_idx * (cols + 1)
            ax = fig.add_subplot(gs[r, c + off])
            ax.imshow(imgs[i], cmap='gray')
            ax.axis('off')
            ok = res[k][i]
            ax.text(0.05, 0.85, '✅' if ok else '❌', color='lime' if ok else 'red',
                    fontsize=18, weight='bold', transform=ax.transAxes, path_effects=eff) #red wrong, green correcft
            
    for k_idx, k in enumerate(keys):
        pos = (k_idx * (cols + 1) + cols/2) / w_tot
        plt.figtext(pos, 0.90, f"{k.upper()}\nAcc: {accs[k]:.2f}%", ha='center', fontsize=12, weight='bold')
        
    plt.savefig('results.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['all'])
    args = parser.parse_args()
    
    all_m = ['pca', 'pca_aug', 'svm', 'svm_aug', 'cnn']
    req = all_m if 'all' in args.methods else args.methods
    
    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
    if not os.path.exists('figures'): os.makedirs('figures')
    
    paths, labs = get_data('faces')
    if not paths: return
    
    tr_idx, te_idx, te_paths = [], [], []
    for s in range(40):
        loc = np.where(labs == s)[0]
        if len(loc) == 0: continue
        lim = 8 if s < 35 else len(loc)
        tr_idx.extend(loc[:8] if s < 35 else [])
        te_idx.extend(loc[8:] if s < 35 else loc)
        te_paths.extend([paths[i] for i in (loc[8:] if s < 35 else loc)])
        
    x = prep_data(paths)
    xtr = x[:, tr_idx]
    xte = x[:, te_idx]
    xtr_aug = augment(xtr)
    
    res, accs = {}, {}
    
    if 'pca' in req:
        res['pca'], accs['pca'] = train_pca(xtr, xte, 'pca')
        print(f"PCA: {accs['pca']:.2f}%")
        
    if 'pca_aug' in req:
        res['pca_aug'], accs['pca_aug'] = train_pca(xtr_aug, xte, 'pca_aug')
        print(f"PCA Aug: {accs['pca_aug']:.2f}%")
        
    if 'svm' in req:
        res['svm'], accs['svm'] = train_svm(xtr, xte, 'svm')
        print(f"SVM: {accs['svm']:.2f}%")
        
    if 'svm_aug' in req:
        res['svm_aug'], accs['svm_aug'] = train_svm(xtr_aug, xte, 'svm_aug')
        print(f"SVM Aug: {accs['svm_aug']:.2f}%")
        
    if 'cnn' in req:
        res['cnn'], accs['cnn'] = train_cnn(xtr_aug, xte, 'cnn')
        print(f"CNN: {accs['cnn']:.2f}%")
        
    te_imgs = [np.array(Image.open(p).convert('L')) for p in te_paths]
    plot(te_imgs, res, accs, req)

if __name__ == '__main__':
    main()