import os
import argparse
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
H, W = 112, 92
NUM_CLASSES = 40

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (H // 8) * (W // 8), 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

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

def augment(x_in, y_in):
    out_x, out_y = [], []
    for i in range(x_in.shape[1]):
        flat = x_in[:, i]
        lab = y_in[i]
        img = flat.reshape(H, W)
        pil = Image.fromarray(img)
        
        out_x.append(flat)
        out_y.append(lab)
        
        out_x.append(np.array(pil.rotate(5, resample=Image.NEAREST)).ravel())
        out_y.append(lab)
        
        out_x.append(np.array(pil.rotate(-5, resample=Image.NEAREST)).ravel())
        out_y.append(lab)
        
    return np.array(out_x).T, np.array(out_y)

def train_pca(x_tr, y_tr, x_te, y_te, name):
    mu = x_tr.mean(axis=1, keepdims=True)
    xc = x_tr - mu
    c = xc.T @ xc / (x_tr.shape[1] - 1)
    v, vec = np.linalg.eigh(c)
    idx = v.argsort()[::-1]
    vec = vec[:, idx]
    u = xc @ vec[:, :100]
    u = u / np.linalg.norm(u, axis=0)
    
    w_tr = u.T @ xc
    w_te = u.T @ (x_te - mu)
    
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(w_tr.T, y_tr)
    
    pred = knn.predict(w_te.T)
    acc = np.mean(pred == y_te) * 100
    
    with open(f'checkpoints/{name}.pkl', 'wb') as f:
        pickle.dump({'u': u, 'mu': mu, 'knn': knn}, f)
        
    return pred, acc

def train_svm(x_tr, y_tr, x_te, y_te, name):
    x_tr_n = x_tr.T / 255.0
    x_te_n = x_te.T / 255.0
    
    clf = SVC(kernel='linear', decision_function_shape='ovr')
    clf.fit(x_tr_n, y_tr)
    
    pred = clf.predict(x_te_n)
    acc = np.mean(pred == y_te) * 100
    
    with open(f'checkpoints/{name}.pkl', 'wb') as f:
        pickle.dump(clf, f)
        
    return pred, acc

def train_cnn(x_tr, y_tr, x_te, y_te, name):
    xt = torch.tensor(x_tr.T.reshape(-1, 1, H, W) / 255.0, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    xe = torch.tensor(x_te.T.reshape(-1, 1, H, W) / 255.0, dtype=torch.float32)
    ye = torch.tensor(y_te, dtype=torch.long)
    
    net = Net()
    opt = optim.Adam(net.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    losses = []
    loop = tqdm(range(100), desc=f"Training {name}")
    
    for _ in loop:
        opt.zero_grad()
        out = net(xt)
        loss = crit(out, yt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())
        
    plt.figure()
    plt.plot(losses)
    plt.title(f'{name} Training Loss')
    plt.savefig(f'figures/{name}_loss.pdf')
    plt.close()
    
    with torch.no_grad():
        logits = net(xe)
        pred = torch.argmax(logits, dim=1).numpy()
        acc = np.mean(pred == y_te) * 100
        
    torch.save(net.state_dict(), f'checkpoints/{name}.pth')
    return pred, acc

def plot(imgs, preds, true_labs, accs, keys):
    n = len(imgs)
    m = len(keys)
    cols = 10
    rows = int(np.ceil(n / cols))
    w_tot = cols * m + (m - 1)
    
    fig = plt.figure(figsize=(10 * m, 14))
    gs = fig.add_gridspec(rows, w_tot, wspace=0.0, hspace=0.3)
    eff = [path_effects.withStroke(linewidth=3, foreground='black')]
    
    for i in range(n):
        r, c = i // cols, i % cols
        for k_idx, k in enumerate(keys):
            off = k_idx * (cols + 1)
            ax = fig.add_subplot(gs[r, c + off])
            ax.imshow(imgs[i], cmap='gray')
            ax.axis('off')
            
            p_lab = preds[k][i]
            t_lab = true_labs[i]
            ok = p_lab == t_lab
            
            ax.text(0.05, 0.85, '✅' if ok else '❌', color='lime' if ok else 'red',
                    fontsize=18, weight='bold', transform=ax.transAxes, path_effects=eff)
            
            ax.text(0.5, -0.15, f"s{p_lab+1}", color='blue' if ok else 'red',
                    fontsize=10, weight='bold', ha='center', transform=ax.transAxes)

    for k_idx, k in enumerate(keys):
        pos = (k_idx * (cols + 1) + cols/2) / w_tot
        plt.figtext(pos, 0.92, f"{k.upper()}\nAcc: {accs[k]:.2f}%", ha='center', fontsize=12, weight='bold')
        
    plt.savefig('results_classification.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['all'])
    args = parser.parse_args()
    
    all_m = ['pca', 'pca_aug', 'svm', 'svm_aug', 'cnn', 'cnn_aug']
    req = all_m if 'all' in args.methods else args.methods
    
    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
    if not os.path.exists('figures'): os.makedirs('figures')
    
    paths, labs = get_data('faces')
    if not paths: return
    
    tr_idx, te_idx = [], []
    te_paths = []
    
    for s in range(40):
        loc = np.where(labs == s)[0]
        if len(loc) == 0: continue
        
        split_point = 8 
        tr_idx.extend(loc[:split_point])
        te_idx.extend(loc[split_point:])
        te_paths.extend([paths[i] for i in loc[split_point:]])
        
    x = prep_data(paths)
    xtr = x[:, tr_idx]
    ytr = labs[tr_idx]
    xte = x[:, te_idx]
    yte = labs[te_idx]
    
    xtr_aug, ytr_aug = augment(xtr, ytr)
    
    res, accs = {}, {}
    
    if 'pca' in req:
        res['pca'], accs['pca'] = train_pca(xtr, ytr, xte, yte, 'pca')
        print(f"PCA: {accs['pca']:.2f}%")
        
    if 'pca_aug' in req:
        res['pca_aug'], accs['pca_aug'] = train_pca(xtr_aug, ytr_aug, xte, yte, 'pca_aug')
        print(f"PCA Aug: {accs['pca_aug']:.2f}%")
        
    if 'svm' in req:
        res['svm'], accs['svm'] = train_svm(xtr, ytr, xte, yte, 'svm')
        print(f"SVM: {accs['svm']:.2f}%")
        
    if 'svm_aug' in req:
        res['svm_aug'], accs['svm_aug'] = train_svm(xtr_aug, ytr_aug, xte, yte, 'svm_aug')
        print(f"SVM Aug: {accs['svm_aug']:.2f}%")
        
    if 'cnn' in req:
        res['cnn'], accs['cnn'] = train_cnn(xtr, ytr, xte, yte, 'cnn')
        print(f"CNN: {accs['cnn']:.2f}%")

    if 'cnn_aug' in req:
        res['cnn_aug'], accs['cnn_aug'] = train_cnn(xtr_aug, ytr_aug, xte, yte, 'cnn_aug')
        print(f"CNN Aug: {accs['cnn_aug']:.2f}%")
        
    te_imgs = [np.array(Image.open(p).convert('L')) for p in te_paths]
    plot(te_imgs, res, yte, accs, req)

if __name__ == '__main__':
    main()