import argparse
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import os

# Naive vars
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

def proc_img(path):
    try:
        img = Image.open(path).convert('L')
        img = img.resize((W, H))
        arr = np.array(img, dtype=np.float64)
        return arr.ravel(), arr
    except:
        return None, None

def run_pca(x, name):
    p = f'checkpoints/{name}.pkl'
    if not os.path.exists(p): return "No Model"
    with open(p, 'rb') as f: d = pickle.load(f)
    u, mu, th = d['u'], d['mu'], d['th']
    x = x.reshape(-1, 1)
    xc = x - mu
    z = u.T @ xc
    rec = u @ z + mu
    err = np.linalg.norm(x - rec)
    return "Face" if err < th else "Not Face"

def run_svm(x, name):
    p = f'checkpoints/{name}.pkl'
    if not os.path.exists(p): return "No Model"
    with open(p, 'rb') as f: clf = pickle.load(f)
    xn = x.reshape(1, -1) / 255.0
    pred = clf.predict(xn)
    return "Face" if pred[0] == 1 else "Not Face"

def run_cnn(x, name):
    p = f'checkpoints/{name}.pth'
    if not os.path.exists(p): return "No Model"
    d = torch.load(p, weights_only=False)
    net = Net()
    net.load_state_dict(d['state'])
    net.eval()
    th = d['th']
    
    t = torch.tensor(x.reshape(1, 1, H, W) / 255.0, dtype=torch.float32)
    with torch.no_grad():
        rec = net(t)
        diff = (t - rec).view(1, -1)
        err = torch.norm(diff).item()
    return "Face" if err < th else "Not Face"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--method', default='all')
    args = parser.parse_args()
    
    flat, _ = proc_img(args.image)
    if flat is None:
        print("Error loading image")
        return
        
    all_m = ['pca', 'pca_aug', 'svm', 'svm_aug', 'cnn']
    req = all_m if args.method == 'all' else [args.method]
    
    print(f"Inference on: {args.image}")
    print("-" * 20)
    
    for m in req:
        res = "Unknown"
        if 'pca' in m: res = run_pca(flat, m)
        elif 'svm' in m: res = run_svm(flat, m)
        elif 'cnn' in m: res = run_cnn(flat, m)
        print(f"{m.upper()}: {res}")

if __name__ == '__main__':
    main()