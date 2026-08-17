import os, random, csv, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

SEED=42
GT_DIR='train/GT'; NOISY_DIR='train/NoisyLR'
CHECKPOINT_PATH='checkpoints/best_dncnn.pth'
OUT='outputs/error_analysis'
RES=os.path.join(OUT,'residuals'); COMP=os.path.join(OUT,'comparisons'); RDIR=os.path.join(OUT,'results')
for d in (OUT,RES,COMP,RDIR): os.makedirs(d,exist_ok=True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)

class KLATestDataset(Dataset):
    def __init__(self, files): self.files=files
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        f=self.files[i]
        noisy=torch.from_numpy(np.load(os.path.join(NOISY_DIR,f)).astype(np.float32)).unsqueeze(0)
        gt=torch.from_numpy(np.load(os.path.join(GT_DIR,f)).astype(np.float32)).unsqueeze(0)
        return noisy,gt,f

# EXACT architecture from evaluate_dncnn.py
class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1=nn.Conv2d(channels,channels,3,padding=1)
        self.bn1=nn.BatchNorm2d(channels)
        self.conv2=nn.Conv2d(channels,channels,3,padding=1)
        self.bn2=nn.BatchNorm2d(channels)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        residual=x
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        return self.relu(out+residual)

class DnCNNRestorer(nn.Module):
    def __init__(self,num_features=64,num_blocks=10):
        super().__init__()
        self.head=nn.Sequential(nn.Conv2d(1,num_features,3,padding=1),nn.ReLU(inplace=True))
        self.body=nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.reconstruction=nn.Conv2d(num_features,num_features,3,padding=1)
        self.upsample=nn.Sequential(nn.Conv2d(num_features,num_features*4,3,padding=1),nn.PixelShuffle(2),nn.ReLU(inplace=True))
        self.output=nn.Conv2d(num_features,1,3,padding=1)
    def forward(self,x):
        features=self.head(x)
        body=self.body(features)
        body=body+features
        body=self.reconstruction(body)
        out=self.upsample(body)
        return self.output(out)

def metrics(pred,gt):
    pred=np.clip(pred,0,1); gt=np.clip(gt,0,1)
    mse=float(np.mean((pred-gt)**2)); mae=float(np.mean(np.abs(pred-gt)))
    psnr=float(peak_signal_noise_ratio(gt,pred,data_range=1.0))
    ssim=float(structural_similarity(gt,pred,data_range=1.0))
    return mse,mae,psnr,ssim

def bicubic(noisy):
    x=torch.from_numpy(noisy).float()[None,None]
    return F.interpolate(x,size=(256,256),mode='bicubic',align_corners=False).squeeze().numpy()

# Same split as evaluate_dncnn.py
files=sorted(f for f in os.listdir(NOISY_DIR) if f.endswith('.npy') and os.path.exists(os.path.join(GT_DIR,f)))
random.Random(SEED).shuffle(files)
n=len(files); train_end=int(.8*n); val_end=int(.9*n)
test_files=files[val_end:]
loader=DataLoader(KLATestDataset(test_files),batch_size=1,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())

print('='*70); print('KLA DnCNN ERROR / RESIDUAL ANALYSIS'); print('='*70)
print('Device:',DEVICE)
if DEVICE.type=='cuda': print('GPU:',torch.cuda.get_device_name(0))
print(f'Total paired images : {n}'); print(f'Testing             : {len(test_files)}')

checkpoint=torch.load(CHECKPOINT_PATH,map_location=DEVICE,weights_only=False)
cfg=checkpoint.get('config',{})
features=cfg.get('num_features',64); blocks=cfg.get('num_blocks',10)
model=DnCNNRestorer(features,blocks).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict']); model.eval()
print(f'Features: {features} | Residual blocks: {blocks}')
print('Checkpoint loaded successfully.')

D={k:[] for k in ['b_mse','b_mae','b_psnr','b_ssim','d_mse','d_mae','d_psnr','d_ssim','time']}
all_res=[]; intens=[]; abs_err=[]
bins=[(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.)]
bin_data={f'{a:.1f}-{b:.1f}':[] for a,b in bins}
rows=[]

with torch.no_grad():
    for i,(noisy,gt,fname) in enumerate(loader):
        noisy=noisy.to(DEVICE); gt=gt.to(DEVICE)
        if DEVICE.type=='cuda': torch.cuda.synchronize()
        t0=time.perf_counter(); pred=model(noisy)
        if DEVICE.type=='cuda': torch.cuda.synchronize()
        elapsed=(time.perf_counter()-t0)*1000
        p=pred.squeeze().cpu().numpy(); g=gt.squeeze().cpu().numpy(); q=noisy.squeeze().cpu().numpy()
        dm,da,dp,ds=metrics(p,g); b=bicubic(q); bm,ba,bp,bs=metrics(b,g)
        D['d_mse'].append(dm);D['d_mae'].append(da);D['d_psnr'].append(dp);D['d_ssim'].append(ds);D['time'].append(elapsed)
        D['b_mse'].append(bm);D['b_mae'].append(ba);D['b_psnr'].append(bp);D['b_ssim'].append(bs)
        r=g-p; ae=np.abs(r); all_res.append(r.ravel())
        idx=np.random.choice(g.size,min(5000,g.size),replace=False); intens.extend(g.ravel()[idx]); abs_err.extend(ae.ravel()[idx])
        for lo,hi in bins:
            mask=(g>=lo)&((g<=hi) if hi==1 else (g<hi)); bin_data[f'{lo:.1f}-{hi:.1f}'].extend((r[mask]).tolist())
        rows.append([fname[0],bm,ba,bp,bs,dm,da,dp,ds,float(r.mean()),float(r.std()),float(ae.mean()),elapsed])
        if i<10:
            Image.fromarray((np.clip(g,0,1)*255).astype(np.uint8)).save(os.path.join(COMP,f'{i:03d}_GT.png'))
            Image.fromarray((np.clip(b,0,1)*255).astype(np.uint8)).save(os.path.join(COMP,f'{i:03d}_Bicubic.png'))
            Image.fromarray((np.clip(p,0,1)*255).astype(np.uint8)).save(os.path.join(COMP,f'{i:03d}_DnCNN.png'))
            Image.fromarray((ae/(ae.max()+1e-8)*255).astype(np.uint8)).save(os.path.join(RES,f'{i:03d}_absolute_error.png'))
        if (i+1)%50==0 or i+1==len(loader): print(f'Processed {i+1}/{len(loader)}')

# Save CSV
csv_path=os.path.join(RDIR,'error_analysis_per_image.csv')
with open(csv_path,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['filename','bicubic_mse','bicubic_mae','bicubic_psnr','bicubic_ssim','dncnn_mse','dncnn_mae','dncnn_psnr','dncnn_ssim','residual_mean','residual_std','residual_mae','inference_ms']); w.writerows(rows)

mean={k:float(np.mean(v)) for k,v in D.items()}
res=np.concatenate(all_res); x=np.asarray(intens); y=np.asarray(abs_err)
corr=float(np.corrcoef(x,y)[0,1])
print('\n'+'='*70); print('GLOBAL ERROR ANALYSIS'); print('='*70)
print(f"BICUBIC  MSE {mean['b_mse']:.8f} | MAE {mean['b_mae']:.8f} | PSNR {mean['b_psnr']:.4f} dB | SSIM {mean['b_ssim']:.6f}")
print(f"DnCNN    MSE {mean['d_mse']:.8f} | MAE {mean['d_mae']:.8f} | PSNR {mean['d_psnr']:.4f} dB | SSIM {mean['d_ssim']:.6f}")
print(f"MSE reduction : {(1-mean['d_mse']/mean['b_mse'])*100:.2f}%")
print(f"PSNR gain     : {mean['d_psnr']-mean['b_psnr']:.4f} dB")
print(f"SSIM gain     : {mean['d_ssim']-mean['b_ssim']:.6f}")
print(f"Mean inference: {mean['time']:.2f} ms/image | Median: {np.median(D['time']):.2f} ms/image")
print('\n'+'='*70); print('RESIDUAL STATISTICS'); print('='*70)
print(f'Total pixels analyzed : {len(res)}'); print(f'Residual mean         : {res.mean():.8f}'); print(f'Residual std          : {res.std():.8f}'); print(f'Residual MAE          : {np.abs(res).mean():.8f}'); print(f'Residual min          : {res.min():.8f}'); print(f'Residual max          : {res.max():.8f}')
print('\n'+'='*70); print('ERROR BY GT INTENSITY'); print('='*70); print(f"{'Range':<12}{'Pixels':>14}{'MSE':>16}{'MAE':>16}{'Std':>16}")
int_csv=os.path.join(RDIR,'intensity_error_analysis.csv')
with open(int_csv,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['intensity_range','pixels','mse','mae','residual_std','mean_residual'])
    for lo,hi in bins:
        name=f'{lo:.1f}-{hi:.1f}'; rr=np.asarray(bin_data[name],dtype=np.float64)
        if rr.size==0: continue
        mse=float(np.mean(rr**2)); mae=float(np.mean(np.abs(rr))); sd=float(np.std(rr)); mr=float(np.mean(rr))
        print(f'{name:<12}{rr.size:>14}{mse:>16.8f}{mae:>16.8f}{sd:>16.8f}'); w.writerow([name,rr.size,mse,mae,sd,mr])
print('\n'+'='*70); print('SIGNAL-DEPENDENT ERROR ANALYSIS'); print('='*70); print(f'GT intensity vs absolute error correlation: {corr:.6f}')

# Plots
plt.figure(figsize=(8,5)); plt.hist(res,bins=100); plt.xlabel('Residual (GT - DnCNN)'); plt.ylabel('Pixel Count'); plt.title('DnCNN Residual Distribution'); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(RDIR,'residual_distribution.png'),dpi=150); plt.close()
idx=np.random.choice(len(x),min(30000,len(x)),replace=False); plt.figure(figsize=(8,5)); plt.scatter(x[idx],y[idx],s=2,alpha=.2); plt.xlabel('Ground Truth Intensity'); plt.ylabel('Absolute Reconstruction Error'); plt.title('DnCNN Error vs Ground Truth Intensity'); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(RDIR,'error_vs_intensity.png'),dpi=150); plt.close()

summary=os.path.join(RDIR,'error_analysis_summary.txt')
with open(summary,'w') as f:
    f.write('KLA DnCNN ERROR / RESIDUAL ANALYSIS\n'+'='*70+'\n')
    f.write(f'Images evaluated: {len(test_files)}\nDevice: {DEVICE}\nFeatures: {features}\nResidual blocks: {blocks}\n\n')
    f.write(f'Bicubic MSE: {mean["b_mse"]:.8f}\nBicubic PSNR: {mean["b_psnr"]:.4f} dB\nBicubic SSIM: {mean["b_ssim"]:.6f}\n\n')
    f.write(f'DnCNN MSE: {mean["d_mse"]:.8f}\nDnCNN PSNR: {mean["d_psnr"]:.4f} dB\nDnCNN SSIM: {mean["d_ssim"]:.6f}\n')
    f.write(f'MSE reduction: {(1-mean["d_mse"]/mean["b_mse"])*100:.2f}%\nPSNR gain: {mean["d_psnr"]-mean["b_psnr"]:.4f} dB\nSSIM gain: {mean["d_ssim"]-mean["b_ssim"]:.6f}\n\n')
    f.write(f'Residual mean: {res.mean():.8f}\nResidual std: {res.std():.8f}\nResidual MAE: {np.abs(res).mean():.8f}\nIntensity/error correlation: {corr:.6f}\n')

print('\n'+'='*70); print('ERROR ANALYSIS COMPLETE'); print('='*70)
print('Per-image CSV :',csv_path); print('Intensity CSV :',int_csv); print('Summary       :',summary); print('Comparisons   :',COMP); print('Residual maps :',RES); print('Plots         :',RDIR); print('='*70)
