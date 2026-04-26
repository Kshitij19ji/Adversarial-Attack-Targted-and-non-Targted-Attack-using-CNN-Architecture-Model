import torch

def proj_lp(v, epsilon, p=np.inf):
    if p == np.inf:
        return torch.clamp(v, -epsilon, epsilon)
    else:
        raise ValueError("Only L∞ norm is supported.")

def targeted_uap_attack(model, dataloader, target_class, epsilon=0.02, alpha=0.005, num_iter=20, device='cuda'):
    model.eval()
    v = torch.zeros(1, 3, 224, 224).to(device)

    for _ in range(num_iter):
        for x, _ in dataloader:
            x = x.to(device)
            x_denorm = denormalize(x)               
            x_adv = (x_denorm + v).clamp(0, 1).detach().clone().requires_grad_(True)

            y_targ = torch.full((x.size(0),), target_class, dtype=torch.long, device=device)
            logits = model(normalize(x_adv))        
            loss = F.cross_entropy(logits, y_targ)
            loss.backward()

            with torch.no_grad():
                grad = x_adv.grad.sign().mean(dim=0, keepdim=True)
                v = (v - alpha * grad).detach()
                v = proj_lp(v, epsilon, p=np.inf)

    return v.detach()

def plot_targeted_uap(X_sel_norm, v, y_true, y_pred, y_targ, title="Targeted UAP", M=2, N=5):
    fig, ax = plt.subplots(M, N, figsize=(12, 6))
    for i in range(M):
        for j in range(N):
            k = i * N + j
            if k >= X_sel_norm.size(0): break

            img = denormalize(X_sel_norm[k].unsqueeze(0)).squeeze(0) + v.squeeze(0)
            img = img.clamp(0, 1).cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img = np.clip(img, 0, 1)

            ax[i][j].imshow(img)
            t, p, tgt = y_true[k].item(), y_pred[k].item(), y_targ[k].item()
            color = 'g' if p == tgt else 'r'
            ax[i][j].set_title(f"True: {t} → Pred: {p}", color=color, fontsize=10)
            ax[i][j].axis('off')

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean


orig_dir   = "/data1/kshitij/PGD/Universal/UniversalOriginalimage"
attack_dir = "/data1/kshitij/PGD/Universal/UniversalAttackimage"
os.makedirs(orig_dir, exist_ok=True)
os.makedirs(attack_dir, exist_ok=True)


true_lbl   = 184
target_lbl = 0
k_imgs     = 10
epsilon    = 0.03
alpha      = 0.005
num_iter   = 100


X_sel, y_sel = [], []
for xb, yb in test_loader:
    m = yb == true_lbl
    if m.any():
        X_sel.append(xb[m])
        y_sel.append(yb[m])
    if sum(x.size(0) for x in X_sel) >= k_imgs:
        break

X_sel = torch.cat(X_sel)[:k_imgs]
y_sel = torch.cat(y_sel)[:k_imgs]
y_targ = torch.full_like(y_sel, target_lbl)


subset = TensorDataset(X_sel, y_sel)
loader_subset = DataLoader(subset, batch_size=2, shuffle=False)


v = targeted_uap_attack(
    model_vgg19, loader_subset,
    target_class=target_lbl,
    epsilon=epsilon, alpha=alpha,
    num_iter=num_iter, device=device
)

X_sel = X_sel.to(device)
X_adv = denormalize(X_sel) + v
X_adv = X_adv.clamp(0, 1)


logits_adv = model_vgg19(normalize(X_adv))
preds_adv = logits_adv.argmax(1)


plot_targeted_uap(X_sel, v, y_sel.cpu(), preds_adv.cpu(), y_targ.cpu(),
                  title=f"VGG19 UAP (True: {true_lbl} → Target: {target_lbl})")


success = (preds_adv == target_lbl).float().mean().item()
print(f" Targeted UAP Success Rate (VGG19): {success * 100:.1f}%")


X_sel_cpu = X_sel.detach().cpu()
X_adv_cpu = X_adv.detach().cpu()

for i in range(k_imgs):
    img_index = (true_lbl - 1) * k_imgs + i + 1

    orig_img = denormalize(X_sel_cpu[i].unsqueeze(0)).squeeze(0).clamp(0, 1)
    adv_img  = X_adv_cpu[i].squeeze(0).clamp(0, 1)

    TF.to_pil_image(orig_img).save(os.path.join(orig_dir, f"Universal{img_index}.png"))
    TF.to_pil_image(adv_img).save(os.path.join(attack_dir, f"UniversalAttack{img_index}.png"))

