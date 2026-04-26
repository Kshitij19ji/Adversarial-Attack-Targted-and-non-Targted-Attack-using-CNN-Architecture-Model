import torch
import torch.nn as nn
import torch.optim as optim

def cw_l2_attack(model, images, labels, target_class, targeted=True, c=0.2, kappa=0, steps=500, lr=0.01):
    """
    Carlini & Wagner L2 Attack (Targeted, Fixed)
    Returns adversarial images.
    """
    device = images.device
    batch_size = images.size(0)

   
    epsilon = 1e-6
    clamped = (images * 2 - 1).clamp(-1 + epsilon, 1 - epsilon)
    w_init = 0.5 * torch.log((1 + clamped) / (1 - clamped))
    w = w_init.clone().detach().to(device)
    w.requires_grad = True 

    optimizer = optim.Adam([w], lr=lr)

    
    target_labels = torch.full_like(labels, target_class)
    one_hot_target = torch.eye(model(images).shape[1], device=device)[target_labels]

    for step in range(steps):
        adv_images = torch.tanh(w) * 0.5 + 0.5
        logits = model(normalize(adv_images))

        real = (one_hot_target * logits).sum(dim=1)
        other = ((1 - one_hot_target) * logits - one_hot_target * 1e4).max(dim=1)[0]

        
        if targeted:
            f_loss = torch.clamp(other - real + kappa, min=0)
        else:
            f_loss = torch.clamp(real - other + kappa, min=0)

        
        l2_loss = ((adv_images - images) ** 2).view(batch_size, -1).sum(dim=1)
        total_loss = l2_loss + c * f_loss

        optimizer.zero_grad()
        total_loss.sum().backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            print(f"[Step {step}] L2: {l2_loss.mean().item():.4f}, f: {f_loss.mean().item():.4f}, Total: {total_loss.mean().item():.4f}")

    return torch.tanh(w) * 0.5 + 0.5   
    
    
    
import matplotlib.pyplot as plt
import numpy as np

def plot_adv_images(X, true_labels, predicted_logits, target_class=None, M=3, N=6):
    preds = predicted_logits.argmax(dim=1)
    fig, ax = plt.subplots(M, N, figsize=(12, 12))

    for i in range(M):
        for j in range(N):
            idx = i * N + j
            if idx >= X.size(0): break

            img = X[idx].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax[i][j].imshow(img)

            t = true_labels[idx].item()
            p = preds[idx].item()
            ax[i][j].set_title(f"T:{t} → P:{p}", color=('g' if p == target_class else 'r'))
            ax[i][j].axis('off')

    plt.tight_layout()
    plt.show() 
    
    
    
    
import os
import torch
import torchvision.transforms.functional as TF


orig_dir = "/data1/kshitij/PGD/CW/CWOriginalimage"
attack_dir = "/data1/kshitij/PGD/CW/CWAttackimage"
os.makedirs(orig_dir, exist_ok=True)
os.makedirs(attack_dir, exist_ok=True)


true_label = 199
target_class = 0
num_images = 10


X_true, y_true = [], []
for X_batch, y_batch in test_loader:
    mask = (y_batch == true_label)
    if mask.sum() > 0:
        X_true.append(X_batch[mask])
        y_true.append(y_batch[mask])
    if sum(len(x) for x in X_true) >= num_images:
        break

if len(X_true) == 0:
    print(f" No images found for label {true_label}")
else:
    X_true = torch.cat(X_true, dim=0)[:num_images].to(device)
    y_true = torch.cat(y_true, dim=0)[:num_images].to(device)
    X_clean = denormalize(X_true)

   
    X_adv = cw_l2_attack(model_vgg19, X_clean, y_true,
                         target_class=target_class, c= 1, kappa=100, steps=1000, lr=0.005)
    yp = model_vgg19(normalize(X_adv)).detach()

   
    for i in range(num_images):
        index = (true_label - 1) * 10 + i + 1
        orig_path = os.path.join(orig_dir, f"cw{index}.png")
        attack_path = os.path.join(attack_dir, f"cw{index}attack.png")

        TF.to_pil_image(X_clean[i].cpu().clamp(0, 1)).save(orig_path)
        TF.to_pil_image(X_adv[i].cpu().clamp(0, 1)).save(attack_path)


    success = (yp.argmax(1) == target_class).float().mean().item()
    print(f"\n CW Attack Success Rate (label {true_label} ➝ {target_class}): {success:.3f}")
    plot_adv_images(X_adv, y_true, yp, target_class=target_class, M=2, N=5)    


