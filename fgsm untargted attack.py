import torch

@torch.no_grad()
def _project_linf(x, x_orig, eps):
    return torch.clamp(x, 0, 1).clamp(x_orig - eps, x_orig + eps)


def fgsm_minimal_untargeted_attack(model, x_orig, y_true, eps_max=0.10, eps_step=0.01):
    """
    FGSM Untargeted Minimal Attack (iterative until success or max_eps)
    Args:
        model     : victim model
        x_orig    : original clean input (batch)
        y_true    : ground-truth labels
        eps_max   : maximum epsilon
        eps_step  : epsilon step
    Returns:
        x_adv     : adversarial images
        n_steps   : steps taken for each image to succeed
    """
    x_adv    = x_orig.clone().detach()
    n_steps  = torch.zeros(x_orig.size(0), dtype=torch.int, device=x_orig.device)
    success  = torch.zeros_like(n_steps, dtype=torch.bool)

    cur_eps = 0.0
    while (not success.all()) and cur_eps <= eps_max:
        x_adv.requires_grad_(True)
        logits = model(normalize(x_adv))
        loss   = nn.CrossEntropyLoss()(logits, y_true)  
        model.zero_grad()
        loss.backward()

        perturb = eps_step * x_adv.grad.sign()
        x_adv   = x_adv.detach() + perturb
        x_adv   = _project_linf(x_adv, x_orig, cur_eps + eps_step)

        preds = model(normalize(x_adv)).argmax(1)
        newly = (preds != y_true) & (~success) 
        n_steps[newly] = int(cur_eps / eps_step) + 1
        success |= newly

        cur_eps += eps_step

    return x_adv.detach(), n_steps.cpu()

def plot_minimal_fgsm(X_adv, y_true, y_pred_logits, y_target, n_steps, M=2, N=5, suptitle=""):
    preds = y_pred_logits.argmax(1).cpu()
    fig, ax = plt.subplots(M, N, figsize=(12, 6))

    for i in range(M):
        for j in range(N):
            k = i * N + j
            if k >= X_adv.size(0): break

            img = denormalize(X_adv[k].unsqueeze(0)).squeeze(0).cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

            img = np.clip(img, 0, 1)
            ax[i][j].imshow(img)

            t = y_true[k].item()
            p = preds[k].item()
            tgt = y_target[k].item()
            color = 'g' if p == tgt else 'r'
            ax[i][j].set_title(f"True: {t} → Pred: {p}", color=color, fontsize=10)
            ax[i][j].axis('off')

    plt.suptitle(suptitle, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

import os
import torch
import torchvision.transforms.functional as TF

true_lbl      = 1
k_imgs        = 10
eps_max       = 50
eps_step      = 0.005

title = f"FGSM Untargeted Attack (True Label: {true_lbl}, ε ≤ {eps_max})"

X_sel, y_sel = [], []
for xb, yb in test_loader:
    m = yb == true_lbl
    if m.any():
        X_sel.append(xb[m])
        y_sel.append(yb[m])
    if sum(x.size(0) for x in X_sel) >= k_imgs:
        break

X_sel  = torch.cat(X_sel)[:k_imgs].to(device)
y_sel  = torch.cat(y_sel)[:k_imgs].to(device)

X_adv, steps = fgsm_minimal_untargeted_attack(
    model_vgg19, X_sel, y_sel,
    eps_max=eps_max, eps_step=eps_step
)

logits  = model_vgg19(normalize(X_adv))
preds   = logits.argmax(1)
success = (preds != y_sel).float().mean().item()

plot_minimal_fgsm(X_adv, y_sel.cpu(), logits.cpu(), y_sel.cpu(), steps,
                  M=2, N=5, suptitle=title)

print(f"Untargeted FGSM Success Rate: {success * 100:.1f}%")

base_path   = "/data1/kshitij/PGD/fgsmUT"
orig_dir    = os.path.join(base_path, "fgsmOriginalimage")
attack_dir  = os.path.join(base_path, "fgsmAttackedimage")
os.makedirs(orig_dir, exist_ok=True)
os.makedirs(attack_dir, exist_ok=True)

for i in range(k_imgs):
    index = (true_lbl - 1) * k_imgs + i + 1

    orig_img   = denormalize(X_sel[i].unsqueeze(0)).squeeze(0).cpu().clamp(0, 1)
    attack_img = denormalize(X_adv[i].unsqueeze(0)).squeeze(0).cpu().clamp(0, 1)

    TF.to_pil_image(orig_img).save(os.path.join(orig_dir,   f"fgsmUT{index}.png"))
    TF.to_pil_image(attack_img).save(os.path.join(attack_dir, f"fgsmUTattack{index}.png"))

