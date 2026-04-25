def pgd_linf_targ(model, X, y, epsilon=0.05, alpha=1e-2, num_iter=10, y_targ=None):
    delta = torch.zeros_like(X, requires_grad=True)

    if isinstance(y_targ, int):
        y_targ = torch.full_like(y, y_targ) 

    for t in range(num_iter):
        yp = model(X + delta)
        loss = F.cross_entropy(yp, y_targ)
        loss.backward()

        
        delta.data = (delta - alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()

    return delta.detach()


def plot_images(X, y, yp, M=3, N=6):
    preds = yp.argmax(dim=1)
    fig, ax = plt.subplots(M, N, figsize=(12, 12))
    for i in range(M):
        for j in range(N):
            idx = i * N + j
            if idx >= X.size(0): break
            img = X[idx].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax[i][j].imshow(img)
            title = ax[i][j].set_title(f"True: {y[idx].item()}, Pred: {preds[idx].item()}")
            plt.setp(title, color=('g' if preds[idx] == y[idx] else 'r'))
            ax[i][j].axis('off')
    plt.tight_layout()
    plt.show()


import os
import torch
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save directories
orig_dir = "/data/kshitij/PGD/PgdOriginalimage"
attack_dir = "/data/kshitij/PGD/pgdAttackimage"
os.makedirs(orig_dir, exist_ok=True)
os.makedirs(attack_dir, exist_ok=True)

# Parameters
true_labels = [146, 147, 148, 149, 150]  
target_class = 0

for true_label in true_labels:
    X_true, y_true = [], []

    for X_batch, y_batch in test_loader:
        mask = (y_batch == true_label)
        if mask.sum() > 0:
            X_true.append(X_batch[mask])
            y_true.append(y_batch[mask])

    if len(X_true) == 0:
        print(f" No samples found for class {true_label}")
        continue

    X_true = torch.cat(X_true, dim=0).to(device)
    y_true = torch.cat(y_true, dim=0).to(device)

    X_sel = X_true[:10]
    y_sel = y_true[:10]

    delta = pgd_linf_targ(
        model_resnet50, X_sel, y_sel,
        epsilon=0.02, alpha=0.005, num_iter=500,
        y_targ=target_class
    )

    X_adv = X_sel + delta
    yp = model_resnet50(X_adv).clamp(0, 1)

    success = (yp.argmax(1) == target_class).float().mean().item()
    print(f" PGD Attack Success (class {true_label} ➝ {target_class}): {success:.3f}")

    X_orig = denormalize(X_sel).clamp(0, 1)
    X_adv = denormalize(X_adv).clamp(0, 1)

    for i in range(10):
        index = (true_label - 1) * 10 + i + 1
        orig_path = os.path.join(orig_dir, f"pgd{index}.png")
        attack_path = os.path.join(attack_dir, f"pgd{index}attack.png")

        TF.to_pil_image(X_orig[i].cpu()).save(orig_path)
        TF.to_pil_image(X_adv[i].cpu()).save(attack_path)

    print(" Saved 10 original and attacked PGD images.")

    plot_images(X_adv, y_sel, yp, M=2, N=5)
