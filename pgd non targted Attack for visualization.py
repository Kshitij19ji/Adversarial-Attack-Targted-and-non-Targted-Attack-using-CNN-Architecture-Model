true_label = 9
epsilon = 0.06
alpha = 0.005
num_iter = 100


X_true, y_true = [], []
for X_batch, y_batch in test_loader:
    mask = (y_batch == true_label)
    if mask.sum() > 0:
        X_true.append(X_batch[mask])
        y_true.append(y_batch[mask])
X_true = torch.cat(X_true, dim=0).to(device)
y_true = torch.cat(y_true, dim=0).to(device)

print(f" Total images found with true label = {true_label}: {X_true.size(0)}")

X_sel = X_true[:10]
y_sel = y_true[:10]


delta = pgd_linf_untarg(model_vgg19, X_sel, y_sel, epsilon=epsilon, alpha=alpha, num_iter=num_iter)

X_adv = X_sel + delta
yp = model_vgg19(X_adv)

success = (yp.argmax(1) != y_sel).float().mean().item()
print(f" Untargeted PGD Success Rate (True class {true_label} misclassified): {success:.3f}")


plot_images(X_adv, y_sel, yp, M=2, N=5, normalized=True)
