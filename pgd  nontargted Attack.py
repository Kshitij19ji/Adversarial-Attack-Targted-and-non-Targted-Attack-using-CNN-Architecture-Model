import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception as e:
            print(f'Warning: skipping corrupted image {path}')
           
            return self.__getitem__((index + 1) % len(self))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

def normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean
    



train_dir = "/data1/kshitij/imagenet_split/train"
test_dir = "/data1/kshitij/imagenet_split/test"



from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



train_dataset = SafeImageFolder(root=train_dir, transform=transform)
test_dataset = SafeImageFolder(root=test_dir, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=150, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=150, shuffle=True, num_workers=8, pin_memory=True)



for images, labels in test_loader:
    print("Batch labels:", labels)
    print("Unique labels in batch:", labels.unique())
    break
  import torch
import torch.nn.functional as F

def pgd_linf_untarg(model, X, y, epsilon=0.05, alpha=1e-2, num_iter=10):
    delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        logits = model(X + delta)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()

    return delta.detach() def plot_images(X, y, yp, M=2, N=5, normalized=True):
    preds = yp.argmax(dim=1)
    if normalized:
        X = denormalize(X).clamp(0, 1)  

    fig, ax = plt.subplots(M, N, figsize=(12, 6))
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
    plt.show() import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimg
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_vgg19 = models.vgg19(pretrained=False)
model_vgg19.classifier[6] = nn.Linear(4096, 990)
model_vgg19 = model_vgg19.to(device) criterion = F.cross_entropy model_vgg19.eval() torch.save(model_vgg19.state_dict(), "model_vgg19.pt")
model_vgg19.load_state_dict(torch.load("model_vgg19.pt", map_location=device)) for X,y in test_loader:
    X,y = X.to(device), y.to(device)
    break import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils



# Paths
orig_path = "/data1/kshitij/PGD/pgdUT/pgdOriginalimage"
attack_path = "/data1/kshitij/PGD/pgdUT/pgdAttackimage"
os.makedirs(orig_path, exist_ok=True)
os.makedirs(attack_path, exist_ok=True)

successful_100 = []
not_successful_100 = []

k_imgs = 10
epsilon = 0.02
alpha = 0.005
num_iter = 100

for true_lbl in range(1,201):
    X_true, y_true = [], []
    for X_batch, y_batch in test_loader:
        mask = (y_batch == true_lbl)
        if mask.sum() > 0:
            X_true.append(X_batch[mask])
            y_true.append(y_batch[mask])
        if sum(len(x) for x in X_true) >= k_imgs:
            break

    if len(X_true) == 0:
        continue

    X_true = torch.cat(X_true, dim=0)[:k_imgs].to(device)
    y_true = torch.cat(y_true, dim=0)[:k_imgs].to(device)

    delta = pgd_linf_untarg(model_vgg19, X_true, y_true, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
    X_adv = X_true + delta

    for i in range(k_imgs):
        index = (true_lbl - 1) * k_imgs + i + 1
        vutils.save_image(X_true[i].detach().cpu(), os.path.join(orig_path, f"pgd{index}.png"), normalize=True)
        vutils.save_image(X_adv[i].detach().cpu(), os.path.join(attack_path, f"pgdattack{index}.png"), normalize=True)

    print(f"Label {true_lbl}: {k_imgs} images attacked and saved.")
    if success == 1.0:
        successful_100.append(true_lbl)
    else:
        not_successful_100.append((true_lbl, success))
print("\n pgd Untargeted Attack Summary")
print(f"✔ 100% Success Labels: {successful_100}")

if not_successful_100:
    print(" Not Fully Successful Labels:")
    for lbl, rate in not_successful_100:
        print(f"  - Label {lbl} ➝ Success Rate: {rate:.2f}")
else:
    print(" All labels achieved 100% success.")


