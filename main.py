import os
import torch
from torchvision import models

def main():
    print("Welcome to the Adversarial Attacks Research Project")
    print("--------------------------------------------------")
    print("Available Attacks:")
    print("1. FGSM (Targeted & Untargeted)")
    print("2. PGD (Targeted & Untargeted)")
    print("3. CW (Targeted & Untargeted)")
    print("4. UAP (Universal Adversarial Perturbations)")
    print("\nModels available in repository:")
    print("- VGG19\n- ResNet50\n- MobileNetV2\n- EfficientNet")
    
    # You can customize this main.py to serve as a wrapper for your scripts.
    # Example: python main.py --attack pgd --target 0
    print("\nPlease run the specific attack scripts to generate results.")

if __name__ == "__main__":
    main()
