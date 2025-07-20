#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from PIL import Image
import torchvision
from torchvision import transforms

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

ROOT_DIR = './cifar10_data'
OUTPUT_DIR = './cifar10_test_images'
SAVE_FORMAT = 'png'
ORGANIZE_BY_CLASS = True

def download_cifar10():
    print("Downloading the CIFAR-10 test set...")
    transform = transforms.ToTensor()
    torchvision.datasets.CIFAR10(
        root=ROOT_DIR, train=False, download=True, transform=transform
    )
    print("Download completed.")

def extract_images():
    print(f"Extracting images to:{OUTPUT_DIR}")
    dataset = torchvision.datasets.CIFAR10(
        root=ROOT_DIR, train=False, download=False, transform=None
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if ORGANIZE_BY_CLASS:
        for class_id, class_name in enumerate(CIFAR10_CLASSES):
            class_dir = os.path.join(OUTPUT_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)

        for i, (img, label) in enumerate(dataset):
            class_name = CIFAR10_CLASSES[label]
            save_path = os.path.join(OUTPUT_DIR, class_name, f"{class_name}_{i:04d}.{SAVE_FORMAT}")
            img.save(save_path)
            if i < 10 or i % 1000 == 0:
                print(f"Saving: {save_path}")
    else:
        for i, (img, label) in enumerate(dataset):
            class_name = CIFAR10_CLASSES[label]
            save_path = os.path.join(OUTPUT_DIR, f"test_{i:05d}_{class_name}.{SAVE_FORMAT}")
            img.save(save_path)
            if i < 10 or i % 1000 == 0:
                print(f"Saving: {save_path}")

    print("Extraction completed.")

def show_statistics():
    dataset = torchvision.datasets.CIFAR10(
        root=ROOT_DIR, train=False, download=False, transform=None
    )
    counts = [0] * 10
    for _, label in dataset:
        counts[label] += 1

    print("Class statistics:")
    for i, name in enumerate(CIFAR10_CLASSES):
        print(f"{name:12}: {counts[i]} images")

if __name__ == '__main__':
    download_cifar10()
    show_statistics()
    extract_images()
