import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from .classes import IMAGENET2012_CLASSES, tiny_imagenet_CLASSES
from PIL import Image
# ===================== Dataset =====================

class FusedLatentDataset(Dataset):
    def __init__(self, fg_root, bg_root, orig_root, transform):
        self.orig_root = orig_root
        self.fg_dataset = datasets.ImageFolder(fg_root)
        self.bg_dataset = datasets.ImageFolder(bg_root)
        self.orig_dataset = datasets.ImageFolder(orig_root)
        assert self.fg_dataset.classes == self.bg_dataset.classes == self.orig_dataset.classes
        self.idx_to_class = {v: k for k, v in self.fg_dataset.class_to_idx.items()}
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])

    def __len__(self):
        return len(self.fg_dataset)

    def extract_alpha_mask(self, img):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        alpha = img.split()[-1]  # get alpha channel
        mask = self.mask_transform(alpha)  # [1, H, W], binary float
        return mask

    def __getitem__(self, idx):
        fg_path, class_id_fg = self.fg_dataset.samples[idx]
        bg_path, class_id_bg = self.bg_dataset.samples[idx]
        orig_path, class_id_orig = self.orig_dataset.samples[idx]

        assert class_id_fg == class_id_bg == class_id_orig
        
        fg_img = Image.open(fg_path).convert("RGBA")
        bg_img = Image.open(bg_path).convert("RGBA")
        orig_img = Image.open(orig_path).convert("RGBA")

        fg_mask = self.extract_alpha_mask(fg_img)
        bg_mask = self.extract_alpha_mask(bg_img)

        if self.transform:
            fg_img = self.transform(orig_img.copy().convert("RGB"))

            orig_img = self.transform(orig_img.copy().convert("RGB"))

        class_name = self.idx_to_class[class_id_fg]
        if "tiny" in self.orig_root:
            prompt = tiny_imagenet_CLASSES.get(class_name, "an object")
        elif "Image" in self.orig_root:
            prompt = IMAGENET2012_CLASSES.get(class_name, "an object")
        elif "CIFAR" in self.orig_root:
            prompt = class_name

        return fg_img, orig_img, prompt, class_id_fg, fg_mask, bg_mask

class CrossAttnDataset(Dataset):
    def __init__(self, orig_root, transform):
        self.orig_root = orig_root
        self.orig_dataset = datasets.ImageFolder(orig_root)

        self.idx_to_class = {v: k for k, v in self.orig_dataset.class_to_idx.items()}
        self.transform = transform


    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):
        orig_path, class_id_orig = self.orig_dataset.samples[idx]
        orig_img = Image.open(orig_path)

        if self.transform:
            orig_img = self.transform(orig_img.copy().convert("RGB"))

        class_name = self.idx_to_class[class_id_orig]
        if "tiny" in self.orig_root:
            prompt = tiny_imagenet_CLASSES.get(class_name, "an object")
        elif "Image" in self.orig_root:
            prompt = IMAGENET2012_CLASSES.get(class_name, "an object")
        elif "CIFAR" in self.orig_root:
            prompt = class_name

        return orig_img, prompt, class_id_orig