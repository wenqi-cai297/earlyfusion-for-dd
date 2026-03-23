import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTokenizer
from collections import OrderedDict
from PIL import Image
import os
from tqdm import tqdm
import argparse
from torch.nn.functional import cosine_similarity
import numpy as np
import random
import kornia
import math


from modules import *

from classes import IMAGENET2012_CLASSES



def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    std = torch.clamp(std, min=1e-2, max=3.0)
    eps = torch.randn_like(std)
    return mu + eps * std

# ===================== Utility Functions =====================

def load_models():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None
    ).to("cuda")

    vae = pipe.vae.eval()
    unet = pipe.unet.train()
    text_encoder = pipe.text_encoder.eval()
    tokenizer = pipe.tokenizer
    scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    return vae, unet, text_encoder, tokenizer, scheduler

def create_dataloader(fg_path, bg_path, orig_path, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = FusedLatentDataset(fg_path, bg_path, orig_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def attn_dataloader(orig_path, batch_size, shuffle=False, image_size=512):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size=image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = CrossAttnDataset(orig_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ===================== softclip function =====================

def soft_clip_logvar(logvar, min_val=-10, max_val=2):
    return max_val - F.softplus(max_val - logvar) + F.softplus(logvar - min_val)

# ===================== KL loss with original latent distribution as target =====================

def kl_to_original_masked(mu_fused, logvar_fused, mu_orig, logvar_orig, fg_mask, bg_mask, fg_weight=1.0, bg_weight=0.2):
    var_fused = torch.exp(logvar_fused) + 1e-6
    var_orig = torch.exp(logvar_orig) + 1e-6
    kl = 0.5 * (logvar_orig - logvar_fused + (var_fused + (mu_fused - mu_orig).pow(2)) / var_orig - 1)

    # Apply masks and compute weighted mean
    mask_sum = (fg_weight * fg_mask + bg_weight * bg_mask).sum()
    kl_weighted = kl * (fg_weight * fg_mask + bg_weight * bg_mask)
    return kl_weighted.sum() / (mask_sum + 1e-6)

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def build_latent_cache(dataloader, vae, fusion_module, text_encoder, tokenizer,
                       cache_dir="./latent_cache", device="cuda"):

    os.makedirs(cache_dir, exist_ok=True)

    z_path = os.path.join(cache_dir, "z.pt")
    text_path = os.path.join(cache_dir, "text.pt")
    label_path = os.path.join(cache_dir, "label.pt")

    print(f"Building latent cache to {cache_dir}...")

    z_all, t_all, l_all = [], [], []

    vae.eval()
    fusion_module.eval()
    text_encoder.eval()

    with torch.no_grad():
        for orig_img, texts, labels in tqdm(dataloader, desc="Caching"):
            orig_img = orig_img.to(device)
            labels = labels.to(device)

            tokens = tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
            text_out = text_encoder(tokens.input_ids)
            text_embeds = getattr(text_out, "last_hidden_state", text_out[0]) 
            orig_mu = vae.encode(orig_img).latent_dist.mean
            z_fused = fusion_module(orig_mu, text_embeds)

            z_all.append(z_fused.cpu())
            t_all.append(text_embeds.cpu())
            l_all.append(labels.cpu())

    z_cat = torch.cat(z_all)
    t_cat = torch.cat(t_all)
    l_cat = torch.cat(l_all)

    torch.save(z_cat, z_path)
    torch.save(t_cat, text_path)
    torch.save(l_cat, label_path)

    print(f"Saved cache to {cache_dir}")


def build_latent1k_cache(dataloader, vae, fusion_module, text_encoder, tokenizer,
                                 cache_dir="./latent_cache", device="cuda", num_samples=None):
    os.makedirs(cache_dir, exist_ok=True)
    z_path = os.path.join(cache_dir, "z.memmap")
    text_path = os.path.join(cache_dir, "text.memmap")
    label_path = os.path.join(cache_dir, "label.memmap")
    meta_path = os.path.join(cache_dir, "meta.pt")


    vae.eval()
    fusion_module.eval()
    text_encoder.eval()

    first_batch = next(iter(dataloader))
    orig_img, texts, labels = first_batch

    B = orig_img.size(0)
    orig_img = orig_img.to(device)

    with torch.no_grad():
        tokens = tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
        text_out = text_encoder(tokens.input_ids)
        text_embeds = getattr(text_out, "last_hidden_state", text_out[0]) 
        orig_mu = vae.encode(orig_img).latent_dist.mean
        z_fused = fusion_module(orig_mu, text_embeds)

    z_shape = z_fused.shape[1:]  # [C, H, W]
    text_shape = text_embeds.shape[1:]  # [77, 768] for CLIP
    label_shape = ()  # scalar

    if num_samples is None:
        num_samples = len(dataloader.dataset)

    dtype = np.float32
    z_memmap = np.memmap(z_path, dtype=dtype, mode='w+', shape=(num_samples, *z_shape))
    text_memmap = np.memmap(text_path, dtype=dtype, mode='w+', shape=(num_samples, *text_shape))
    label_memmap = np.memmap(label_path, dtype=np.int64, mode='w+', shape=(num_samples,))

    idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Memmap] Caching"):
            orig_img, texts, labels = batch
            B = orig_img.size(0)

            orig_img = orig_img.to(device)
            labels = labels.to(device)

            tokens = tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
            text_out = text_encoder(tokens.input_ids)
            text_embeds = getattr(text_out, "last_hidden_state", text_out[0]) 
            orig_mu = vae.encode(orig_img).latent_dist.mean
            z_fused = fusion_module(orig_mu, text_embeds)

            z_memmap[idx:idx + B] = z_fused.cpu().numpy()
            text_memmap[idx:idx + B] = text_embeds.cpu().numpy()
            label_memmap[idx:idx + B] = labels.cpu().numpy()

            idx += B

    z_memmap.flush()
    text_memmap.flush()
    label_memmap.flush()

    torch.save({
        "num_samples": num_samples,
        "z_shape": z_shape,
        "text_shape": text_shape,
        "label_shape": label_shape,
    }, meta_path)

    print(f"[✓] Cached {idx} samples into memmap at {cache_dir}")


# def info_nce_loss(projected_z: torch.Tensor, text_embed: torch.Tensor, temperature: float = 0.07):
#     """
#     projected_z: (B, D)  latent vectors after cross-attn and projector (unnormalized)
#     text_embed: (B, D)  corresponding text embeddings (unnormalized)
#     temperature: scaling factor, default 0.07
    
#     Returns:
#         loss: scalar InfoNCE loss
#     """
#     projected_z = projected_z.float()
#     text_embed  = text_embed.float()
#     # print(f"projected z shape: {projected_z.shape}, text embed shape: {text_embed.shape}")
#     # L2 normalize embeddings
#     projected_z_norm = F.normalize(projected_z, dim=-1)        # (B, D)
#     text_embed_norm = F.normalize(text_embed, dim=-1)  # (B, D)

#     # Compute similarity matrix (logits), shape: (B, B)
#     logits = torch.matmul(projected_z_norm, text_embed_norm.t())  # (B, B)
#     logits = logits / temperature

#     # Target labels are diagonal indices (each sample corresponds to the matching text)
#     labels = torch.arange(projected_z.size(0), device=projected_z.device)

#     # Cross-entropy loss
#     loss = F.cross_entropy(logits, labels)
#     return loss

def l2_loss(fused_z: torch.Tensor, orig_z: torch.Tensor):
    """
    fused_z: (B, C, H, W) - fused latent tensor
    orig_z: (B, C, H, W) - original latent tensor
    
    Returns:
        loss: scalar L2 loss
    """
    loss = F.mse_loss(fused_z.float(), orig_z.float())
    return loss

def ramp(value_start, value_end, cur_step, total_steps):
    """Linear schedule from value_start to value_end across total_steps."""
    t = min(max(cur_step / max(total_steps, 1), 0.0), 1.0)
    return value_start + (value_end - value_start) * t

def multi_pos_acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    logits: (B, B) 相似度 (img->text 或 text->img)
    labels: (B,) 每个样本的类别标签
    """
    pred_idx = logits.argmax(dim=1)  # (B,)
    correct = (labels[pred_idx] == labels).float()
    return correct.mean().item()


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])