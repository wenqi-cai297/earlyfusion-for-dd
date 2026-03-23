import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTokenizer
from PIL import Image
import os
from tqdm import tqdm
import argparse
import numpy as np
import random
from accelerate import Accelerator
from classes import IMAGENET2012_CLASSES
from modules import *
from utils import *
import json

# ===================== Utils =====================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--image_size', type=int, choices=[256, 512], default=512)
    parser.add_argument('--epoch', type=int, default=30, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--lambda1', type=float, default=0.1, help="lambda for reconstruction loss.")
    parser.add_argument('--dataset', type=str, default="imagenet")
    parser.add_argument('--subset', type=str, default="nette")
    parser.add_argument('--original_dataset_path', type=str, default="path/ImageNette/train")
    args = parser.parse_args()
    return args


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================== Losses =====================

class SymmetricInfoNCE_MultiPos(nn.Module):
    """
    Symmetric multi-positive InfoNCE with learnable temperature.
    - Treats same-class pairs as positives (multi-positives).
    - image->text and text->image losses are averaged.
    - Invariant to #positives via log-mean-exp over positives.

    Args:
        init_temp: initial temperature T (default 0.07)
        include_self: whether (i,i) is considered positive. Usually True.
    """
    def __init__(self, init_temp: float = 0.07, include_self: bool = True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temp)))
        self.include_self = include_self

    @staticmethod
    def _multi_pos_ce(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, B) raw similarity logits
        pos_mask: (B, B) boolean mask of positives for each row
        Returns mean over batch of -log(mean P(positives))
        """
        # Normalize row-wise to probabilities (in log-space for stability)
        log_probs = logits - logits.logsumexp(dim=1, keepdim=True)  # (B, B)

        # Mask out non-positives with -inf so they don't contribute
        neg_inf = torch.finfo(logits.dtype).min
        log_p_pos = torch.where(pos_mask, log_probs, torch.full_like(log_probs, neg_inf))

        # log-mean-exp over positives: log( sum(exp)/#pos )
        # -> invariant to how many positives are in the row
        log_sum_pos = log_p_pos.logsumexp(dim=1)  # (B,)
        num_pos = pos_mask.sum(dim=1).clamp_min(1)  # avoid div-by-zero
        loss = -(log_sum_pos - num_pos.float().log()).mean()
        return loss

    def forward(self,
                img_feat: torch.Tensor,
                txt_feat: torch.Tensor,
                labels: torch.Tensor = None,
                pos_mask: torch.Tensor = None) -> torch.Tensor:
        """
        img_feat, txt_feat: (B, D)
        labels: (B,) same label => positive (if provided)
        pos_mask: (B, B) boolean; overrides labels if provided
        """
        # Normalize in fp32
        z = F.normalize(img_feat.float(), dim=-1)  # (B, D)
        t = F.normalize(txt_feat.float(), dim=-1)  # (B, D)
        scale = self.logit_scale.exp()

        logits_i2t = z @ t.t() * scale  # (B, B)
        logits_t2i = t @ z.t() * scale  # (B, B)

        B = z.size(0)
        if pos_mask is None:
            assert labels is not None, "Provide either labels or a pos_mask."
            pos_mask = labels.unsqueeze(1).eq(labels.unsqueeze(0))  # (B,B) same-class = True
        if not self.include_self:
            # optionally exclude self-pairs from positives
            eye = torch.eye(B, dtype=torch.bool, device=pos_mask.device)
            pos_mask = pos_mask & (~eye)

        loss_i2t = self._multi_pos_ce(logits_i2t, pos_mask)
        loss_t2i = self._multi_pos_ce(logits_t2i, pos_mask)
        return 0.5 * (loss_i2t + loss_t2i)




# ===================== Training =====================

def train_one_epoch(args, epoch, dataloader, vae, text_encoder, tokenizer,
                    fusion_module, latent_projector, sym_nce_loss,
                    optimizer, device, accelerator, accumulation_steps=2):
    fusion_module.train()
    latent_projector.train()

    lambda1 = args.lambda1
    total_steps = args.epoch * len(dataloader) / 2

    running_loss = 0.0
    count = 0

    optimizer.zero_grad()

    for batch_idx, (orig_img, texts, labels) in enumerate(
        tqdm(dataloader, desc=f"Epoch {epoch} Training", disable=not accelerator.is_local_main_process)
    ):
        # global schedule for lambda2 across the entire training run
        global_step = epoch * len(dataloader) + batch_idx
        lambda2 = ramp(0.05, 1.0, global_step, total_steps)

        # --- to device ---
        orig_img = orig_img.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # --- tokenize on CPU, then move tensors ---
        tokenized = tokenizer(
            texts,
            padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        ).to(device)

        # --- frozen encoders forward (under autocast fp16) ---
        with torch.no_grad(), accelerator.autocast():
            text_out = text_encoder(tokenized.input_ids)
            text_embeds = getattr(text_out, "last_hidden_state", text_out[0])  # (B, 77, D)
            orig_mu = vae.encode(orig_img).latent_dist.mean                    # (B, C, H, W)

        # --- EOT pooling (fallback to last valid token by attention_mask) ---
        ids = tokenized.input_ids
        attn = tokenized.attention_mask
        B, L = ids.size()

        eos_id = getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "eot_token_id", None)
        if eos_id is not None:
            idx = torch.arange(L, device=ids.device).unsqueeze(0).expand_as(ids)  # (B, L)
            mask = (ids == eos_id)
            last_eos_pos = torch.where(mask, idx, torch.full_like(idx, -1)).max(dim=1).values  # (B,)
        else:
            last_eos_pos = torch.full((B,), -1, device=ids.device, dtype=torch.long)

        last_valid_pos = attn.sum(dim=1) - 1  # (B,)
        eot_pos = torch.where(last_eos_pos >= 0, last_eos_pos, last_valid_pos).clamp(min=0, max=L-1)  # (B,)

        text_vec = text_embeds[torch.arange(B, device=ids.device), eot_pos, :]  # (B, D)

        p_empty = 0.2
        p_token_drop = 0.1

        if torch.rand(()) < p_empty:
            cond = torch.zeros_like(text_embeds) 
        else:
            cond = F.dropout(text_embeds, p=p_token_drop, training=True) 
            cond = cond + 0.01 * torch.randn_like(cond)

        # --- fp16 boundary: switch to float32 before trainable modules / losses ---
        text_vec = text_vec.float()
        orig_mu = orig_mu.float()

        with accelerator.accumulate(fusion_module):
            fused_mu = fusion_module(orig_mu, cond.float())
            projected_mu = latent_projector(fused_mu)  # (B, D), should match text D (e.g., 768)

            # symmetric InfoNCE + L2
            loss1 = sym_nce_loss(projected_mu, text_vec, labels=labels)
            loss2 = l2_loss(fused_mu, orig_mu)

            loss = lambda1 * loss1 + lambda2 * loss2

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params = (
                    list(fusion_module.parameters())
                    + list(latent_projector.parameters())
                    
                )
                accelerator.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # --- metrics across processes ---
        loss_g = accelerator.gather_for_metrics(loss.detach())
        running_loss += loss_g.mean().item()
        count += 1

        # --- lightweight monitoring (main process only) ---
        if accelerator.is_main_process and (batch_idx % 50 == 0):
            with torch.no_grad():
                # normalized features -> cosine similarity
                z = F.normalize(projected_mu.float(), dim=-1)   # (B, D)
                t = F.normalize(text_vec.float(), dim=-1)       # (B, D)
                sims = z @ t.t()                                # (B, B)  image->text

                # multi-positive mask by class
                pos_mask = labels.unsqueeze(1).eq(labels.unsqueeze(0))      # (B,B) True if same class
                neg_mask = ~pos_mask

                # multi-positive accuracy (image->text): top-1 matches any same-class text is counted correct
                pred_idx = sims.argmax(dim=1)                               # (B,)
                acc_mp = (labels[pred_idx] == labels).float().mean().item()

                # mean positive cosine (row-wise mean over all positives, then batch mean)
                pos_counts = pos_mask.sum(dim=1).clamp_min(1)
                cos_pos_mean = ((sims * pos_mask).sum(dim=1) / pos_counts).mean().item()

                # hardest negative cosine (row-wise max over negatives, then batch mean)
                sims_neg = sims.masked_fill(pos_mask, float('-inf'))
                hardest_neg = sims_neg.max(dim=1).values
                hardest_neg[~torch.isfinite(hardest_neg)] = 0.0
                cos_margin = (cos_pos_mean - hardest_neg.mean().item())

                # temperature record
                sd = accelerator.get_state_dict(sym_nce_loss)
                ls = sd["logit_scale"].item()
                T = 1.0 / np.exp(ls)


            with open("monitor_epoch.txt", "a") as f:
                f.write(
                    f"[epoch {epoch} batch {batch_idx}] "
                    f"sym-info: {loss1.item():.4f}, l2: {loss2.item():.4f}, total loss: {loss.item():.4f}, "
                    f"acc_mp(i2t): {acc_mp:.3f}, cos_pos: {cos_pos_mean:.4f}, margin: {cos_margin:.4f}, "
                    f"lambda1: {lambda1:.3f}, lambda2: {lambda2:.3f}, logit_scale: {ls: .4f}, temperature: {T: .4f}\n"
                )


    avg_loss = running_loss / max(count, 1)
    if accelerator.is_main_process:
        with open("monitor_epoch.txt", "a") as f:
            f.write(f"Epoch {epoch} Summary -> Total: {avg_loss:.4f}\n\n")
    return avg_loss


# ===================== Main =====================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    accumulation_steps = 2
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps,
        mixed_precision="fp16"
    )

    # load_models() should return: vae, _, text_encoder, tokenizer, _
    vae, _, text_encoder, tokenizer, _ = load_models()

    fusion_module = CrossAttentionFusion(num_layers=2, image_size=args.image_size)
    latent_projector = LatentProjector()
    sym_nce_loss = SymmetricInfoNCE_MultiPos(init_temp=0.07, include_self=True).to(accelerator.device)

    # include fusion, projector, and loss parameters (logit_scale) in the optimizer
    optimizer = torch.optim.AdamW([
        {"params": fusion_module.parameters(),    "lr": 3e-4, "weight_decay": 1e-2},
        {"params": latent_projector.parameters(), "lr": 1e-4, "weight_decay": 1e-2},
  
    ])

    dataloader = attn_dataloader(args.original_dataset_path, batch_size=args.batch_size, shuffle=True, image_size=args.image_size)

    # prepare trainable/sync objs (do NOT prepare tokenizer/vae)
    dataloader, text_encoder, fusion_module, latent_projector, sym_nce_loss, optimizer = accelerator.prepare(
        dataloader, text_encoder, fusion_module, latent_projector, sym_nce_loss, optimizer
    )

    # freeze encoders
    vae.eval()
    text_encoder.eval()
    sym_nce_loss.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    # put VAE on the correct device (not prepared)
    vae.to(accelerator.device)

    fusion_module.train()
    latent_projector.train()

    for epoch in tqdm(range(args.epoch), desc="Training Epochs", disable=not accelerator.is_local_main_process):
        avg_loss = train_one_epoch(
            args, epoch, dataloader, vae, text_encoder, tokenizer,
            fusion_module, latent_projector, sym_nce_loss,
            optimizer, device, accelerator, accumulation_steps
        )

        if accelerator.is_main_process:
            # save_dir = "./fusion_module_with_text"
            save_dir = "./fusion_module_with_text/ablation"
            os.makedirs(save_dir, exist_ok=True)

            state = {
                "fusion_module": accelerator.get_state_dict(fusion_module),
                "latent_projector": accelerator.get_state_dict(latent_projector),
                "sym_nce_loss": accelerator.get_state_dict(sym_nce_loss),
                "epoch": epoch,
                "avg_loss": avg_loss,
            }
            accelerator.save(state, f"{save_dir}/{args.dataset}-{args.subset}-{args.image_size}x{args.image_size}.pth")



if __name__ == "__main__":
    args = parse_args()
    main(args)
