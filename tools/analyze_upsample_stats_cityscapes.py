#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import tqdm
from PIL import Image
import torch


def analyze(input, patch_sizes):
    img = np.asarray(Image.open(input))
    img = torch.from_numpy(img)
    img, pad_h_w = pad_to_nearest_multiple(img, patch_sizes[0])
    upsample_ratio_per_scale = {}
    prev_upsample_ratio = 1
    for ps in patch_sizes:
        edge_map = count_edges_in_patch(img, pad_h_w, ps)
        needs_upsampling = edge_map.nonzero().shape[0]
        if prev_upsample_ratio == 0:
            upsample_ratio = 0
        else:
            upsample_ratio = needs_upsampling / (edge_map.shape[0] * prev_upsample_ratio)
        upsample_ratio_per_scale[ps] = upsample_ratio
        prev_upsample_ratio *= upsample_ratio

    return upsample_ratio_per_scale


def pad_to_nearest_multiple(img, patch_size):
    h, w = img.shape
    if h % patch_size == 0:
        pad_h = 0
    else:
        pad_h = patch_size - h % patch_size
    if w % patch_size == 0:
        pad_w = 0
    else:
        pad_w = patch_size - w % patch_size
    new_h = h + pad_h
    new_w = w + pad_w
    padded_masks = torch.zeros((new_h, new_w), dtype=img.dtype)
    padded_masks[:] = 254
    padded_masks[: img.shape[0], : img.shape[1]] = img
    return padded_masks, (pad_h, pad_w)

def count_edges_in_patch(targets, targets_pad, patch_size=32):
    targets_batch = targets.squeeze()
    targets_shifted = (targets_batch.byte() + 2).long()
    pad_h, pad_w = targets_pad
    border_mask = get_ignore_mask(targets_shifted, pad_h, pad_w)
    edge_mask = compute_edge_mask_with_ignores(targets_shifted, border_mask)
    edges = count_edges_per_patch_masked(edge_mask, patch_size=patch_size)
    return edges

def get_ignore_mask(label_map, pad_h, pad_w, border_size=5):
    H, W = label_map.shape
    usable_h = H - pad_h
    usable_w = W - pad_w

    ignore_mask = (label_map == 0)
    border_mask = torch.zeros_like(label_map, dtype=torch.bool)
    border_mask[:border_size, :usable_w] = True
    border_mask[usable_h - border_size:usable_h, :usable_w] = True
    border_mask[:usable_h, :border_size] = True
    border_mask[:usable_h, usable_w - border_size:usable_w] = True

    class1_mask = (label_map == 1)
    ignore_mask |= class1_mask & border_mask
    return ignore_mask


def compute_edge_mask_with_ignores(label_map, ignore_mask):
    H, W = label_map.shape
    edge_mask = torch.zeros_like(label_map, dtype=torch.bool)

    # Top neighbor (i, j) vs (i-1, j)
    valid = (~ignore_mask[1:, :]) & (~ignore_mask[:-1, :])
    diff = label_map[1:, :] != label_map[:-1, :]
    edge_mask[1:, :] |= valid & diff

    # Bottom neighbor
    valid = (~ignore_mask[:-1, :]) & (~ignore_mask[1:, :])
    diff = label_map[:-1, :] != label_map[1:, :]
    edge_mask[:-1, :] |= valid & diff

    # Left neighbor
    valid = (~ignore_mask[:, 1:]) & (~ignore_mask[:, :-1])
    diff = label_map[:, 1:] != label_map[:, :-1]
    edge_mask[:, 1:] |= valid & diff

    # Right neighbor
    valid = (~ignore_mask[:, :-1]) & (~ignore_mask[:, 1:])
    diff = label_map[:, :-1] != label_map[:, 1:]
    edge_mask[:, :-1] |= valid & diff

    return edge_mask

def count_edges_per_patch_masked(edge_mask, patch_size):
    H, W = edge_mask.shape
    P = patch_size
    patches = edge_mask.view(H // P, P, W // P, P).permute(0, 2, 1, 3)
    patches = patches.reshape(-1, P, P)
    return patches.sum(dim=(1, 2))

def find_pos_org_order(pos_org, pos_shuffled):
    dists = torch.cdist(pos_org.float(), pos_shuffled.float(), p=1)  # Manhattan distance
    pos_indices = torch.argmin(dists, dim=2)  # (B, N_)

    return pos_indices




if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "cityscapes"
    annotation_dir = dataset_dir / "gtFine" / "train"
    files = glob.glob(os.path.join(annotation_dir, '*/*_instanceIds.png'))
    patch_sizes = [64, 32, 16, 8, 4, 2]
    print_freq = 1000
    i = 1
    all_upsample_ratios = {ps: [] for ps in patch_sizes}
    for file in tqdm.tqdm(files):
        upsample_ratios_per_scale = analyze(file, patch_sizes)
        if i % print_freq == 0:
            print("For file {} upsample ratios are: {}".format(file, upsample_ratios_per_scale))
        for ps in patch_sizes:
            all_upsample_ratios[ps].append(upsample_ratios_per_scale[ps])
        i += 1
    for ps in patch_sizes:
        upsample_ratios = all_upsample_ratios[ps]
        upsample_ratios = np.asarray(upsample_ratios)
        print("Patch_size {}. Mean: {}. Median: {}. Min: {}. Max: {}".format(ps, upsample_ratios.mean(), np.median(upsample_ratios), upsample_ratios.min(), upsample_ratios.max()))
        plt.figure()
        plt.title('Histogram for upsample ratios of {} patch size'.format(ps))
        plt.hist(upsample_ratios, bins=10)
        plt.show()
