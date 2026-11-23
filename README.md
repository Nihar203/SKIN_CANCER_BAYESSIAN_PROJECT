# Skin Cancer Image Dataset Loading & Preprocessing

A Python/Jupyter Notebook pipeline built on Kaggle for inspecting, loading, preprocessing, and visualizing clinical skin lesion images for downstream machine learning and deep learning tasks.

## Overview

This notebook automates the initial data preparation workflow for a 2,298-image skin cancer dataset:
- **Directory & File Auditing:** Verifies dataset paths and counts image files across nested folders.
- **Image Preprocessing:** Loads all `.png` lesion images with OpenCV, converts color channels from BGR to RGB, resizes images to `224 × 224`, and normalizes pixel values to `[0.0, 1.0]`.
- **Array Construction & Indexing:** Compiles the processed images into a NumPy tensor of shape `(2298, 224, 224, 3)` alongside a parallel `image_ids` list for lookup and metadata alignment.
- **Sample Visualization:** Retrieves images by patient/lesion ID (e.g., `PAT_492_937_958`) and displays them using Matplotlib.

## Dataset Structure

The notebook expects the dataset in the following directory layout (as mounted on Kaggle under `/kaggle/input/datasets/niharkumardash/skin-cancer-dataset`):

```text
skin-cancer-dataset/
├── metadata.csv
├── imgs_part_1/
│   └── imgs_part_1/        # 911 PNG images
├── imgs_part_2/
│   └── imgs_part_2/        # 659 PNG images
└── imgs_part_3/
    └── imgs_part_3/        # 728 PNG images

