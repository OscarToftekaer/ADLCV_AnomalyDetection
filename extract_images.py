import pandas as pd
import os
import shutil
from pathlib import Path

df = pd.read_csv('./che_pm_shortcut_labels.csv')

# Filtering and sampling for each list
def sample_and_remove(df, query, n_samples):
    subset = df.query(query).sample(n=n_samples, random_state=42)
    df = df.drop(subset.index)  # Remove the sampled entries to ensure uniqueness
    return subset['img_name'].tolist(), df

# queries and required counts
queries_counts = [
    ("PM == 1 and Cardiomegaly == 1", 600),   # cla_PMill
    ("PM == 0 and Cardiomegaly == 0", 600),   # cla_NPMhealthy
    ("PM == 0 and Cardiomegaly == 1", 5548),  # dif_NPMill
    ("PM == 0 and Cardiomegaly == 0", 5548),  # dif_NPMhealthy
    ("PM == 1 and Cardiomegaly == 1", 369),   # inf_PMill
    ("PM == 1 and Cardiomegaly == 0", 369)    # inf_PMhealthy
]

# Apply the sampling function for each condition
lists = []  # This will store the lists of img_names
for query, count in queries_counts:
    sampled_list, df = sample_and_remove(df, query, count)
    lists.append(sampled_list)

# Access each list
cla_PMill, cla_NPMhealthy, dif_NPMill, dif_NPMhealthy, inf_PMill, inf_PMhealthy = lists

# Base directory for the new splits
base_dir = '/path/to/classification_split'
os.makedirs(os.path.join(base_dir, 'cla_PMill'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'cla_NPMhealthy'), exist_ok=True)

base_dir = '/path/to/diffusion_split'
os.makedirs(os.path.join(base_dir, 'dif_NPMill'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'dif_NPMhealthy'), exist_ok=True)

base_dir = '/path/to/inference_split'
os.makedirs(os.path.join(base_dir, 'inf_PMill'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'inf_PMhealthy'), exist_ok=True)


def copy_images(image_list, source_dirs, dest_dir):
    """
    Copy images from source directories to destination directory.

    Parameters:
    - image_list: List of image filenames.
    - source_dirs: List of directories to search for images.
    - dest_dir: Destination directory to copy images to.
    """
    for img_name in image_list:
        # Checking entire dataset (train + validation)
        found = False
        for base_path in source_dirs:
            source_path = os.path.join(base_path, img_name)
            if Path(source_path).exists():
                shutil.copy(source_path, dest_dir)
                found = True
                break
        if not found:
            print(f"Image not found: {img_name}")

# Base paths to catch all data
train_path = '/work3/s194632/train'
valid_path = '/work3/s194632/valid'

# run the following
copy_images(cla_PMill, [train_path, valid_path], '/path/to/classification_split/cla_PMill')
copy_images(cla_NPMhealthy, [train_path, valid_path], '/path/to/classification_split/cla_NPMhealthy')
copy_images(dif_NPMill, [train_path, valid_path], '/path/to/diffusion_split/dif_NPMill')
copy_images(dif_NPMhealthy, [train_path, valid_path], '/path/to/diffusion_split/dif_NPMhealthy')
copy_images(inf_PMill, [train_path, valid_path], '/path/to/inference_split/inf_PMill')
copy_images(inf_PMhealthy, [train_path, valid_path], '/path/to/inference_split/inf_PMhealthy')