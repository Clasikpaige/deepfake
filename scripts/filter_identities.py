import os
import shutil

DATASET_DIR = "datasets/celebA/img_align_celeba/"
IDENTITY_FILE = "datasets/celebA/identity_CelebA.txt"
OUTPUT_DIR = "datasets/aligned_faces/"

def filter_identities(target_identities):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(IDENTITY_FILE, "r") as f:
        for line in f:
            image_name, identity = line.strip().split()
            if identity in target_identities:
                src_path = os.path.join(DATASET_DIR, image_name)
                dest_path = os.path.join(OUTPUT_DIR, image_name)
                shutil.copy(src_path, dest_path)
    print(f"Filtered faces saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    target_ids = ["1", "5", "10"]  # Example: Replace with target identity IDs
    filter_identities(target_ids)
