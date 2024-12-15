import os
import cv2

INPUT_DIR = "datasets/aligned_faces/"
OUTPUT_DIR = "datasets/aligned_faces/aligned/"

def align_and_preprocess():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for image_name in os.listdir(INPUT_DIR):
        img_path = os.path.join(INPUT_DIR, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        resized_img = cv2.resize(img, (256, 256))
        output_path = os.path.join(OUTPUT_DIR, image_name)
        cv2.imwrite(output_path, resized_img)
    print(f"Aligned faces saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    align_and_preprocess()
