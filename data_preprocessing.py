import os
from PIL import Image
from tqdm import tqdm
import glob

SOURCE_ROOT = "SEN-2 LULC"
DEST_ROOT = "SEN-2_LULC_preprocessed"
IMG_SIZE = 256

DIRS = {
    "train_images": os.path.join(SOURCE_ROOT, "train_images", "train"),
    "train_masks": os.path.join(SOURCE_ROOT, "train_masks", "train"),
    "val_images": os.path.join(SOURCE_ROOT, "val_images", "val"),
    "val_masks": os.path.join(SOURCE_ROOT, "val_masks", "val"),
}

def preprocess_images():
    """Resizes all images and masks and saves them to a new directory."""
    for key, source_dir in DIRS.items():
        dest_dir = os.path.join(DEST_ROOT, key)
        os.makedirs(dest_dir, exist_ok=True)
        
        files = glob.glob(os.path.join(source_dir, "*.*"))
        print(f"\nProcessing {len(files)} files in: {source_dir}")

        for f_path in tqdm(files, desc=f"Resizing {key}"):
            try:
                resample_method = Image.Resampling.NEAREST if "mask" in key else Image.Resampling.LANCZOS
                
                with Image.open(f_path) as img:
                    resized_img = img.resize((IMG_SIZE, IMG_SIZE), resample=resample_method)
                    resized_img.save(os.path.join(dest_dir, os.path.basename(f_path)))
            except Exception as e:
                print(f"Error processing {f_path}: {e}")

    print(f"\nPreprocessing complete! Resized data is in '{DEST_ROOT}' 🚀")

if __name__ == "__main__":
    preprocess_images()