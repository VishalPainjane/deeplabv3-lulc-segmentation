import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import timm
import cv2
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- 1. CONFIGURATION: UPDATE THIS SECTION CAREFULLY ---
class CFG:
    # --- Paths to your data and models ---
    # Using the preprocessed data directory as it's common to all
    DATA_DIR = r"SEN-2_LULC_preprocessed" 
    VAL_IMG_DIR = os.path.join(DATA_DIR, "val_images")
    VAL_MASK_DIR = os.path.join(DATA_DIR, "val_masks")
    
    # --- Directory to save all generated plots and results ---
    OUTPUT_DIR = "./evaluation_results2"

    # --- Device Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Model & Data Parameters (Must match training configurations) ---
    IN_CHANNELS = 3
    NUM_CLASSES = 8
    IMG_SIZE = 256
    BATCH_SIZE = 8 # Use a reasonable batch size for evaluation
    NUM_WORKERS = 0

    # --- Define Models to Evaluate ---
    # Each entry is a tuple: (Display Name, Model Type ID, Path to .pth file)
    # The 'Model Type ID' is critical for loading the correct architecture and transforms.
    MODELS_TO_EVALUATE = [
        (
            "U-Net (ResNet34)", 
            "unet_resnet34", 
            "./best_model.pth"
        ),
        (
            "U-Net (SegFormer-B4)", 
            "unet_segformer_b4", 
            "./best_segformer_model.pth"
        ),
        (
            "DeepLabV3+ (EfficientNet-B2)", 
            "deeplabv3p_effnet_b2", 
            r"outputs_rgb_optimized\best_model_optimized_83.98.pth"
        ),
    ]
    
    # --- Visualization Parameters ---
    # Class names for plotting
    CLASS_NAMES = [
        'Urban', 'Shrubland', 'Water', 'Barren', 
        'Cropland', 'Snow/Ice', 'Forest', 'Wetland'
    ]
    # Color palette for segmentation masks (RGB values)
    PALETTE = [
        [255, 0, 0],      # Urban (Red)
        [0, 255, 0],      # Shrubland (Green)
        [0, 0, 255],      # Water (Blue)
        [128, 128, 128],  # Barren (Gray)
        [255, 255, 0],    # Cropland (Yellow)
        [255, 255, 255],  # Snow/Ice (White)
        [0, 128, 0],      # Forest (Dark Green)
        [0, 255, 255],    # Wetland (Cyan)
    ]
    NUM_QUALITATIVE_IMAGES = 10 # Number of example prediction images to save


# --- 2. MODEL DEFINITIONS (Copied from training scripts) ---
# It's necessary to have the class definitions available to load the models.

# From Code 3: CustomDeepLabV3Plus
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CustomDeepLabV3Plus(nn.Module):
    def __init__(self, encoder_name, in_channels, classes):
        super().__init__()
        self.smp_model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights="imagenet", in_channels=in_channels, classes=classes)
        decoder_channels = self.smp_model.segmentation_head[0].in_channels
        self.se_layer = SELayer(decoder_channels)
        self.segmentation_head = self.smp_model.segmentation_head
        self.smp_model.segmentation_head = nn.Identity()
    def forward(self, x):
        decoder_features = self.smp_model(x)
        attended_features = self.se_layer(decoder_features)
        output = self.segmentation_head(attended_features)
        return output

# --- 3. HELPER FUNCTIONS for Loading Models and Data ---

def get_model(model_type, model_path):
    """Loads a model based on the model_type identifier."""
    print(f"Loading model architecture: {model_type}")
    
    if model_type == "unet_resnet34":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None, # Set to None, we are loading our own weights
            in_channels=CFG.IN_CHANNELS,
            classes=CFG.NUM_CLASSES,
        )
    elif model_type == "unet_segformer_b4":
        model = smp.Unet(
            encoder_name="mit_b4",
            encoder_weights=None,
            in_channels=CFG.IN_CHANNELS,
            classes=CFG.NUM_CLASSES,
        )
    elif model_type == "deeplabv3p_effnet_b2":
        model = CustomDeepLabV3Plus(
            encoder_name="timm-efficientnet-b2",
            in_channels=CFG.IN_CHANNELS,
            classes=CFG.NUM_CLASSES
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device(CFG.DEVICE)))
    model.to(CFG.DEVICE)
    model.eval()
    return model

def get_transforms(model_type):
    """Gets the correct validation transforms for a given model type."""
    print(f"Applying transforms for: {model_type}")
    
    # Code 1 and 2 used simple normalization
    if model_type in ["unet_resnet34", "unet_segformer_b4"]:
        return A.Compose([
            A.Resize(height=CFG.IMG_SIZE, width=CFG.IMG_SIZE),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ])
    # Code 3 used ImageNet normalization
    elif model_type == "deeplabv3p_effnet_b2":
        DATASET_MEAN = [0.485, 0.456, 0.406]
        DATASET_STD = [0.229, 0.224, 0.225]
        return A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# Dataset Class (a generic version that works for all)
class LULCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        # Ensure mask extension is handled if different (.tif, .png, etc.)
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.tif'))])
        assert len(self.images) == len(self.masks), "Mismatch in number of images and masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask

# --- 4. PLOTTING FUNCTIONS ---

def plot_iou_barchart(iou_scores, model_name, save_path):
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=CFG.CLASS_NAMES, y=iou_scores, palette="viridis")
    ax.set_title(f'Per-Class IoU Scores for {model_name}', fontsize=16)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Intersection over Union (IoU)', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.bar_label(ax.containers[0], fmt='%.3f')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"IoU bar chart saved to {save_path}")

def plot_confusion_matrix(cm, model_name, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CFG.CLASS_NAMES, yticklabels=CFG.CLASS_NAMES)
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def colorize_mask(mask, palette):
    """Converts a grayscale class mask to a colorized RGB mask."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        color_mask[mask == i] = color
    return color_mask

def save_qualitative_results(model, dataset, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Use a fixed seed for reproducibility of selected images
    random.seed(42)
    indices = random.sample(range(len(dataset)), k=min(CFG.NUM_QUALITATIVE_IMAGES, len(dataset)))
    
    for i, idx in enumerate(tqdm(indices, desc=f"Saving qualitative results for {model_name}")):
        image_tensor, true_mask_tensor = dataset[idx]
        
        with torch.no_grad():
            pred_mask_tensor = model(image_tensor.unsqueeze(0).to(CFG.DEVICE))
            pred_mask_labels = torch.argmax(pred_mask_tensor, dim=1).squeeze(0).cpu().numpy()

        # Denormalize image for visualization
        # This requires knowing the normalization transform. We'll do a simple rescale.
        image = image_tensor.permute(1, 2, 0).numpy()
        image = (image - image.min()) / (image.max() - image.min()) # Simple rescale to 0-1
        image = (image * 255).astype(np.uint8)

        true_mask_labels = true_mask_tensor.numpy().astype(np.uint8)
        
        # Colorize masks
        true_mask_color = colorize_mask(true_mask_labels, CFG.PALETTE)
        pred_mask_color = colorize_mask(pred_mask_labels, CFG.PALETTE)

        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(true_mask_color)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')

        axes[2].imshow(pred_mask_color)
        axes[2].set_title(f'Predicted Mask ({model_name})')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"comparison_{i+1}.png")
        plt.savefig(save_path)
        plt.close()

# --- 5. MAIN EVALUATION LOGIC ---

# NEW, MEMORY-EFFICIENT VERSION
def evaluate_model(model, loader):
    """Calculates metrics for a given model and dataloader using a memory-efficient, batch-wise approach."""
    # 1. Initialize a total confusion matrix with zeros. It's a small 8x8 array.
    total_cm = np.zeros((CFG.NUM_CLASSES, CFG.NUM_CLASSES), dtype=np.int64)
    
    model.eval()
    with torch.no_grad():
        # 2. Loop through each batch
        for x, y in tqdm(loader, desc="Calculating Metrics"):
            x = x.to(CFG.DEVICE)
            # Flatten the ground truth labels for this batch
            y_true_np = y.numpy().flatten()
            
            # Get model predictions and flatten them
            preds = model(x)
            pred_labels = torch.argmax(preds, dim=1).cpu().numpy().flatten()
            
            # 3. Calculate the confusion matrix for just this batch
            batch_cm = confusion_matrix(y_true_np, pred_labels, labels=list(range(CFG.NUM_CLASSES)))
            
            # 4. Add the batch's confusion matrix to the running total
            total_cm += batch_cm

    # 5. After the loop, calculate all metrics from the final total confusion matrix
    cm = total_cm  # Use the same variable name as before for consistency
    
    # Pixel Accuracy
    pixel_accuracy = np.diag(cm).sum() / cm.sum()
    
    # Per-class IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou_per_class = (intersection + 1e-6) / (union + 1e-6)
    
    # Mean IoU
    mean_iou = np.nanmean(iou_per_class)
    
    return pixel_accuracy, mean_iou, iou_per_class, cm


def main():
    print(f"Starting evaluation on device: {CFG.DEVICE}")
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    for model_name, model_type, model_path in CFG.MODELS_TO_EVALUATE:
        print("\n" + "="*50)
        print(f"EVALUATING: {model_name}")
        print("="*50)

        if not os.path.exists(model_path):
            print(f"WARNING: Model file not found at {model_path}. Skipping.")
            continue
            
        # --- 1. Load Model and Data ---
        model = get_model(model_type, model_path)
        val_transform = get_transforms(model_type)
        
        val_dataset = LULCDataset(
            image_dir=CFG.VAL_IMG_DIR,
            mask_dir=CFG.VAL_MASK_DIR,
            transform=val_transform
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CFG.BATCH_SIZE,
            num_workers=CFG.NUM_WORKERS,
            shuffle=False
        )

        # --- 2. Calculate Metrics ---
        pixel_acc, m_iou, iou_scores, conf_matrix = evaluate_model(model, val_loader)

        print(f"\n--- Results for {model_name} ---")
        print(f"Pixel Accuracy: {pixel_acc*100:.2f}%")
        print(f"Mean IoU (mIoU): {m_iou:.4f}")
        for i, iou in enumerate(iou_scores):
            print(f"  - IoU for Class '{CFG.CLASS_NAMES[i]}': {iou:.4f}")

        # --- 3. Generate and Save Plots ---
        model_output_dir = os.path.join(CFG.OUTPUT_DIR, model_name.replace(" ", "_").replace("+", "Plus"))
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Plot IoU scores
        plot_iou_barchart(
            iou_scores, 
            model_name, 
            save_path=os.path.join(model_output_dir, "per_class_iou.png")
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            conf_matrix, 
            model_name, 
            save_path=os.path.join(model_output_dir, "confusion_matrix.png")
        )
        
        # Save qualitative prediction images
        qualitative_dir = os.path.join(model_output_dir, "qualitative_predictions")
        save_qualitative_results(
            model,
            val_dataset, # Pass dataset to access original images
            model_name,
            qualitative_dir
        )
        
        print(f"All plots and results for {model_name} saved in: {model_output_dir}")
        
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()