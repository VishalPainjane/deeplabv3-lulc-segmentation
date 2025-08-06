import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
from tqdm.notebook import tqdm
import segmentation_models_pytorch as smp
import json
import matplotlib.pyplot as plt

class CFG:
    DATA_DIR = r"/workspace/Vishal_Painjane_23bcs267/SEN2_LULC_data/SEN-2_LULC_preprocessed"
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
    TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train_masks")
    VAL_IMG_DIR = os.path.join(DATA_DIR, "val_images")
    VAL_MASK_DIR = os.path.join(DATA_DIR, "val_masks")
    
    OUTPUT_DIR = "./outputs_rgb_optimized"
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model_optimized.pth")
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")
    HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.json") 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "CustomDeepLabV3+"
    ENCODER_NAME = "timm-efficientnet-b2"
    LOSS_FN_NAME = "DiceFocal"
    IN_CHANNELS = 3; NUM_CLASSES = 8; IMG_SIZE = 256
    BATCH_SIZE = 64; ACCUMULATION_STEPS = 1
    NUM_WORKERS = 8
    LEARNING_RATE = 1e-4; EPOCHS = 50
    SEED = 42
    SUBSET_FRACTION = 1

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__(); self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size(); y = self.avg_pool(x).view(b, c); y = self.fc(y).view(b, c, 1, 1); return x * y.expand_as(x)
class CustomDeepLabV3Plus(nn.Module):
    def __init__(self, encoder_name, in_channels, classes):
        super().__init__(); self.smp_model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights="imagenet", in_channels=in_channels, classes=classes)
        decoder_channels = self.smp_model.segmentation_head[0].in_channels; self.se_layer = SELayer(decoder_channels)
        self.segmentation_head = self.smp_model.segmentation_head; self.smp_model.segmentation_head = nn.Identity()
    def forward(self, x):
        decoder_features = self.smp_model(x); attended_features = self.se_layer(decoder_features)
        output = self.segmentation_head(attended_features); return output
class CombinedLoss(nn.Module):
    def __init__(self, loss1, loss2, alpha=0.5):
        super(CombinedLoss, self).__init__(); self.loss1 = loss1; self.loss2 = loss2; self.alpha = alpha
        self.name = f"{alpha}*{self.loss1.__class__.__name__} + {1-alpha}*{self.loss2.__class__.__name__}"
    def forward(self, prediction, target):
        loss1_val = self.loss1(prediction, target); loss2_val = self.loss2(prediction, target); return self.alpha * loss1_val + (1 - self.alpha) * loss2_val

class LULCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, subset_fraction=1.0, is_train=False, img_size=256):
        self.image_dir = image_dir; self.mask_dir = mask_dir; self.is_train = is_train; self.img_size = img_size
        all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        all_masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
        num_samples = int(len(all_images) * subset_fraction)
        self.images = all_images[:num_samples]; self.masks = all_masks[:num_samples]
        assert len(self.images) == len(self.masks), "Mismatch"; print(f"Found {len(all_images)} total images, USING {len(self.images)} samples ({subset_fraction*100:.2f}%) from {image_dir}")
        self.normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx]); mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB"); mask = Image.open(mask_path).convert("L")
        image = F.resize(image, [self.img_size, self.img_size], interpolation=T.InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.img_size, self.img_size], interpolation=T.InterpolationMode.NEAREST)
        if self.is_train:
            if random.random() > 0.5: image = F.hflip(image); mask = F.hflip(mask)
            if random.random() > 0.5: image = F.vflip(image); mask = F.vflip(mask)
            angle = T.RandomRotation.get_params([-35, 35])
            image = F.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
            mask = F.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
        image = F.to_tensor(image); mask = torch.from_numpy(np.array(mask, dtype=np.int64)); image = self.normalize_transform(image)
        return image, mask

def get_model():
    if CFG.MODEL_NAME == "CustomDeepLabV3+": model = CustomDeepLabV3Plus(encoder_name=CFG.ENCODER_NAME, in_channels=CFG.IN_CHANNELS, classes=CFG.NUM_CLASSES)
    else: model = smp.DeepLabV3Plus(encoder_name=CFG.ENCODER_NAME, encoder_weights="imagenet", in_channels=CFG.IN_CHANNELS, classes=CFG.NUM_CLASSES)
    return model.to(CFG.DEVICE)
def get_loss_fn():
    if CFG.LOSS_FN_NAME == "DiceFocal": dice = smp.losses.DiceLoss(mode='multiclass'); focal = smp.losses.FocalLoss(mode='multiclass'); return CombinedLoss(focal, dice, alpha=0.5)
    else: return smp.losses.DiceLoss(mode='multiclass')

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Training")
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(CFG.DEVICE, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.long().to(CFG.DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=CFG.DEVICE, dtype=torch.bfloat16, enabled=(CFG.DEVICE=="cuda")):
            predictions = model(data)
            loss = loss_fn(predictions, targets) / CFG.ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        if (batch_idx + 1) % CFG.ACCUMULATION_STEPS == 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        
        current_loss = loss.item() * CFG.ACCUMULATION_STEPS
        total_loss += current_loss
        loop.set_postfix(loss=current_loss)
    
    return total_loss / len(loader) 

def evaluate_model(loader, model, loss_fn):
    model.eval()
    intersection, union = torch.zeros(CFG.NUM_CLASSES, device=CFG.DEVICE), torch.zeros(CFG.NUM_CLASSES, device=CFG.DEVICE)
    pixel_correct, pixel_total, total_loss = 0, 0, 0
    loop = tqdm(loader, desc="Evaluating")

    with torch.no_grad():
        for x, y in loop:
            x = x.to(CFG.DEVICE, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(CFG.DEVICE, non_blocking=True).long()
            with torch.amp.autocast(device_type=CFG.DEVICE, dtype=torch.bfloat16, enabled=(CFG.DEVICE=="cuda")):
                preds = model(x)
                loss = loss_fn(preds, y)
                total_loss += loss.item()
            pred_labels = torch.argmax(preds, dim=1)
            pixel_correct += (pred_labels == y).sum()
            pixel_total += torch.numel(y)
            for cls in range(CFG.NUM_CLASSES):
                pred_mask = (pred_labels == cls); true_mask = (y == cls)
                intersection[cls] += (pred_mask & true_mask).sum()
                union[cls] += (pred_mask | true_mask).sum()

    pixel_acc = (pixel_correct / pixel_total) * 100
    iou_per_class = (intersection + 1e-6) / (union + 1e-6)
    mean_iou = iou_per_class.mean()
    avg_loss = total_loss / len(loader)
    
    print(f"Validation Results -> Avg Loss: {avg_loss:.4f}, Pixel Acc: {pixel_acc:.2f}%, mIoU: {mean_iou:.4f}")
    
    return avg_loss, mean_iou, pixel_acc.cpu().item()

def plot_training_history(history, save_dir):
    """Plots and saves the training history graphs."""
#     plt.style.use("seaborn-v0_8-whitegrid")
    plt.style.use("ggplot")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    ax.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    ax.set_title('Training and Validation Loss', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend()
    loss_plot_path = os.path.join(save_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path, dpi=300)
    print(f"Loss plot saved to {loss_plot_path}")
    plt.close()

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.plot(epochs, history['val_miou'], 'g-o', label='Validation mIoU')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('mIoU', color='g', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='g')
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['val_acc'], 'm-s', label='Validation Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='m', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='m')
    
    ax1.set_title('Validation Metrics (mIoU and Accuracy)', fontsize=16)
    fig.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    metrics_plot_path = os.path.join(save_dir, 'metrics_plot.png')
    plt.savefig(metrics_plot_path, dpi=300)
    print(f"Metrics plot saved to {metrics_plot_path}")
    plt.close()

def save_checkpoint(state, filename="checkpoint.pth"): print("=> Saving checkpoint"); torch.save(state, filename)

def main():
    torch.manual_seed(CFG.SEED); np.random.seed(CFG.SEED); os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True
    
    train_ds = LULCDataset(CFG.TRAIN_IMG_DIR, CFG.TRAIN_MASK_DIR, subset_fraction=CFG.SUBSET_FRACTION, is_train=True, img_size=CFG.IMG_SIZE)
    val_ds = LULCDataset(CFG.VAL_IMG_DIR, CFG.VAL_MASK_DIR, subset_fraction=CFG.SUBSET_FRACTION, is_train=False, img_size=CFG.IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, num_workers=CFG.NUM_WORKERS, pin_memory=True, shuffle=False)

    model = get_model(); model = model.to(memory_format=torch.channels_last)
    loss_fn = get_loss_fn(); optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=(CFG.DEVICE=="cuda")); scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)

    start_epoch = 0; best_val_miou = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_miou': [], 'val_acc': []}

    if os.path.exists(CFG.CHECKPOINT_PATH):
        print("=> Loading checkpoint...")

    for epoch in range(start_epoch, CFG.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{CFG.EPOCHS} ---")
        
        avg_train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        avg_val_loss, current_miou, current_acc = evaluate_model(val_loader, model, loss_fn)
        scheduler.step()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_miou'].append(current_miou.cpu().item()) 
        history['val_acc'].append(current_acc)

        if current_miou > best_val_miou:
            best_val_miou = current_miou
            print(f"New best mIoU: {best_val_miou:.4f}! Saving best model to {CFG.MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), CFG.MODEL_SAVE_PATH)
        
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'best_val_miou': best_val_miou}
        save_checkpoint(checkpoint, filename=CFG.CHECKPOINT_PATH)
        
    print("\n--- Training Complete. ---")
    
    print("Saving training history...")
    with open(CFG.HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"History saved to {CFG.HISTORY_PATH}")

    print("\nGenerating plots...")
    plot_training_history(history, CFG.OUTPUT_DIR)

if __name__ == "__main__":
    main()