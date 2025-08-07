import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

class CFG:
    """Configuration for the LULC Analysis Platform."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = r"outputs_rgb_optimized\best_model_optimized_83.98.pth"
    MODEL_NAME = "CustomDeepLabV3+"
    ENCODER_NAME = "timm-efficientnet-b2"
    NUM_CLASSES = 8
    IMG_SIZE = 256

CLASS_INFO = {
    0: {"name": "Unclassified", "hex": "#969696"},
    1: {"name": "Water Bodies", "hex": "#0000FF"},
    2: {"name": "Dense Forest", "hex": "#006400"},
    3: {"name": "Built up", "hex": "#800080"},
    4: {"name": "Agriculture land", "hex": "#00FF00"},
    5: {"name": "Barren land", "hex": "#FFFF00"},
    6: {"name": "Fallow land", "hex": "#D2B48C"},
    7: {"name": "Sparse Forest", "hex": "#3CB371"},
}

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

def load_model():
    print(f"Loading model from {CFG.MODEL_PATH} on device {CFG.DEVICE}...")
    model = CustomDeepLabV3Plus(encoder_name=CFG.ENCODER_NAME, in_channels=3, classes=CFG.NUM_CLASSES)
    if not os.path.exists(CFG.MODEL_PATH): raise FileNotFoundError(f"Model file not found at {CFG.MODEL_PATH}.")
    model.load_state_dict(torch.load(CFG.MODEL_PATH, map_location=torch.device(CFG.DEVICE), weights_only=True))
    model.to(CFG.DEVICE); model.eval()
    print("Model loaded successfully!")
    return model

model = load_model()
transform = A.Compose([
    A.Resize(height=CFG.IMG_SIZE, width=CFG.IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def create_color_map():
    color_map = np.zeros((256, 3), dtype=np.uint8)
    for class_id, info in CLASS_INFO.items():
        color_map[class_id] = tuple(int(info['hex'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    return color_map
COLOR_MAP_NUMPY = create_color_map()

def analyze_one_image(image: Image.Image):
    if image is None: return None, {}
    image_np = np.array(image.convert("RGB"))
    transformed = transform(image=image_np)
    input_tensor = transformed['image'].unsqueeze(0).to(CFG.DEVICE)
    with torch.no_grad():
        prediction = model(input_tensor)
    pred_mask = torch.argmax(prediction.squeeze(), dim=0).cpu().numpy()
    
    area_results = {}
    class_indices, pixel_counts = np.unique(pred_mask, return_counts=True)
    PIXEL_AREA_SQ_METERS = 10 * 10
    SQ_METERS_PER_HECTARE = 10000
    for class_id, count in zip(class_indices, pixel_counts):
        if class_id in CLASS_INFO:
            area_hectares = (count * PIXEL_AREA_SQ_METERS) / SQ_METERS_PER_HECTARE
            area_results[CLASS_INFO[class_id]["name"]] = area_hectares
    return pred_mask, area_results

def single_image_analysis(image: Image.Image):
    if image is None: raise gr.Error("Please upload an image to analyze.")
    mask_np, areas_dict = analyze_one_image(image)
    mask_pil = Image.fromarray(COLOR_MAP_NUMPY[mask_np])
    formatted_areas = {k: f"{v:.4f} Hectares" for k, v in areas_dict.items()}
    return mask_pil, formatted_areas

def compare_land_cover(image1: Image.Image, image2: Image.Image):
    if image1 is None or image2 is None:
        raise gr.Error("Please upload both a 'Before' and 'After' image for comparison.")

    mask1_np, areas1_dict = analyze_one_image(image1)
    mask2_np, areas2_dict = analyze_one_image(image2)
    
    mask1_pil = Image.fromarray(COLOR_MAP_NUMPY[mask1_np])
    mask2_pil = Image.fromarray(COLOR_MAP_NUMPY[mask2_np])
    
    all_class_names = sorted(list(set(areas1_dict.keys()) | set(areas2_dict.keys())))
    data_for_df = [[name, areas1_dict.get(name, 0), areas2_dict.get(name, 0)] for name in all_class_names]
    df = pd.DataFrame(data_for_df, columns=["Class", "Area 1 (ha)", "Area 2 (ha)"])
    df['Change (ha)'] = df['Area 2 (ha)'] - df['Area 1 (ha)']
    df['% Change'] = df.apply(lambda row: (row['Change (ha)'] / row['Area 1 (ha)'] * 100) if row['Area 1 (ha)'] > 0 else float('inf'), axis=1)

    df_display = df.copy()
    for col in ["Area 1 (ha)", "Area 2 (ha)"]: df_display[col] = df_display[col].map('{:.2f}'.format)
    df_display["Change (ha)"] = df_display["Change (ha)"].map('{:+.2f}'.format)
    df_display["% Change"] = df_display["% Change"].apply(lambda x: f"{x:+.2f}%" if x != float('inf') else "New")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6)); index = np.arange(len(df)); bar_width = 0.35
    ax.bar(index - bar_width/2, df['Area 1 (ha)'], bar_width, label='Area 1 (Before)', color='cornflowerblue')
    ax.bar(index + bar_width/2, df['Area 2 (ha)'], bar_width, label='Area 2 (After)', color='salmon')
    ax.set_xlabel('Land Cover Class', fontweight='bold'); ax.set_ylabel('Area (Hectares)', fontweight='bold')
    ax.set_title('Land Cover Change Analysis', fontsize=16, fontweight='bold')
    ax.set_xticks(index); ax.set_xticklabels(df['Class'], rotation=45, ha="right"); ax.legend(); fig.tight_layout()

    total_area = df['Area 1 (ha)'].sum()
    summary = f"Change analysis for a total area of approximately {total_area:.2f} hectares:\n\n"
    df_sorted = df.reindex(df['Change (ha)'].abs().sort_values(ascending=False).index)
    for _, row in df_sorted.head(3).iterrows():
        if abs(row['Change (ha)']) > 0.01:
            direction = "increased" if row['Change (ha)'] > 0 else "decreased"
            summary += f"- **{row['Class']}** has {direction} by **{abs(row['Change (ha)']):.2f} hectares** ({row['% Change']}).\n"
            
    if all(abs(df['Change (ha)']) < 0.01):
        summary += "No significant changes were detected between the two images."

    return mask1_pil, mask2_pil, df_display, fig, summary

example_folder = "examples"
example_list = [os.path.join(example_folder, f) for f in os.listdir(example_folder)] if os.path.exists(example_folder) else []

with gr.Blocks(theme=gr.themes.Soft(), title="LULC Analysis Platform") as demo:
    gr.Markdown("# Land Use & Land Cover (LULC) Analysis Platform")
    gr.Markdown("Welcome! This tool leverages a Deep Learning model to analyze satellite imagery. Choose a feature below.")
    
    with gr.Tabs():
        with gr.TabItem("Single Image Analysis"):
            gr.Markdown("### **Usability: Get a Quick Snapshot**")
            gr.Markdown("Use this to quickly assess a site, generate a baseline report, or quantify the land cover types in an image patch.")
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    single_img_input = gr.Image(type="pil", label="Upload Satellite Image")
                    single_analyze_btn = gr.Button("Analyze Image", variant="primary")
                with gr.Column(scale=1):
                    single_mask_output = gr.Image(type="pil", label="Predicted Land Cover Mask")
                    single_json_output = gr.JSON(label="Land Cover Area Distribution")
            gr.Examples(examples=example_list, inputs=single_img_input, label="Click an Example to Start")
        
        with gr.TabItem("Change Detection Tool"):
            gr.Markdown("### **Usability: Monitor and Report Changes Over Time**")
            gr.Markdown("Ideal for monitoring deforestation, urban sprawl, or disaster impact by comparing two images.")
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    gr.Markdown("**Step 1: Upload Images**")
                    compare_img1 = gr.Image(type="pil", label="Image 1 (e.g., Before / 2020)")
                    compare_img2 = gr.Image(type="pil", label="Image 2 (e.g., After / 2024)")
                    compare_analyze_btn = gr.Button("Analyze Changes", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("**Step 2: View Segmentation Masks**")
                    compare_mask1 = gr.Image(type="pil", label="Mask for Image 1")
                    compare_mask2 = gr.Image(type="pil", label="Mask for Image 2")
            
            gr.Markdown("### **Step 3: Review Analysis**")
            with gr.Tabs():
                with gr.TabItem("Change Chart"):
                    compare_plot = gr.Plot(label="Land Cover Change Visualization")
                with gr.TabItem("Comparison Table"):
                    compare_df = gr.DataFrame(label="Area Comparison (Hectares)", interactive=False)
                with gr.TabItem("Key Summary"):
                    compare_summary = gr.Textbox(label="Summary of Key Changes", lines=4)
            gr.Examples(examples=example_list, inputs=compare_img1, label="Click an Example for the 'Before' Image")

    single_analyze_btn.click(fn=single_image_analysis, inputs=single_img_input, outputs=[single_mask_output, single_json_output])
    compare_analyze_btn.click(fn=compare_land_cover, inputs=[compare_img1, compare_img2], outputs=[compare_mask1, compare_mask2, compare_df, compare_plot, compare_summary])

if __name__ == "__main__":
    demo.launch(debug=True)