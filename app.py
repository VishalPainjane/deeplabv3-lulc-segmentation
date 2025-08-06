import os
import torch
import gradio as gr
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

class CFG:
    MODEL_PATH = r"outputs_rgb_optimized\best_model_optimized_83.98.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    MODEL_NAME = "CustomDeepLabV3+"
    ENCODER_NAME = "timm-efficientnet-b2" 
    
    NUM_CLASSES = 8
    IMG_SIZE = 256
    
    COLOR_MAP = np.array([
        [0, 0, 128],    # Class 0: Water (Navy)
        [0, 128, 0],    # Class 1: Dense Forest (Green)
        [152, 251, 152],# Class 2: Sparse Forest (Pale Green)
        [139, 69, 19],  # Class 3: Barren Land (Brown)
        [128, 128, 128],# Class 4: Built-up (Gray)
        [255, 255, 0],  # Class 5: Agriculture (Yellow)
        [244, 164, 96], # Class 6: Fallow Land (Sandy Brown)
        [0, 0, 0]       # Class 7: Other/Background (Black)
    ], dtype=np.uint8)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CustomDeepLabV3Plus(nn.Module):
    def __init__(self, encoder_name, in_channels, classes):
        super().__init__()
        self.smp_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name, encoder_weights="imagenet",
            in_channels=in_channels, classes=classes
        )
        decoder_channels = self.smp_model.segmentation_head[0].in_channels
        self.se_layer = SELayer(decoder_channels)
        self.segmentation_head = self.smp_model.segmentation_head
        self.smp_model.segmentation_head = nn.Identity()

    def forward(self, x):
        decoder_features = self.smp_model(x)
        attended_features = self.se_layer(decoder_features)
        output = self.segmentation_head(attended_features)
        return output

print(f"Loading model from {CFG.MODEL_PATH} on device {CFG.DEVICE}...")

model = CustomDeepLabV3Plus(
    encoder_name=CFG.ENCODER_NAME,
    in_channels=3,
    classes=CFG.NUM_CLASSES,
).to(CFG.DEVICE)

try:
    model.load_state_dict(torch.load(CFG.MODEL_PATH, map_location=CFG.DEVICE, weights_only=True))
except TypeError:
    model.load_state_dict(torch.load(CFG.MODEL_PATH, map_location=CFG.DEVICE))

model.eval()
print("Model loaded successfully!")

val_transform = A.Compose([
    A.Resize(height=CFG.IMG_SIZE, width=CFG.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def predict(input_image):
    """
    Takes a PIL image, runs inference, and returns the colorized mask.
    """
    if input_image is None:
        return None

    image_np = np.array(input_image)
    
    transformed = val_transform(image=image_np)
    input_tensor = transformed['image'].unsqueeze(0).to(CFG.DEVICE)
    
    if CFG.DEVICE == "cuda":
        input_tensor = input_tensor.to(memory_format=torch.channels_last)

    with torch.no_grad():
        prediction = model(input_tensor)

    pred_mask = torch.argmax(prediction.squeeze(), dim=0).cpu().numpy()
    color_mask = CFG.COLOR_MAP[pred_mask]
    
    return Image.fromarray(color_mask)

if not os.path.exists("examples"):
    os.makedirs("examples")
    print("Created 'examples' folder. Please add some test images to it.")

example_list = [os.path.join("examples", filename) for filename in os.listdir("examples")]

with gr.Blocks(theme=gr.themes.Soft(), title="LULC Segmentation") as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'>Land Use / Land Cover Segmentation</h1>"
        "<p style='text-align: center;'>An application to demonstrate a DeepLabV3+ model with an EfficientNet-B2 encoder. "
        "Upload an image or use an example to see the model predict 8 different land cover classes.</p>"
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            submit_btn = gr.Button("Segment Image", variant="primary")
            
            legend_html = """
            <div style='padding: 10px; border: 1px solid #e0e0e0; border-radius: 8px;'>
                <h4 style='margin-top:0; margin-bottom:10px; text-align:center;'>Prediction Legend</h4>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                    <div style='display: flex; align-items: center;'><div style='width: 20px; height: 20px; background-color: #000080; border-radius: 3px;'></div><span style='margin-left: 10px;'>Water</span></div>
                    <div style='display: flex; align-items: center;'><div style='width: 20px; height: 20px; background-color: #008000; border-radius: 3px;'></div><span style='margin-left: 10px;'>Dense Forest</span></div>
                    <div style='display: flex; align-items: center;'><div style='width: 20px; height: 20px; background-color: #98FB98; border-radius: 3px;'></div><span style='margin-left: 10px;'>Sparse Forest</span></div>
                    <div style='display: flex; align-items: center;'><div style='width: 20px; height: 20px; background-color: #8B4513; border-radius: 3px;'></div><span style='margin-left: 10px;'>Barren Land</span></div>
                    <div style='display: flex; align-items: center;'><div style='width: 20px; height: 20px; background-color: #808080; border-radius: 3px;'></div><span style='margin-left: 10px;'>Built-up</span></div>
                    <div style='display: flex; align-items: center;'><div style='width: 20px; height: 20px; background-color: #FFFF00; border-radius: 3px;'></div><span style='margin-left: 10px;'>Agriculture</span></div>
                    <div style='display: flex; align-items: center;'><div style='width: 20px; height: 20px; background-color: #F4A460; border-radius: 3px;'></div><span style='margin-left: 10px;'>Fallow Land</span></div>
                    <div style='display: flex; align-items: center;'><div style='width: 20px; height: 20px; background-color: #000000; border-radius: 3px;'></div><span style='margin-left: 10px;'>Other</span></div>
                </div>
            </div>
            """
            gr.HTML(legend_html)
            
        with gr.Column():
            output_mask = gr.Image(type="pil", label="Predicted Land Cover Map")

    gr.Examples(
        examples=example_list,
        inputs=input_image,
        outputs=output_mask,
        fn=predict,
        cache_examples=False 
    )
            
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=output_mask
    )

if __name__ == "__main__":
    demo.launch(share=False)
