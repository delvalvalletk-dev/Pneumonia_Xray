import torch
from PIL import Image

from pathlib import Path

def predict_image(model, image_path, transform, class_names, device):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)       

    model.eval()
    with torch.no_grad():
        logits = model(x)                              
        probs = torch.softmax(logits, dim=1)[0]        

    best = probs.argmax().item()
    print(f"{image_path.name}")
    print(f"{class_names[best]} - {round(probs[best].item()*100,2)}%\n") 

def predict(model, images_path, transform, class_names, device):
    for image_path in images_path.iterdir():
        predict_image(model, image_path, transform, class_names, device) 