# api.py
from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCNN
from classes import CIFAR10_CLASSES

app = FastAPI(title="CNN Classifier API", version="1.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10).to(device)
state = torch.load("checkpoints/cifar10_simplecnn.pt", map_location=device)
model.load_state_dict(state["model_state"])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Body: multipart/form-data，字段名 'file'，值为图片文件
    Return: {"class": str, "prob": float, "index": int}
    """
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        prob = float(probs[idx].item())
    return {"class": CIFAR10_CLASSES[idx], "prob": prob, "index": idx}

# 为了避免顶层导入 io
import io

# 本地启动： uvicorn api:app --host 0.0.0.0 --port 8000
