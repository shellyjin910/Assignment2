# infer.py
import io
import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCNN
from classes import CIFAR10_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

def load_model(ckpt_path="checkpoints/cifar10_simplecnn.pt"):
    model = SimpleCNN(num_classes=10).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model

@torch.no_grad()
def predict_image(img_path: str, ckpt_path="checkpoints/cifar10_simplecnn.pt"):
    model = load_model(ckpt_path)
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return CIFAR10_CLASSES[idx], float(probs[idx].item())

if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    label, prob = predict_image(p)
    print(f"Pred: {label} (p={prob:.3f})")
