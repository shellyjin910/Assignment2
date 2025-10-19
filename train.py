# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN

def get_loaders(batch_size=128, num_workers=2):
    # 题面要求输入 64x64，所以这里把 CIFAR-10 从 32x32 Resize 到 64x64
    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])

    root = "./data"
    train_set = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

def train(epochs=10, lr=1e-3, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(batch_size=batch_size)

    model = SimpleCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/cifar10_simplecnn.pt"

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

            if i % 100 == 0:
                print(f"[Epoch {ep} | Step {i}] loss={running/100:.4f}")
                running = 0.0

        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"Epoch {ep}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "acc": best_acc,
            }, ckpt_path)
            print(f"✔ Saved best to {ckpt_path} (acc={best_acc:.4f})")

    print("Training finished. Best acc:", best_acc)

if __name__ == "__main__":
    # 可用命令行参数也行，这里给默认参数
    train(epochs=10, lr=1e-3, batch_size=128)
