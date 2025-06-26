import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os

MODEL_PATH = "model/mnist_cnn.pt"

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_and_save_model():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    model = DigitCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()

    model.train()
    for epoch in range(1):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model saved to", MODEL_PATH)

def load_model():
    model = DigitCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

# 👇 Add this
if __name__ == "__main__":
    train_and_save_model()
