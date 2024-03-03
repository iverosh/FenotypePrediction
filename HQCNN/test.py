import torch
from torchvision.io import read_image
from HybridQNN import HybridQNN
model = HybridQNN()

model.load_state_dict(torch.load("HQCNN\\HQCNN.pt"))
model.eval()
img = read_image("HQCNN/image.png").to(torch.float).unsqueeze(0)
print(model)
print(model(img))