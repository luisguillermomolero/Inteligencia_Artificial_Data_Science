import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from datetime import datetime

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = "exercise1_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 128
LATENT_DIM = 64
EPOCHS = 10
LR = 2e-4  # 0.0002

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False)

class Generator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.net(z)
        return x.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.net(img)

G = Generator(LATENT_DIM).to(DEVICE)
D = Discriminator().to(DEVICE)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0)
        
G.apply(weights_init)
D.apply(weights_init)

if __name__ == "__main__":
    
    fixed_noise = torch.randn(36, LATENT_DIM, device=DEVICE)

    for epoch in range(1, EPOCHS + 1):
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            bs = imgs.size(0)
            
            real_labels = torch.ones(bs, 1, device=DEVICE)
            fake_labels = torch.zeros(bs, 1, device=DEVICE)
            
            opt_D.zero_grad()
            outputs_real = D(imgs)
            loss_real = criterion(outputs_real, real_labels)
            
            noise = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fake_imgs = G(noise)
            
            outputs_fake = D(fake_imgs.detach())
            loss_fake = criterion(outputs_fake, fake_labels)
            
            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()
            
            opt_G.zero_grad()
            noise = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fake_imgs = G(noise)
            
            outputs = D(fake_imgs)
            loss_G = criterion(outputs, real_labels)
            loss_G.backward()
            opt_G.step()
            
            if (i + 1) % 200 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Step [{i+1}/{len(loader)}] "
                    f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}"
                    )
        with torch.no_grad():
            samples = G(fixed_noise).cpu()
            grid = make_grid(samples, nrow=6, normalize=True, value_range=(-1, 1))
            save_image(grid, os.path.join(OUT_DIR, f"epoch_{epoch:02d}.png"))        
            
        torch.save(G.state_dict(), os.path.join(OUT_DIR, f"G_epoch_{epoch}.pth"))
        torch.save(D.state_dict(), os.path.join(OUT_DIR, f"D_epoch_{epoch}.pth"))

    print("Entrenamiento finalizado. Imágenes y chekpoints en: ", OUT_DIR)

    img = plt.imread(os.path.join(OUT_DIR, f"epoch_{EPOCHS:02d}.png"))
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Ejemplo despues de la epoca {EPOCHS}")
    plt.show()