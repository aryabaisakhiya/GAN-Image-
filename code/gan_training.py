# gan_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Generator
class Generator(nn.Module):
    def __init__(self, input_size=100, output_size=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Save generated images
def save_samples(generator, epoch, folder="outputs", z_dim=100, device="cpu"):
    generator.eval()
    z = torch.randn(25, z_dim).to(device)
    with torch.no_grad():
        generated = generator(z).reshape(-1, 1, 28, 28)
    generator.train()

    grid = torch.cat([img for img in generated], dim=2).squeeze().cpu().numpy()
    plt.figure(figsize=(10, 2))
    for i in range(25):
        plt.subplot(2, 13, i + 1)
        plt.imshow(generated[i].squeeze().cpu().numpy(), cmap="gray")
        plt.axis("off")
    os.makedirs(folder, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{folder}/epoch_{epoch}.png")
    plt.close()

# Training loop
def train():
    batch_size = 64
    z_dim = 100
    epochs = 50
    lr = 0.0002
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator(z_dim).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real, _ in loader:
            real = real.view(-1, 784).to(device)
            batch_size = real.size(0)

            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            z = torch.randn(batch_size, z_dim).to(device)
            fake = G(z)
            D_loss_real = criterion(D(real), real_labels)
            D_loss_fake = criterion(D(fake.detach()), fake_labels)
            D_loss = D_loss_real + D_loss_fake

            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            # Train Generator
            z = torch.randn(batch_size, z_dim).to(device)
            fake = G(z)
            G_loss = criterion(D(fake), real_labels)

            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {D_loss.item():.4f}, G_loss: {G_loss.item():.4f}")
        save_samples(G, epoch, device=device)

if __name__ == "__main__":
    train()
