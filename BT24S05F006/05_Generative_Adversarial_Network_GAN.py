"""
PRACTICAL 5: GENERATIVE ADVERSARIAL NETWORKS (GANs)
Roll No.: BT24S05F006

AIM:
To study the architecture of Generative Adversarial Networks (GANs)
and implement a simple GAN to generate synthetic data.

THEORY:
A GAN has two neural networks:
1. Generator - creates fake/synthetic samples from random noise.
2. Discriminator - distinguishes real samples from generated samples.

The Generator tries to fool the Discriminator, while the Discriminator
tries to correctly classify real and fake samples. Both improve through
adversarial training.

This practical implements a lightweight 2D GAN using PyTorch. The real
data is a mixture of two Gaussian clusters, and the GAN learns its
distribution and generates new synthetic samples.

INSTALLATION:
pip install numpy matplotlib torch
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cpu")


def create_real_data(n_samples=2000):
    """Create a 2D mixture of two Gaussian clusters."""
    n1 = n_samples // 2
    n2 = n_samples - n1

    cluster1 = np.random.normal([-2.0, -2.0], 0.55, (n1, 2))
    cluster2 = np.random.normal([2.0, 2.0], 0.55, (n2, 2))

    data = np.vstack([cluster1, cluster2]).astype(np.float32)
    return data / 3.0


real_data = create_real_data(2000)
real_tensor = torch.tensor(real_data, dtype=torch.float32)


class Generator(nn.Module):
    def __init__(self, noise_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(noise_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        )

    def forward(self, z):
        return self.network(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


noise_dim = 2
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(
    generator.parameters(), lr=0.001, betas=(0.5, 0.999)
)
d_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=0.001, betas=(0.5, 0.999)
)

epochs = 1000
batch_size = 64
d_losses = []
g_losses = []

print("=" * 60)
print("PRACTICAL 5: GENERATIVE ADVERSARIAL NETWORK (GAN)")
print("=" * 60)
print("Real dataset shape:", real_data.shape)
print("Generator parameters:",
      sum(p.numel() for p in generator.parameters()))
print("Discriminator parameters:",
      sum(p.numel() for p in discriminator.parameters()))
print("Training epochs:", epochs)
print()

for epoch in range(epochs):
    indices = np.random.randint(0, len(real_tensor), batch_size)
    real_samples = real_tensor[indices].to(device)

    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    # Train Discriminator
    d_optimizer.zero_grad()

    real_output = discriminator(real_samples)
    real_loss = criterion(real_output, real_labels)

    noise = torch.randn(batch_size, noise_dim, device=device)
    fake_samples = generator(noise)

    fake_output = discriminator(fake_samples.detach())
    fake_loss = criterion(fake_output, fake_labels)

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()

    # Train Generator
    g_optimizer.zero_grad()

    noise = torch.randn(batch_size, noise_dim, device=device)
    generated_samples = generator(noise)
    output = discriminator(generated_samples)

    # Generator wants fake samples to be classified as real.
    g_loss = criterion(output, real_labels)
    g_loss.backward()
    g_optimizer.step()

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1:4d}/{epochs}] | "
            f"D Loss: {d_loss.item():.4f} | "
            f"G Loss: {g_loss.item():.4f}"
        )

generator.eval()
with torch.no_grad():
    test_noise = torch.randn(1000, noise_dim, device=device)
    generated_data = generator(test_noise).cpu().numpy()
generator.train()

print()
print("Generated samples:", generated_data.shape)
print(
    "Generated data range:",
    f"{generated_data.min():.3f} to {generated_data.max():.3f}"
)

# Real vs generated data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(real_data[:, 0], real_data[:, 1], s=10, alpha=0.5)
plt.title("Real Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(alpha=0.2)

plt.subplot(1, 2, 2)
plt.scatter(generated_data[:, 0], generated_data[:, 1], s=10, alpha=0.5)
plt.title("GAN Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()

# Training loss
plt.figure(figsize=(9, 5))
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.title("GAN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

print()
print("OBSERVATION:")
print("1. The GAN consists of a Generator and a Discriminator.")
print("2. The Generator creates synthetic samples from random noise.")
print("3. The Discriminator distinguishes real samples from fake samples.")
print("4. Both networks improve through adversarial training.")
print("5. Generated samples learn the general distribution of real data.")

print()
print("CONCLUSION:")
print(
    "A simple Generative Adversarial Network was successfully implemented "
    "using PyTorch. The Generator learned to produce synthetic samples "
    "similar to the real data distribution, demonstrating the basic "
    "working principle of GANs."
)
