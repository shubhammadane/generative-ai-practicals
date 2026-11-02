"""
PRACTICAL 6: DIFFUSION MODELS

Roll No.: BT24S05F006

AIM:
To study the working principle of Diffusion Models and implement
a simple diffusion-based generative model to generate synthetic data.

THEORY:
Diffusion Models are generative models that learn to generate data by
reversing a gradual noising process.

Forward Process:
Real data -> add noise gradually -> noisy data

Reverse Process:
Random noise -> remove noise step by step -> generated data

In this practical, a lightweight 2D diffusion model is implemented
using PyTorch. It does not download Stable Diffusion or any large
pretrained model, making it suitable for systems with limited storage.

REQUIREMENTS:
    pip install numpy matplotlib torch
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")


# ============================================================
# 2. CREATE REAL DATA
# ============================================================

def create_real_data(n_samples=2000):
    """Create a simple 2D mixture of Gaussian clusters."""

    n1 = n_samples // 2
    n2 = n_samples - n1

    cluster1 = np.random.normal(
        loc=[-1.5, -1.5],
        scale=0.35,
        size=(n1, 2)
    )

    cluster2 = np.random.normal(
        loc=[1.5, 1.5],
        scale=0.35,
        size=(n2, 2)
    )

    data = np.vstack([cluster1, cluster2]).astype(np.float32)

    return data


real_data = create_real_data(2000)

real_tensor = torch.tensor(
    real_data,
    dtype=torch.float32
)


# ============================================================
# 3. DIFFUSION SCHEDULE
# ============================================================

T = 50

beta = torch.linspace(
    0.0005,
    0.02,
    T,
    device=device
)

alpha = 1.0 - beta

alpha_bar = torch.cumprod(
    alpha,
    dim=0
)


# ============================================================
# 4. FORWARD DIFFUSION PROCESS
# ============================================================

def add_noise(x0, timestep):
    """
    Add Gaussian noise to clean data according to the
    selected diffusion timestep.
    """

    noise = torch.randn_like(x0)

    a_bar = alpha_bar[timestep].view(-1, 1)

    noisy_data = (
        torch.sqrt(a_bar) * x0
        + torch.sqrt(1.0 - a_bar) * noise
    )

    return noisy_data, noise


# ============================================================
# 5. TIME EMBEDDING
# ============================================================

def time_embedding(t, dimension=16):
    """Create a simple sinusoidal time embedding."""

    half = dimension // 2

    frequencies = torch.exp(
        -np.log(10000)
        * torch.arange(
            half,
            device=t.device
        )
        / max(half - 1, 1)
    )

    angles = t.float().view(-1, 1) * frequencies.view(1, -1)

    return torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=1
    )


# ============================================================
# 6. NOISE PREDICTION NETWORK
# ============================================================

class NoisePredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2 + 16, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, x, t):

        t_embed = time_embedding(t, 16)

        input_data = torch.cat(
            [x, t_embed],
            dim=1
        )

        return self.network(input_data)


model = NoisePredictor().to(device)


# ============================================================
# 7. TRAINING
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_function = nn.MSELoss()

epochs = 1000

batch_size = 128

losses = []

print("=" * 60)
print("PRACTICAL 6: DIFFUSION MODEL")
print("=" * 60)

print("Real dataset shape:", real_data.shape)
print("Diffusion timesteps:", T)
print("Training epochs:", epochs)
print()

for epoch in range(epochs):

    indices = np.random.randint(
        0,
        len(real_tensor),
        batch_size
    )

    clean_data = real_tensor[
        indices
    ].to(device)

    # Select random diffusion timestep
    timesteps = torch.randint(
        0,
        T,
        (batch_size,),
        device=device
    )

    # Add noise
    noisy_data, true_noise = add_noise(
        clean_data,
        timesteps
    )

    # Predict the noise
    predicted_noise = model(
        noisy_data,
        timesteps
    )

    # Compare predicted and actual noise
    loss = loss_function(
        predicted_noise,
        true_noise
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch [{epoch + 1:4d}/{epochs}] | "
            f"Loss: {loss.item():.6f}"
        )


# ============================================================
# 8. REVERSE DIFFUSION / SAMPLING
# ============================================================

model.eval()

num_samples = 1000

with torch.no_grad():

    # Start from random Gaussian noise
    samples = torch.randn(
        num_samples,
        2,
        device=device
    )

    # Reverse the diffusion process
    for step in reversed(range(T)):

        t = torch.full(
            (num_samples,),
            step,
            device=device,
            dtype=torch.long
        )

        predicted_noise = model(
            samples,
            t
        )

        alpha_t = alpha[step]
        alpha_bar_t = alpha_bar[step]
        beta_t = beta[step]

        # DDPM reverse-step approximation
        samples = (
            (1 / torch.sqrt(alpha_t))
            * (
                samples
                - (
                    beta_t
                    / torch.sqrt(1 - alpha_bar_t)
                )
                * predicted_noise
            )
        )

        if step > 0:

            noise = torch.randn_like(samples)

            samples = samples + torch.sqrt(beta_t) * noise


    generated_data = samples.cpu().numpy()


print()
print("Generated samples:", generated_data.shape)


# ============================================================
# 9. REAL DATA VS GENERATED DATA
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.scatter(
    real_data[:, 0],
    real_data[:, 1],
    s=10,
    alpha=0.5
)

plt.title("Real Data")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.grid(alpha=0.2)


plt.subplot(1, 2, 2)

plt.scatter(
    generated_data[:, 0],
    generated_data[:, 1],
    s=10,
    alpha=0.5
)

plt.title("Diffusion Generated Data")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.grid(alpha=0.2)

plt.tight_layout()

plt.show()


# ============================================================
# 10. TRAINING LOSS
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    losses,
    label="Training Loss"
)

plt.title("Diffusion Model Training Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.legend()

plt.grid(alpha=0.2)

plt.tight_layout()

plt.show()


# ============================================================
# 11. OBSERVATION
# ============================================================

print()
print("OBSERVATION:")

print(
    "1. Diffusion models gradually add noise to real data."
)

print(
    "2. The neural network learns to predict the noise."
)

print(
    "3. Generation starts from random noise."
)

print(
    "4. The reverse diffusion process removes noise step by step."
)

print(
    "5. The final samples learn the general distribution of the "
    "original dataset."
)


# ============================================================
# 12. CONCLUSION
# ============================================================

print()
print("CONCLUSION:")

print(
    "A lightweight diffusion-based generative model was successfully "
    "implemented using PyTorch. The model learned to predict noise "
    "at different diffusion timesteps and generated synthetic samples "
    "through the reverse diffusion process."
)
