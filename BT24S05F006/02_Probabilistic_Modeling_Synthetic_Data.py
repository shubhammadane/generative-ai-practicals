# ================================================================
# Practical 2: Probabilistic Modeling and Synthetic Data
# Roll No.: BT24S05F006
# ================================================================

"""
AIM
---
To implement probabilistic modeling techniques and generate
synthetic data samples.

THEORY
------
Probabilistic modeling represents data using probability
distributions. A Gaussian (Normal) distribution is commonly used
to model continuous data.

For a Gaussian distribution:

        f(x) = 1/(sigma*sqrt(2*pi)) *
               exp(-(x-mu)^2/(2*sigma^2))

where mu is the mean and sigma is the standard deviation.

In this practical:
1. We visualize a Gaussian probability density function (PDF).
2. We generate synthetic samples from Gaussian distributions.
3. We fit a Gaussian Mixture Model (GMM) to observed data.
4. We generate new synthetic samples from the fitted GMM.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

np.random.seed(42)

print("=" * 65)
print("PRACTICAL 2: PROBABILISTIC MODELING AND SYNTHETIC DATA")
print("=" * 65)

# ---------------------------------------------------------------
# 1. Gaussian Probability Density Function
# ---------------------------------------------------------------
mu = 50
sigma = 10

x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)

gaussian_pdf = (
    1 / (sigma * np.sqrt(2 * np.pi))
    * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
)

plt.figure(figsize=(8, 5))
plt.plot(x, gaussian_pdf, linewidth=2)
plt.title("Gaussian Probability Density Function")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.grid(alpha=0.3)
plt.show()

print("\n1. Gaussian Distribution")
print("Mean (mu):", mu)
print("Standard deviation (sigma):", sigma)

# ---------------------------------------------------------------
# 2. Generate synthetic samples from Gaussian distribution
# ---------------------------------------------------------------
synthetic_samples = np.random.normal(
    loc=mu,
    scale=sigma,
    size=1000
)

print("\n2. Synthetic Gaussian Data")
print("Number of generated samples:", len(synthetic_samples))
print("Sample mean:", round(synthetic_samples.mean(), 2))
print("Sample standard deviation:", round(synthetic_samples.std(), 2))
print("First 10 samples:", np.round(synthetic_samples[:10], 2))

plt.figure(figsize=(8, 5))
plt.hist(
    synthetic_samples,
    bins=30,
    density=True,
    alpha=0.7,
    label="Synthetic samples"
)
plt.plot(
    x,
    gaussian_pdf,
    linewidth=2,
    label="Theoretical Gaussian PDF"
)
plt.title("Synthetic Data from Gaussian Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ---------------------------------------------------------------
# 3. Create observed data from two Gaussian distributions
# ---------------------------------------------------------------
data_1 = np.random.normal(loc=35, scale=5, size=500)
data_2 = np.random.normal(loc=70, scale=8, size=500)

observed_data = np.concatenate([data_1, data_2]).reshape(-1, 1)

print("\n3. Observed Mixed Data")
print("Number of observations:", len(observed_data))

plt.figure(figsize=(8, 5))
plt.hist(observed_data, bins=35, density=True, alpha=0.7)
plt.title("Observed Data from Two Gaussian Sources")
plt.xlabel("Value")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.show()

# ---------------------------------------------------------------
# 4. Fit Gaussian Mixture Model
# ---------------------------------------------------------------
gmm = GaussianMixture(
    n_components=2,
    random_state=42
)

gmm.fit(observed_data)

print("\n4. Gaussian Mixture Model")
print("Number of components:", gmm.n_components)
print("Learned means:", np.round(gmm.means_.flatten(), 2))
print("Learned standard deviations:",
      np.round(np.sqrt(gmm.covariances_.flatten()), 2))
print("Learned weights:", np.round(gmm.weights_, 3))

# ---------------------------------------------------------------
# 5. Generate synthetic data using fitted GMM
# ---------------------------------------------------------------
generated_data, component_labels = gmm.sample(1000)

print("\n5. Generated Data using GMM")
print("Number of generated samples:", len(generated_data))
print("Generated mean:", round(generated_data.mean(), 2))
print("Generated standard deviation:",
      round(generated_data.std(), 2))

plt.figure(figsize=(8, 5))
plt.hist(
    observed_data,
    bins=35,
    density=True,
    alpha=0.5,
    label="Observed data"
)
plt.hist(
    generated_data,
    bins=35,
    density=True,
    alpha=0.5,
    label="GMM synthetic data"
)
plt.title("Observed Data vs GMM Synthetic Data")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ---------------------------------------------------------------
# 6. Display a small sample of generated values
# ---------------------------------------------------------------
print("\nFirst 10 GMM-generated samples:")
print(np.round(generated_data[:10].flatten(), 2))

print("\n" + "=" * 65)
print("OBSERVATION")
print("=" * 65)
print("1. A Gaussian distribution models continuous data using mean")
print("   and standard deviation.")
print("2. Random sampling produces synthetic observations following")
print("   the selected probability distribution.")
print("3. A Gaussian Mixture Model can represent data containing")
print("   multiple Gaussian components.")
print("4. The fitted GMM can generate new synthetic samples similar")
print("   to the observed data distribution.")

print("\n" + "=" * 65)
print("CONCLUSION")
print("=" * 65)
print("Probabilistic modeling can represent uncertainty and data")
print("distributions. Gaussian distributions and Gaussian Mixture")
print("Models can be used to generate useful synthetic data samples.")
print("=" * 65)
