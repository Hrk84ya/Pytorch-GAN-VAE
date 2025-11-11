# Generative Models: GAN and VAE Implementation

This project implements two fundamental generative models for image generation: **Generative Adversarial Networks (GANs)** and **Variational Autoencoders (VAEs)** using PyTorch and the MNIST dataset.

## Overview

Both models learn to generate new handwritten digit images by understanding the underlying data distribution of the MNIST dataset, but they use fundamentally different approaches.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train Both Models
```bash
python main.py --model both --epochs 50
```

### Train Only GAN
```bash
python main.py --model gan --epochs 100
```

### Train Only VAE
```bash
python main.py --model vae --epochs 50
```

## Model Architectures

### Generative Adversarial Network (GAN)

#### Architecture Overview
GANs consist of two neural networks competing against each other:

**Generator Network:**
- Input: Random noise vector (latent_dim=100)
- Architecture: 4 fully connected layers with batch normalization
- Layers: 100 → 128 → 256 → 512 → 1024 → 784
- Activation: LeakyReLU (except output layer uses Tanh)
- Output: 28×28 grayscale image

**Discriminator Network:**
- Input: 28×28 image (flattened to 784)
- Architecture: 3 fully connected layers
- Layers: 784 → 512 → 256 → 1
- Activation: LeakyReLU + final Sigmoid
- Output: Probability (real vs fake)

#### Training Process

1. **Adversarial Training Loop:**
   - Generator creates fake images from random noise
   - Discriminator learns to distinguish real from fake images
   - Generator learns to fool the discriminator

2. **Loss Functions:**
   - **Generator Loss:** `BCE(D(G(z)), 1)` - wants discriminator to classify fake as real
   - **Discriminator Loss:** `[BCE(D(real), 1) + BCE(D(G(z)), 0)] / 2`

3. **Optimization:**
   - Adam optimizer with learning rate 0.0002
   - Beta values: (0.5, 0.999) for stability
   - Alternating updates between generator and discriminator

### Variational Autoencoder (VAE)

#### Architecture Overview

**Encoder Network:**
- Input: 28×28 image (flattened to 784)
- Hidden layer: 784 → 400 (ReLU activation)
- Output: Two vectors (μ and log σ²) of size 20 each
- Purpose: Maps input to latent space distribution parameters

**Decoder Network:**
- Input: Latent vector (size 20)
- Hidden layer: 20 → 400 (ReLU activation)
- Output: 400 → 784 (Sigmoid activation)
- Purpose: Reconstructs image from latent representation

#### Training Process

1. **Forward Pass:**
   - Encode input image to get μ and log σ²
   - Sample latent vector z using reparameterization trick
   - Decode z to reconstruct image

2. **Reparameterization Trick:**
   ```
   z = μ + σ * ε, where ε ~ N(0,1)
   ```
   - Enables backpropagation through stochastic sampling
   - Maintains differentiability of the network

3. **Loss Function (ELBO):**
   ```
   Loss = Reconstruction Loss + KL Divergence
   Loss = BCE(x, x_reconstructed) + KL(q(z|x) || p(z))
   ```

## Dataset: MNIST

- **Size:** 60,000 training images of handwritten digits (0-9)
- **Image Dimensions:** 28×28 pixels, grayscale
- **Preprocessing:**
  - GAN: Normalized to [-1, 1] range using `(x - 0.5) / 0.5`
  - VAE: Normalized to [0, 1] range using `ToTensor()`
- **Why MNIST:** Standard benchmark, simple structure, fast training

## Training Details

### GAN Training
- **Batch Size:** 64
- **Learning Rate:** 0.0002
- **Epochs:** 100 (recommended)
- **Loss:** Binary Cross Entropy
- **Optimization:** Adam with β₁=0.5, β₂=0.999

### VAE Training
- **Batch Size:** 128
- **Learning Rate:** 0.001
- **Epochs:** 50 (recommended)
- **Loss:** ELBO (Reconstruction + KL Divergence)
- **Optimization:** Adam with default parameters

## Output and Results

### Generated Images
- **GAN:** `gan_images/epoch_X.png` - 5×5 grid of generated samples
- **VAE:** `vae_images/sample_epoch_X.png` - 5×5 grid of generated samples
- **VAE Reconstructions:** `vae_images/reconstruction_epoch_X.png` - Original vs reconstructed

### Training Monitoring
- Loss values printed every 400 batches
- Sample images saved every 10 epochs
- Progress tracking for both models

## Comparison: GAN vs VAE

| Aspect | GAN | VAE |
|--------|-----|-----|
| **Image Quality** | Often sharper, more realistic | Slightly blurrier but consistent |
| **Training Stability** | Can be unstable, requires tuning | More stable and predictable |
| **Latent Space** | Less structured | Well-organized, interpolatable |
| **Mode Coverage** | Risk of mode collapse | Better mode coverage |
| **Computational Cost** | Higher (two networks) | Lower (single network) |
| **Use Cases** | High-quality generation | Representation learning, interpolation |

## Key Differences in Approach

### GAN Philosophy
- **Adversarial Learning:** Two networks in competition
- **Implicit Density Model:** Doesn't model p(x) directly
- **Sharp Generation:** Can produce very realistic samples
- **Training Challenge:** Balancing generator/discriminator strength

### VAE Philosophy
- **Probabilistic Framework:** Based on variational inference
- **Explicit Density Model:** Models p(x) through latent variables
- **Structured Latent Space:** Enables meaningful interpolation
- **Stable Training:** Well-defined objective function

## Advanced Concepts

### GAN Training Dynamics
- **Generator Gradient:** Flows through discriminator feedback
- **Discriminator Gradient:** Direct supervision from real/fake labels
- **Equilibrium:** Both networks reach optimal performance simultaneously
- **Common Issues:** Mode collapse, vanishing gradients, training instability

### VAE Mathematical Foundation
- **Evidence Lower Bound (ELBO):** Maximizes log p(x) indirectly
- **KL Divergence:** Regularizes latent space to match prior N(0,I)
- **Reconstruction Term:** Ensures generated images match input distribution
- **Posterior Approximation:** q(z|x) approximates true posterior p(z|x)

## Practical Applications

### GANs
- High-resolution image generation
- Style transfer and image-to-image translation
- Data augmentation for training datasets
- Creative applications (art, design)

### VAEs
- Dimensionality reduction and data compression
- Anomaly detection (reconstruction error)
- Semi-supervised learning
- Controlled generation through latent manipulation

## Future Improvements

1. **Advanced GAN Variants:**
   - DCGAN (Convolutional layers)
   - WGAN (Wasserstein loss)
   - StyleGAN (Style-based generation)

2. **VAE Extensions:**
   - β-VAE (Controllable disentanglement)
   - Conditional VAE (Class-conditional generation)
   - VQ-VAE (Discrete latent space)

3. **Technical Enhancements:**
   - Progressive training
   - Spectral normalization
   - Self-attention mechanisms
   - Mixed precision training

## Troubleshooting

### Common GAN Issues
- **Mode Collapse:** Reduce learning rates, try different architectures
- **Training Instability:** Balance generator/discriminator updates
- **Poor Quality:** Increase model capacity, adjust hyperparameters

### Common VAE Issues
- **Blurry Images:** Increase latent dimension, try different decoders
- **Posterior Collapse:** Adjust β in β-VAE, modify architecture
- **Poor Reconstruction:** Increase model capacity, check preprocessing

This implementation provides a solid foundation for understanding and experimenting with generative models, offering both theoretical insights and practical experience with state-of-the-art techniques.