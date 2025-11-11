import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar layer
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class VAETrainer:
    def __init__(self, latent_dim=20, lr=1e-3, device='cpu'):
        self.device = device
        self.latent_dim = latent_dim
        
        self.model = VAE(latent_dim=latent_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def train(self, dataloader, epochs=100):
        os.makedirs("vae_images", exist_ok=True)
        self.model.train()
        
        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                
                recon_batch, mu, logvar = self.model(data)
                loss = vae_loss_function(recon_batch, data, mu, logvar)
                
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                
                if batch_idx % 400 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                          f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
            
            print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')
            
            if epoch % 10 == 0:
                self.save_sample_images(epoch)
                self.save_reconstructions(dataloader, epoch)
    
    def save_sample_images(self, epoch):
        with torch.no_grad():
            sample = torch.randn(25, self.latent_dim).to(self.device)
            sample = self.model.decode(sample).cpu()
            
            fig, axes = plt.subplots(5, 5, figsize=(10, 10))
            for i in range(25):
                axes[i//5, i%5].imshow(sample[i].view(28, 28), cmap='gray')
                axes[i//5, i%5].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'vae_images/sample_epoch_{epoch}.png')
            plt.close()
    
    def save_reconstructions(self, dataloader, epoch):
        with torch.no_grad():
            data, _ = next(iter(dataloader))
            data = data[:8].to(self.device)
            recon, _, _ = self.model(data)
            
            comparison = torch.cat([data[:8], recon.view(-1, 1, 28, 28)[:8]])
            
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            for i in range(8):
                # Original
                axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstruction
                axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'vae_images/reconstruction_epoch_{epoch}.png')
            plt.close()

def load_mnist_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloader = load_mnist_data()
    vae_trainer = VAETrainer(device=device)
    vae_trainer.train(dataloader, epochs=50)