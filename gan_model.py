import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN:
    def __init__(self, latent_dim=100, lr=0.0002, device='cpu'):
        self.device = device
        self.latent_dim = latent_dim
        
        self.generator = Generator(latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        
        self.adversarial_loss = nn.BCELoss()
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    def train(self, dataloader, epochs=200):
        os.makedirs("gan_images", exist_ok=True)
        
        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(dataloader):
                batch_size = imgs.shape[0]
                
                # Adversarial ground truths
                valid = torch.ones(batch_size, 1, requires_grad=False).to(self.device)
                fake = torch.zeros(batch_size, 1, requires_grad=False).to(self.device)
                
                real_imgs = imgs.to(self.device)
                
                # Train Generator
                self.optimizer_G.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                gen_imgs = self.generator(z)
                
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                self.optimizer_G.step()
                
                # Train Discriminator
                self.optimizer_D.zero_grad()
                
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                
                d_loss.backward()
                self.optimizer_D.step()
                
                if i % 400 == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            
            if epoch % 10 == 0:
                self.save_sample_images(epoch)
    
    def save_sample_images(self, epoch):
        with torch.no_grad():
            z = torch.randn(25, self.latent_dim).to(self.device)
            gen_imgs = self.generator(z)
            
            fig, axes = plt.subplots(5, 5, figsize=(10, 10))
            for i in range(25):
                axes[i//5, i%5].imshow(gen_imgs[i].cpu().squeeze(), cmap='gray')
                axes[i//5, i%5].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"gan_images/epoch_{epoch}.png")
            plt.close()

def load_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
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
    gan = GAN(device=device)
    gan.train(dataloader, epochs=100)