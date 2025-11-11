import torch
from gan_model import GAN, load_mnist_data as load_gan_data
from vae_model import VAETrainer, load_mnist_data as load_vae_data
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train GAN or VAE models')
    parser.add_argument('--model', type=str, choices=['gan', 'vae', 'both'], 
                       default='both', help='Model to train')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs to train')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.model in ['gan', 'both']:
        print("\n=== Training GAN ===")
        gan_dataloader = load_gan_data(batch_size=64)
        gan = GAN(device=device)
        gan.train(gan_dataloader, epochs=args.epochs)
        print("GAN training completed!")
    
    if args.model in ['vae', 'both']:
        print("\n=== Training VAE ===")
        vae_dataloader = load_vae_data(batch_size=128)
        vae_trainer = VAETrainer(device=device)
        vae_trainer.train(vae_dataloader, epochs=args.epochs)
        print("VAE training completed!")
    
    print("\nTraining completed! Check the generated images in 'gan_images' and 'vae_images' directories.")

if __name__ == "__main__":
    main()