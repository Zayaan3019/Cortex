# train.py
import torch
import yaml
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from models.cortex_synth import CortexSynth
from models.discriminator import DomainDiscriminator
from data.dataset import create_dataloaders
from utils.losses import geometry_loss, spectral_loss, adversarial_loss_g, adversarial_loss_d

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup model, discriminator, and optimizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CortexSynth(config).to(device)
    discriminator = DomainDiscriminator(in_features=config['encoder_channels'][-1]).to(device)

    optimizer_g = AdamW(model.parameters(), lr=config['learning_rate'])
    optimizer_d = AdamW(discriminator.parameters(), lr=config['learning_rate'])
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=config['num_epochs'])

    # Setup dataloaders for synthetic and real data
    real_loader, synthetic_loader = create_dataloaders(config)

    for epoch in range(config['num_epochs']):
        for (real_data, syn_data) in zip(real_loader, synthetic_loader):
            real_images, _ = real_data
            syn_images, (gt_joints, gt_adj) = syn_data
            
            real_images, syn_images = real_images.to(device), syn_images.to(device)
            gt_joints, gt_adj = gt_joints.to(device), gt_adj.to(device)

            # --- Generator (Cortex-Synth) Training ---
            optimizer_g.zero_grad()
            
            # Forward pass on synthetic data to get predictions
            pred_joints, pred_adj, syn_features = model(syn_images)
            
            # Compute main task losses
            loss_geom = config['lambda_geometry'] * geometry_loss(pred_joints, gt_joints)
            loss_spec = config['lambda_spectral'] * spectral_loss(pred_adj, gt_adj)
            
            # Compute adversarial loss for domain adaptation
            d_fake_output = discriminator(syn_features)
            loss_adv_g = config['lambda_domain'] * adversarial_loss_g(d_fake_output)
            
            # Total generator loss
            total_loss_g = loss_geom + loss_spec + loss_adv_g
            total_loss_g.backward()
            optimizer_g.step()

            # --- Discriminator Training ---
            optimizer_d.zero_grad()
            
            # Get features from real and synthetic domains
            with torch.no_grad():
                _, _, real_features = model(real_images)
            
            d_real_output = discriminator(real_features.detach())
            d_fake_output = discriminator(syn_features.detach())
            
            loss_d = adversarial_loss_d(d_real_output, d_fake_output)
            loss_d.backward()
            optimizer_d.step()
        
        scheduler_g.step()
        print(f"Epoch {epoch+1}, G_Loss: {total_loss_g.item():.4f}, D_Loss: {loss_d.item():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Cortex-Synth Model")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    main(args)
