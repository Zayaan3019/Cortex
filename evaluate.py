# evaluate.py
import torch
import yaml
import argparse
from torch.utils.data import DataLoader
from models.cortex_synth import CortexSynth
from data.dataset import SyntheticSkeletonDataset # Use a test set
from utils.evaluation_metrics import mean_per_joint_position_error, chamfer_distance, graph_edit_distance

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CortexSynth(config).to(device)
    
    # Load the trained model checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Create a dataloader for the test set
    test_dataset = SyntheticSkeletonDataset(num_samples=200) # Assuming a test split
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    total_mpjpe = 0
    total_chamfer = 0
    total_ged = 0
    num_batches = 0

    with torch.no_grad():
        for image, (gt_joints, gt_adj) in test_loader:
            image, gt_joints, gt_adj = image.to(device), gt_joints.to(device), gt_adj.to(device)
            
            pred_joints, pred_adj, _ = model(image)
            
            # Calculate metrics for the batch
            total_mpjpe += mean_per_joint_position_error(pred_joints, gt_joints).item()
            total_chamfer += chamfer_distance(pred_joints, gt_joints).item()
            total_ged += graph_edit_distance(pred_adj, gt_adj)
            
            num_batches += 1

    # Calculate and print average metrics
    avg_mpjpe = total_mpjpe / num_batches
    avg_chamfer = total_chamfer / num_batches
    avg_ged = total_ged / num_batches

    print("--- Evaluation Results ---")
    print(f"Mean Per Joint Position Error (MPJPE): {avg_mpjpe:.4f}")
    print(f"Chamfer Distance: {avg_chamfer:.6f}")
    print(f"Graph Edit Distance (GED): {avg_ged:.4f}")
    print("-------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Cortex-Synth Model")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to trained model checkpoint")
    args = parser.parse_args()
    main(args)
