import argparse
import yaml

class Configurations:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Configuration for Contrastive Scene Transfer")
        self._add_arguments()
        self.args = self.parser.parse_args()

    def _add_arguments(self):
        # General settings
        self.parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

        # Dataset settings
        self.parser.add_argument('--datasets_path', type=str, required=True, help="Path to the datasets folder")
        self.parser.add_argument('--dataset', type=str, default='with-augmentations', help="Dataset to use for training")
        self.parser.add_argument('--val_room', type=int, default=2, help="Validation room")
        self.parser.add_argument('--metric', type=str, default='lidar', choices=['lidar', 'goal', 'both'], help="Metric for denoting similarity")
        self.parser.add_argument('--mask', type=str, choices=['naive', 'binary', 'soft'], help='LiDAR readings mask')
        self.parser.add_argument('--shift', type=float, help="Shift of the sigmoid for soft LiDAR masking")
        self.parser.add_argument('--n_pos', type=int, default=0, help="Number of positive examples to sample per anchor during training")
        self.parser.add_argument('--pos_thresh', type=float, default=0.7, help="Positive similarity threshold")
        self.parser.add_argument('--n_neg', type=int, default=0, help="Number of negative examples to sample per anchor during training")
        self.parser.add_argument('--neg_thresh', type=float, default=0.5, help="Negative similarity threshold")
        self.parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training and evaluation")
        self.parser.add_argument('--num_workers', type=int, default=14, help="Number of workers for data loading")

        # Loss, Optimizer and Scheduler settings
        self.parser.add_argument('--loss', type=str, default='sim', choices=['sim', 'l1', 'l2'], help="CL Loss Function")
        self.parser.add_argument('--min_tau', type=float, default=0.1, help="Minimum temperature")
        self.parser.add_argument('--max_tau', type=float, default=1.0, help="Maximum temperature")
        self.parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the optimizer")

        # Training settings
        self.parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
        self.parser.add_argument('--val_freq', type=int, default=1, help="Interval for validation during training")
    
    def get_args(self):
        return self.args

    def save_yaml(self, dir: str):
        with open(f'{dir}/config.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)

if __name__=='__main__':
    conf = Configurations()
    print(conf.args)