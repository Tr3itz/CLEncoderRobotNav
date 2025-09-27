import torch

def lidar_sims(lidars: torch.Tensor, mask: str='soft', shift: float=0.0):
    """
    Compute in-batch lidar distances.
    """

    # Define LiDAR readings mask
    w = torch.zeros(size=(lidars.shape[1],))
    match mask:
        case 'naive':
            w += 1
        case 'binary':
            # In FOV readings
            w[64:164] += 1
        case 'soft':
            # In FOV readings
            w[64:164] += 1
            # Out of FOV readings
            x = torch.linspace(0.0, 1.0, w[164:].shape[0])
            s_right = 1 - 0.9*(1 / (1+torch.exp(-x + shift))) # Sigmoid 1.0 -> 0.1
            s_left = 0.1 + 0.9*(1 / (1+torch.exp(-x + shift))) # Sigmoid 0.1 -> 1.0
            w[164:] += s_right
            w[:64] += s_left
            
    # Mask and Normalizer    
    mask_w = w.to(lidars.device)
    norm = torch.sqrt(w.sum())

    # Weighted eucledian distances between in-batch anchors lidar readings
    lid_dists = (lidars.unsqueeze(0) - lidars.unsqueeze(1)).pow(2) * mask_w        
    lid_dists = lid_dists.sum(dim=-1).sqrt() 
    # Normalize distances to [0, 1]
    lid_dists /= norm

    return 1 - lid_dists
    
def goal_sims(gds: torch.Tensor, angles: torch.Tensor):
    """
    Compute in-batch position differences w.r.t. the goal.
    """

    # Differences between in-batch anchors goal distances
    gd_diffs = torch.abs(gds.unsqueeze(0) - gds.unsqueeze(1))

    # Differences between in-batch anchors orientations w.r.t. the goal
    ori_diffs = ((angles.unsqueeze(0) - angles.unsqueeze(1)) + torch.pi) % (2 * torch.pi) - torch.pi
    ori_diffs = torch.abs(ori_diffs) / torch.pi

    return 1 - gd_diffs*ori_diffs

    
def robot_nav_scores(*args, lidars: torch.Tensor, gds: torch.Tensor, angles: torch.Tensor, metric: str='both'):
    """
    Compute in-batch negative scores for RoomAllAgents anchors.
    """

    # Distances between in-batch examples
    if metric == 'lidar':
        mask, shift = args
        batch_scores = lidar_sims(lidars, mask, shift)
    elif metric == 'goal':
        batch_scores = goal_sims(gds, angles)
    else:
        mask, shift = args
        batch_scores = lidar_sims(lidars, mask, shift) *  goal_sims(gds, angles)      

    return batch_scores

def airsim_scores(*args, positions: torch.Tensor, velocities: torch.Tensor, quaternions: torch.Tensor):
    """
    Compute in-batch negative scores for AirSim anchors.
    """
    Wp, Wv, Wpos, Wrot = args

    # Position similarity
    pos_dist_mat = torch.cdist(positions, positions, p=2.0)
    vel_magnitudes = torch.linalg.norm(velocities, dim=1)
    avg_vel_mat = (vel_magnitudes[:, None] + vel_magnitudes) / 2.0
    dynamic_scale_mat = Wp / (1 + avg_vel_mat * Wv)
    pos_sim_mat = torch.exp(-dynamic_scale_mat * pos_dist_mat)

    # Rotation similarity
    norms = torch.linalg.norm(quaternions, dim=1, keepdim=True)
    quaternions_normalized = quaternions / norms
    rot_sim_mat = torch.abs(quaternions_normalized @ quaternions_normalized.T)

    # Weighted combination
    sim_scores_mat = (pos_sim_mat * Wpos) + (rot_sim_mat * Wrot)
    sim_scores_mat.fill_diagonal_(1.0)

    return sim_scores_mat