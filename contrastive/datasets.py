# Torch imports
import torch
import torch.distributed as distr
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2

# Utils imports
import os
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from PIL import Image
from multiprocessing import shared_memory  
from glob import glob
from tqdm.auto import tqdm
from abc import ABC, abstractmethod


class ContrastiveDataset(Dataset, ABC):
    def __init__(
            self, 
            dir: str,  
            transforms: v2,
            algo: str,
            n_pos: int,
            pos_thresh: float,
            n_neg: int,
            neg_thresh: float,
            batch_size: int,
            micro_bsize: int,
            augmentations: list=None,
            mode: str='train',
            multi_gpu: bool=False,
            seed: int=42
        ):
        """
        Base dataset class.
        ----------
        Parameters:
        - dir: str            - directory of the dataset
        - transforms: v2      - image transformations to apply
        - algo: str           - Contrastive Learning framework
        - n_pos: int          - number of positive examples (scene-transfer framework)
        - pos_thresh: float   - positive similarity threshold (scene-transfer framework)
        - n_neg: int          - number of negative examples (scene-transfer framework)
        - neg_thresh: float   - negative similarity threshold (scene-transfer framework)
        - batch_size: int     - size of the batch returned by the DataLoader
        - micro_bsize: int    - size of the micro-batch for gradient accumulation
        - augmentations       - augmentations for examples
        - mode: str           - dataset mode (train or val)
        - multi_gpu: bool     - whether the dataset is used for training a model on multiple GPUs
        - seed: int           - random seed for reproducibility        
        """
        super().__init__()

        # Set the random seed for reproducibility
        self.seed = seed
        
        # Directory of the dataset
        self.dir = dir

        # Image transformations and augmentations
        self.transforms = transforms
        self.augmentations = augmentations

        # Contrastive Learning framework
        assert algo in ['simclr', 'scene-transfer'], f'Unknown framework {algo}. Must be either `simclr` or `scene-transfer`!'
        self.algo = algo
        if self.algo == 'scene-transfer':
            assert n_pos > 0 and n_neg > 0, 'Scene-transfer network with no examples.'
           
            # Positive examples
            self.n_pos = n_pos
            self.pos_thresh = pos_thresh
            
            # Negative examples
            self.n_neg = n_neg
            self.neg_thresh = neg_thresh

        # Dataset mode
        assert mode in ['train', 'val']
        self.mode = mode

        # Distributed training
        self.multi_gpu = multi_gpu

        # Batch and Micro-batch
        assert micro_bsize == 0 or batch_size % micro_bsize == 0, f'Invalid micro-batch size: batch={batch_size}, micro-batch={micro_bsize}'
        self.batch_size = batch_size
        self.micro_bsize = micro_bsize if micro_bsize > 0 else batch_size            

        # Annotations dataframe
        assert os.path.exists(self.dir)
        try:
            with open(f'{self.dir}/annotations.pkl', 'rb') as f:
                self.annot_df = pickle.load(f)
        except FileNotFoundError:
            print(f"{'Creating annotations file...' if not self.multi_gpu else f'[GPU:{distr.get_rank()}] Retrieving additional info...'}")
            self.annot_df = self._annot()

        # Pandas methods will be used on this dataframe
        assert isinstance(self.annot_df, pd.DataFrame), f'Annotations object is required to be a Pandas DataFrame. Found type {type(self.annot_df)}.'

    def __del__(self):
        if hasattr(self, '_shm'):
            self._shm.close()

    def __len__(self):
        return self.annot_df.shape[0]   
    
    @abstractmethod
    def __getitem__(self): pass
    
    @abstractmethod
    def _annot(self):
        """
        Create a global annotations file of the dataset.
        """
        pass

    @abstractmethod
    def _init_sim_matrix(self):
        """
        Initialize similarity matrix (n_samples, n_samples), given additional information in self.annot_df. 
        """
        pass

    def _free_shm(self, rank):
        if hasattr(self, '_shm'):
            self._shm.close()
            if rank == 0:
                self._shm.unlink()
    
    def _shared_sim_mat(self):
        print([print(f"[GPU:{distr.get_rank()}] COMPUTING {'TRAINING' if self.mode == 'train' else 'VALIDATION'} SIMILARITY SCORES MATRIX...")])
        sim_scores_mat = self._init_sim_matrix()

        # Matrix info 
        shape = sim_scores_mat.shape
        dtype = sim_scores_mat.dtype
        size = sim_scores_mat.nbytes

        print([print(f"[GPU:{distr.get_rank()}] Copying similarity scores matrix on shared memory...")])
        # Shared memory block creation
        try:
            shm = shared_memory.SharedMemory(name=f'{self.mode}_sim_scores_mat', create=True, size=size)
            shared_mat = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
            shared_mat[:] = sim_scores_mat[:] # Copy on shared memory
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=f'{self.mode}_sim_scores_mat', create=False)
            shared_mat = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)

        return shm, shared_mat
    
    def set_shared_sim_mat(self):
        if not hasattr(self, 'sim_scores_mat'):
            try:
                shm = shared_memory.SharedMemory(name=f'{self.mode}_sim_scores_mat', create=False)
                self.sim_scores_mat = np.ndarray(shape=(self.annot_df.shape[0], self.annot_df.shape[0]), dtype=np.float32, buffer=shm.buf)
                self._shm = shm
            except FileNotFoundError:
                print(f'Could not find shared memory block named `{self.mode}_sim_scores_mat`.')

    def get_DataLoader(self, num_workers: int=None) -> DataLoader:
        """ 
        Return the torch DataLoader of the dataset.
        """
        if self.multi_gpu:
            return DataLoader(
                dataset=self,
                batch_size=self.batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(self.mode == 'train'),
                sampler=DistributedSampler(self)
            )
        else:
            return DataLoader(
                dataset=self,
                batch_size=self.batch_size,
                shuffle=(self.mode == 'train'),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(self.mode == 'train'),
            )


class RoomAllAgentsDataset(ContrastiveDataset):
    def __init__(
            self,
            dir: str,
            algo: str,
            metric: str,
            mask: str,
            shift: float,
            n_pos: int,
            pos_thresh: float,
            n_neg: int,
            neg_thresh: float,
            batch_size: int,
            micro_bsize: int,
            val_room: int,
            transforms: v2,
            augmentations: list=None,
            mode: str='train',
            multi_gpu: bool=False,
            seed: int=42
        ):
        """
        Torch implementation of Contrastive Dataset for Obstacle Avoidance in Robotic Navigation.
        ----------
        Parameters:
        - dir: str            - directory of the dataset
        - metric: str         - metric for computing sample similarity (lidar, goal, both)
        - mask: str           - LiDAR readings mask type
        - algo: str           - Contrastive Learning framework
        - n_pos: int          - number of positive examples (scene-transfer framework)
        - pos_thresh: float   - positive similarity threshold (scene-transfer framework)
        - n_neg: int          - number of negative examples (scene-transfer framework)
        - neg_thresh: float   - negative similarity threshold (scene-transfer framework)
        - batch_size: int     - size of the batch returned by the DataLoader
        - micro_bsize: int    - size of the micro-batch for gradient accumulation
        - val_room: int       - validation room
        - transforms: v2      - image transformations to apply
        - augmentations       - additional augmentations for positive examples
        - mode: str           - dataset mode (train or val)
        - multi_gpu: bool     - whether the dataset is used for training a model on multiple GPUs
        - seed: int           - random seed for reproducibility
        """
        super().__init__(
            dir=dir,
            transforms=transforms,
            algo=algo,
            n_pos=n_pos,
            pos_thresh=pos_thresh,
            n_neg=n_neg,
            neg_thresh=neg_thresh,
            batch_size=batch_size,
            micro_bsize=micro_bsize,
            augmentations=augmentations,
            mode=mode,
            multi_gpu=multi_gpu,
            seed=seed
        )
            
        # Scenes partitions
        self.SCENES = {
            'train': [3, 4, 5, 6],
            'val': [1, 2, 7]
        }

        # Scene map
        self.SCENE_MAP = {
            'train': {
                0: 'No wall',
                1: 'No background',
                2: 'Warehouse 1',
                3: 'Warehouse 2'
            },
            'val': {
                0: 'Stadium',
                1: 'Office',
                2: 'Warehouse 3'
            } 
        }

        # Similarity metric
        assert metric in ['lidar', 'goal', 'both']
        self.metric = metric
        if metric in ['lidar', 'both']:
            assert mask in ['naive', 'binary', 'soft']
            self.mask = mask

            # Define LiDAR readings mask
            rand_sample = self.annot_df.sample(n=1).iloc[0]
            w = np.zeros(rand_sample['laser_readings']['scan'].squeeze().shape[0])
            match self.mask:
                case 'naive':
                    w += 1
                case 'binary':
                    # In FOV readings
                    w[64:164] += 1
                case 'soft':
                    assert shift is not None
                    self.shift = shift

                    # In FOV readings
                    w[64:164] += 1
                    # Out of FOV readings
                    x = np.linspace(0.0, 1.0, w[164:].shape[0])
                    sigmoid = 1 - 0.9*(1 / (1+np.exp(-x + self.shift))) # Sigmoid 1.0 -> 0.1
                    w[164:] += sigmoid
                    w[63::-1] += sigmoid
                
            # Mask and Normalizer    
            self.mask_w = w.astype(np.float32)
            self.norm = np.sqrt(w.sum()).astype(np.float32)

        # Annotations
        assert val_room in self.annot_df['room'].unique()
        if self.mode == 'train':
            self.annot_df = self.annot_df[self.annot_df['room'] != val_room]
        else:
            self.annot_df = self.annot_df[self.annot_df['room'] == val_room]
        self.annot_df.reset_index(inplace=True, drop=True)

        # Initialize similarity matrix
        if self.mode == 'val' or self.algo == 'scene-transfer':
            if self.multi_gpu:
                if distr.get_rank() == 0:
                    self._shm, self.sim_scores_mat = self._shared_sim_mat()
                    self.sim_scores_range = self.sim_scores_mat.max() - self.sim_scores_mat.min()
            else:   
                self.sim_scores_mat = self._init_sim_matrix()
                self.sim_scores_range = self.sim_scores_mat.max() - self.sim_scores_mat.min()

    def __getitem__(self, idx: int):
        if self.algo == 'simclr':
            return self._simclr_partition(idx)
        else:
            return self._scene_transfer_partition(idx)
    
    def _simclr_partition(self, idx: int, penalty: float=0.3):
        """
        Partition the dataset in anchors and positive examples with both
        belonging to a random scene similar to SimCLR.
        """

        # Retrieve image location from the annotations dataframe 
        record = self.annot_df.iloc[idx]
        R = record['room']
        S = record['setting']
        agent = record['agent']
        ep = record['episode']
        step = record['step']        
        
        # Select the anchor scene uniformly 
        scenes = self.SCENES[self.mode]
        n = len(scenes)
        anchor_scene = np.random.choice(scenes).item()
        
        # Select the positive scene with a weighted distribution
        probs = np.ones(n)
        probs[scenes.index(anchor_scene)] = penalty
        probs /= probs.sum() 
        pos_scene = np.random.choice(scenes, p=probs).item() 
        
        # Load images from `augmented_results`
        anchor_img = Image.open(f'{self.dir}/Room{R}/Setting{S}/{agent}/episode_{ep:04}/augmented_results/aug{anchor_scene}_rgb_{step:05}.png')
        pos_img = Image.open(f'{self.dir}/Room{R}/Setting{S}/{agent}/episode_{ep:04}/augmented_results/aug{pos_scene}_rgb_{step:05}.png')
        
        # Convert images to tensors
        anchor = self.transforms(anchor_img)
        pos_ex = self.transforms(pos_img)

        # Retrieve additional information for the anchor
        lidar, gd, phi = self._info(record=record)

        if self.mode == 'val':
            # In validation return all different scenes for visualziation
            scenes = self._scenes(record)
            return anchor, scenes, pos_ex, (lidar, gd, phi)

        return anchor, pos_ex, (lidar, gd, phi)
    
    def _scene_transfer_partition(self, idx: int):
        """
        Partition the dataset in anchors, positive examples and negative examples
        following a scene-transfer Constrastive Learning approach: both anchors and 
        examples will belong to a random scene.
        """

        # Retrieve image location from the annotations dataframe 
        record = self.annot_df.iloc[idx]
        R = record['room']
        S = record['setting']
        agent = record['agent']
        ep = record['episode']
        step = record['step']
        
        # Load anchor image from `augmented_results`
        scene = np.random.choice(self.SCENES[self.mode]).item() 
        img = Image.open(f'{self.dir}/Room{R}/Setting{S}/{agent}/episode_{ep:04}/augmented_results/aug{scene}_rgb_{step:05}.png')
        anchor = self.transforms(img)

        temp_df = self.annot_df.drop(index=idx).reset_index()
        sim_scores = np.concatenate([self.sim_scores_mat[idx, :idx], self.sim_scores_mat[idx, idx+1:]])

        # Load positive examples from any other episode
        # pos_df = self.annot_df[
        #     (self.annot_df['setting'] != S) |
        #     (self.annot_df['agent'] != agent) |
        #     (self.annot_df['episode'] != ep)         
        # ].copy()

        # Similarity scores
        # pos_sim_scores = sim_scores[pos_df.index]
        # pos_df.reset_index(inplace=True, drop=True)

        # Find enough positive examples to sample from 
        cur_thresh = self.pos_thresh
        pos_recs = temp_df[sim_scores >= cur_thresh]
        while pos_recs.shape[0] == 0:
            cur_thresh -= 0.05
            pos_recs = temp_df[sim_scores >= cur_thresh]

        # Sample n_pos negative examples from all samples with score above the threshold
        pos_recs = pos_recs.sample(n=self.n_pos, random_state=self.seed, replace=True)
        pos_ex = self._load(pos_recs)
        pos_sim_scores = sim_scores[pos_recs.index]

        # Load negative examples from the same setting of the room
        # neg_df = self.annot_df[
        #     (self.annot_df['setting'] == S) &   
        #     ((self.annot_df['agent'] != agent) | (self.annot_df['episode'] != ep))         
        # ].copy()
        # neg_df.reset_index(inplace=True, drop=True)

        # Similarity scores
        # neg_sim_scores = sim_scores[neg_df.index]

        # Find enough negative examples to sample from 
        cur_thresh = self.neg_thresh
        neg_recs = temp_df[sim_scores <= cur_thresh]
        while neg_recs.shape[0] == 0:
            cur_thresh += 0.05
            neg_recs = temp_df[sim_scores <= cur_thresh]

        # Sample n_neg negative examples from all samples with score below the threshold
        neg_recs = neg_recs.sample(n=self.n_neg, random_state=self.seed, replace=True)
        neg_ex = self._load(neg_recs)
        neg_sim_scores = sim_scores[neg_recs.index]

        if self.mode == 'val':
            # In validation return all different scenes for visualziation
            scenes = self._scenes(record)
            return anchor, scenes, pos_ex, pos_sim_scores, neg_ex, neg_sim_scores
        
        # Return both anchors and respective positive/negative partitions
        return anchor, pos_ex, pos_sim_scores, neg_ex, neg_sim_scores
    
    def _annot(self):
        """
        Create a global annotations file of the dataset.
        """
        # Filter warnings
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Create a global annotations file
        annot_df = []
        for room in range(1, len(glob(f'{self.dir}/*'))+1):
            room_dir = f'{self.dir}/Room{room}'

            for setting in range(1, len(glob(f'{room_dir}/*'))+1):
                set_dir = f'{room_dir}/Setting{setting}'

                for agent_dir in glob(f'{set_dir}/*'):
                    agent = agent_dir.split('/')[-1]

                    for ep_dir in sorted(glob(f'{agent_dir}/episode_*')):     

                        ep = ep_dir.split('/')[-1]
                        try:
                            with open(f'{ep_dir}/{ep}.pkl', 'rb') as f:
                                df = pickle.load(f)

                                if ep not in df['episode'].unique():
                                    print(f'[WARN] Fixed episode in DataFrame {ep_dir}/{ep:04}.pkl not matching the name of the directory.')
                                    df['episode'] =  np.ones(df.shape[0], dtype=int) * ep

                                df.insert(0, 'agent', [agent for _ in range(df.shape[0])])
                                df.insert(0, 'setting', np.ones(df.shape[0], dtype=int) * setting)
                                df.insert(0, 'room', np.ones(df.shape[0], dtype=int) * room)
                                annot_df.append(df)
                        except FileNotFoundError:
                            print(f'[WARN] File not found: {ep_dir}/{ep}.pkl')

        annot_df = pd.concat(annot_df)
        annot_df.index = list(range(0, annot_df.shape[0]))

        if not self.multi_gpu:
            annot_df.to_pickle(f'{self.dir}/annotations.pkl')

        return annot_df
    
    def _init_sim_matrix(self, batch_size: int=1000):
        if not self.multi_gpu: print(f"\nCOMPUTING {'TRAINING' if self.mode == 'train' else 'VALIDATION'} SIMILARITY SCORES MATRIX...")

        print('Extracting LiDARs...')
        # Extract all LiDAR scans
        if self.metric in ['lidar', 'both']:
            all_lidar_scans = np.vstack(
                self.annot_df['laser_readings'].map(lambda x: x['scan'].squeeze()).to_numpy()
            ).astype(np.float32)

        print('Extracting goal info...')
        # Extract all goal information 
        if self.metric in ['goal', 'both']:
            all_goal_distances = self.annot_df.apply(lambda x: self._goal_distance(x), axis=1).to_numpy().astype(np.float32)
            all_relative_angles = self.annot_df.apply(lambda x: self._relative_angle(x), axis=1).to_numpy().astype(np.float32)

        n_samples = self.annot_df.shape[0]
        sim_scores_mat = np.ones((n_samples, n_samples), dtype=np.float32)

        for i in tqdm(range(0, n_samples, batch_size), disable=self.multi_gpu, unit='rec'):
            start, end = i, min(i + batch_size, n_samples)

            # Default similarity values
            lid_sim = 1.0
            gd_sim = 1.0
            ori_sim = 1.0

            if self.metric in ['lidar', 'both']:
                # Load batch
                batch_scans = all_lidar_scans[start:end, :]

                # Pair-wise weighted LiDAR distances
                weighted_scans = all_lidar_scans * np.sqrt(self.mask_w)
                weighted_batch = batch_scans * np.sqrt(self.mask_w)
                lid_dist = cdist(weighted_batch, weighted_scans, metric='euclidean') / self.norm
                
                # Square matrix recontruction
                # lid_dist_mat = squareform(condensed_dists) / self.norm
                lid_sim = (1 - lid_dist)

            if self.metric in ['goal', 'both']:
                # Load batches
                batch_gd = all_goal_distances[start:end]
                batch_ori = all_relative_angles[start:end]

                # Goal distance difference matrix
                gd_mat = np.abs(batch_gd[:, np.newaxis] - all_goal_distances)
                gd_sim = (1 - gd_mat)

                # Goal orientation difference matrix
                ori_diffs = batch_ori[:, np.newaxis] - all_relative_angles
                ori_diff_mat = np.abs(self._normalize_angle(ori_diffs)) / np.pi
                ori_sim = (1 - ori_diff_mat)

            # Similarity scores matrix
            batch_sim_scores = lid_sim * gd_sim * ori_sim
            sim_scores_mat[start:end, :] = batch_sim_scores
        
        # Ensure the diagonal is 1 (similarity of an element with itself)
        np.fill_diagonal(sim_scores_mat, 1.0)
        return sim_scores_mat
    
    def _init_sim_matrix_depr(self):
        """
        Initialize similarity matrix of samples in the dataset,
        given additional information in self.annot_df. 
        """

        df = self.annot_df.copy()
        shape = df.shape[0]

        lid_dist_mat = np.ones(shape=(shape, shape), dtype=np.float32)
        gd_mat = np.ones(shape=(shape, shape), dtype=np.float32)
        ori_diff_mat = np.ones(shape=(shape, shape), dtype=np.float32)

        # Compute distance matrices
        if not self.multi_gpu: print(f"\nCOMPUTING {'TRAINING' if self.mode == 'train' else 'VALIDATION'} SIMILARITY SCORES MATRIX...")
        for idx, obs in tqdm(df.iterrows(), unit='obs', disable=self.multi_gpu, total=shape):
            # Observations info
            lidar = obs['laser_readings']['scan'].squeeze()
            gd = self._goal_distance(obs)
            phi = self._relative_angle(obs)            
            
            if self.metric in ['lidar', 'both']:
                # Weighted LiDAR eucledian distance
                eucledian_dists = df['laser_readings'].map(lambda x: np.sqrt(np.sum(self.mask*(lidar - x['scan'].squeeze())**2)) / self.norm).to_numpy()
                lid_dist_mat[idx] *= eucledian_dists
            
            if self.metric in ['goal', 'both']:
                # Goal distance differences
                gd_diffs = df.apply(lambda x: abs(gd - self._goal_distance(x)), axis=1).to_numpy()
                gd_mat[idx] *= gd_diffs
                # Differences in the orientation w.r.t. the goal
                ori_diffs = df.apply(lambda x: np.abs(self._normalize_angle(phi - self._relative_angle(x))) / np.pi, axis=1)
                ori_diff_mat[idx] *= ori_diffs

        # Compute similarities
        if self.metric == 'lidar':
            sim_scores_mat = (1 - lid_dist_mat)
        elif self.metric == 'goal':
            sim_scores_mat = (1 - gd_mat)*(1 - ori_diff_mat)
        else:
            sim_scores_mat = (1 - lid_dist_mat)*(1 - gd_mat)*(1 - ori_diff_mat)
        
        return sim_scores_mat
    
    def _opposite_corner(self, x: int | float, y: int | float) -> tuple[int]:
        """
        Return the corner on the opposite quadrant w.r.t (x, y).
        ----------
        ASSUMPTION: 10mx10m room with reference point at the center.
        """

        corner_x = -5 if x >= 0 else 5
        corner_y = -5 if y >= 0 else 5
        return corner_x, corner_y 
    
    def _goal_distance(self, record: pd.Series):
        """
        Compute distance w.r.t. the goal position
        """
        # Info
        goal_x, goal_y = record['target_point_x'], record['target_point_y']
        corner_x, corner_y = self._opposite_corner(goal_x, goal_y)

        # Normalized goal distance
        max_gd = np.sqrt((goal_x - corner_x)**2 + (goal_y - corner_y)**2)
        goal_dist = np.sqrt((record['robot_pos_x']  - goal_x)**2 + (record['robot_pos_y']  - goal_y)**2)
        return goal_dist / max_gd
    
    def _normalize_angle(self, angle: float | np.ndarray):
        """
        Normalize angle in [-pi, pi]
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _relative_angle(self, record: pd.Series):
        """
        Compute orientation w.r.t. the goal position
        """
        # Info
        robot_x, robot_y = record['robot_pos_x'], record['robot_pos_y']
        goal_x, goal_y = record['target_point_x'], record['target_point_y']
        theta_r = record['robot_yaw'] 

        # Relative angle
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        theta_g = np.arctan2(dy, dx)
        return self._normalize_angle(theta_g - theta_r)
    
    def _scenes(self, record: pd.Series) -> list[torch.Tensor]:
        """
        Retrieve in-dataset augmentations of an anchor image.
        """
        R = record['room']
        S = record['setting']
        agent = record['agent']
        ep = record['episode']
        step = record['step']

        # Path to augmentations directory
        aug_dir = f'{self.dir}/Room{R}/Setting{S}/{agent}/episode_{ep:04}/augmented_results'

        # No wall
        aug_1 = self.transforms(Image.open(f'{aug_dir}/aug3_rgb_{step:05}.png'))
        # No background
        aug_2 = self.transforms(Image.open(f'{aug_dir}/aug4_rgb_{step:05}.png'))
        # Warehouse 1
        aug_3 = self.transforms(Image.open(f'{aug_dir}/aug5_rgb_{step:05}.png'))
        # Warehouse 2
        aug_4 = self.transforms(Image.open(f'{aug_dir}/aug6_rgb_{step:05}.png'))
        # Stadium
        aug_5 = self.transforms(Image.open(f'{aug_dir}/aug1_rgb_{step:05}.png'))
        # Office
        aug_6 = self.transforms(Image.open(f'{aug_dir}/aug2_rgb_{step:05}.png'))
        # Warehouse 3
        aug_7 = self.transforms(Image.open(f'{aug_dir}/aug7_rgb_{step:05}.png'))

        return torch.stack([aug_1, aug_2, aug_3, aug_4, aug_5, aug_6, aug_7])
    
    def _info(self, idx: int=None, record: pd.Series=None) -> tuple:
        """
        Retrieve additional information.
        """
        assert idx is not None or record is not None

        # Retrieve the record from the dataframe if needed        
        if record is None:
            record = self.annot_df.iloc[idx]

        # Retrieve information
        lidar = record['laser_readings']['scan'].squeeze()
        robot_x, robot_y = record['robot_pos_x'], record['robot_pos_y']
        goal_x, goal_y = record['target_point_x'], record['target_point_y']
        theta_r = record['robot_yaw']

        # Compute the maximum possible distance to the goal of the observation
        corner_x, corner_y = self._opposite_corner(goal_x, goal_y)
        max_gd = np.sqrt((goal_x - corner_x)**2 + (goal_y - corner_y)**2)
                
        # Compute normalized goal distance
        gd = np.sqrt((record['robot_pos_x'] - goal_x)**2 + (record['robot_pos_y'] - goal_y)**2) 
        gd /= max_gd

        # Compute angle with respect to goal the position (normalized in [-pi, pi])
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        theta_g = np.arctan2(dy, dx)
        phi = self._normalize_angle(theta_g - theta_r)

        return lidar, gd, phi
    
    def _sim(self, lidars: list, gds: list, angles: list):
        """
        Measure sample similarity.
        """
        assert len(lidars) == len(gds) == len(angles) == 2

        if self.metric in ['lidar', 'both']:
            # LiDAR weighted eucledian distances
            lid_dist = np.sqrt(np.sum(self.mask*(lidars[0] - lidars[1])**2)) / self.norm

        if self.metric in ['goal', 'both']:
            # Goal distances difference
            gd_diff = np.abs(gds[0] - gds[1])
            # Difference in the orientation w.r.t. the goal
            phi_diff = np.abs(self._normalize_angle(angles[0] - angles[1])) / np.pi

        if self.metric == 'lidar':
            return (1 - lid_dist)
        elif self.metric == 'goal':
            return (1 - gd_diff)*(1 - phi_diff)
        else:
            return (1 - lid_dist)*(1 - gd_diff)*(1 - phi_diff)
        
    def _load(self, records: pd.DataFrame) -> torch.Tensor:
        """
        Load examples from the dataset.
        """

        examples = []
        for _, rec in records.iterrows():
            rec_r = rec['room']
            rec_s = rec['setting']
            rec_agent = rec['agent']
            rec_ep = rec['episode']
            rec_step = rec['step']

            scene = np.random.choice(self.SCENES[self.mode]).item() if self.algo == 'scene-trasnfer' else 2 
            ex_img = Image.open(f'{self.dir}/Room{rec_r}/Setting{rec_s}/{rec_agent}/episode_{rec_ep:04}/augmented_results/aug{scene}_rgb_{rec_step:05}.png')
            examples.append(self.transforms(ex_img))

        return torch.stack(examples)
    

class AirSimDataset(ContrastiveDataset):    
    def __init__(
            self,
            dir: str,
            batch_size: int,
            micro_bsize: int,
            transforms: v2,
            algo: str='simclr',
            n_pos: int=0,
            pos_thresh: float=0.8,
            n_neg: int=0,
            neg_thresh: float=0.2,
            augmentations: list=None,
            mode: str='train',
            multi_gpu: bool=False,
            seed: int=42
        ):
        """
        Torch implementation of Contrastive Dataset for AirSim Drone Navigation.
        ----------
        Parameters:
        - dir: str            - directory of the dataset
        - algo: str           - Contrastive Learning framework
        - n_pos: int          - number of positive examples (scene-transfer framework)
        - pos_thresh: float   - positive similarity threshold (scene-transfer framework)
        - n_neg: int          - number of negative examples (scene-transfer framework)
        - neg_thresh: float   - negative similarity threshold (scene-transfer framework)
        - batch_size: int     - size of the batch returned by the DataLoader
        - micro_bsize: int    - size of the micro-batch for gradient accumulation
        - transforms: v2      - image transformations to apply
        - augmentations       - additional augmentations for positive examples
        - mode: str           - dataset mode (train or val)
        - multi_gpu: bool     - whether the dataset is used for training a model on multiple GPUs
        - seed: int           - random seed for reproducibility
        """
        super().__init__(
            dir=dir,
            transforms=transforms,
            algo=algo,
            n_pos=n_pos,
            pos_thresh=pos_thresh,
            n_neg=n_neg,
            neg_thresh=neg_thresh,
            batch_size=batch_size,
            micro_bsize=micro_bsize,
            augmentations=augmentations,
            mode=mode,
            multi_gpu=multi_gpu,
            seed=seed
        )
        self.samples = []

        print("Pre-caching dataset paths...")
        # Trova tutte le cartelle anchor_XXXXX
        anchor_dirs = sorted(glob.glob(os.path.join(self.dir, "anchor_*")))

        for anchor_dir in anchor_dirs:
            anchor_path = os.path.join(anchor_dir, "anchor.png")
            positive_paths = glob.glob(os.path.join(anchor_dir, "positive_*.png"))

            if os.path.exists(anchor_path) and positive_paths:
                self.samples.append((anchor_path, positive_paths))

        print(f"Found {len(self.samples)} valid anchor/positive pairs.")

        if len(self.samples) == 0:
            raise ValueError(f"No valid anchor/positive pairs found in {self.dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.algo == 'simclr':
            return self._simclr_partition(idx)
        else:
            raise NotImplementedError(f'Contrastive Learning framework {self.algo} has not been implemented, yet.')        
    
    def _simclr_partition(self, idx: int):
        anchor_path, positive_paths = self.samples[idx]

        # Carica anchor
        anchor_img = Image.open(anchor_path).convert('RGB')

        # Scegli un positivo casuale dalla lista pre-caricata
        positive_path = np.random.choice(positive_paths)
        positive_img = Image.open(positive_path).convert('RGB')

        # Applica trasformazioni se specificate
        if self.transforms:
            anchor_img = self.transforms(anchor_img)
            positive_img = self.transforms(positive_img)

        return {
            'anchor': anchor_img,
            'positive': positive_img,
            'anchor_dir': os.path.basename(os.path.dirname(anchor_path))
        }