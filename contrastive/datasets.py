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
from PIL import Image
from multiprocessing import shared_memory  
from glob import glob
from tqdm.auto import tqdm
from abc import ABC, abstractmethod


class ContrastiveDataset(Dataset, ABC):
    def __init__(
            self, 
            dir: str,
            metric: str,
            mask: str,
            shift: float,  
            transforms: v2,
            augmentations: list=None,
            mode: str='train',
            multi_gpu: bool=False,
            seed: int=42
        ):
        """
        Torch implementation of Contrastive Dataset for Robotic Navigation.
        ----------
        Parameters:
        - dir: str            - directory of the dataset
        - metric: str         - metric for computing similarity (lidar, goal, both)
        - mask: str           - LiDAR readings mask type
        - transforms: v2      - image transformations to apply
        - augmentations       - augmentations for positive examples
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

        # Dataset mode
        assert mode in ['train', 'val']
        self.mode = mode

        # Distributed training
        self.multi_gpu = multi_gpu            

        # Annotations dataframe
        assert os.path.exists(self.dir)
        try:
            with open(f'{self.dir}/annotations.pkl', 'rb') as f:
                self.annot_df = pickle.load(f)
        except FileNotFoundError:
            print(f"{'Creating annotations file...' if self.multi_gpu else f'[GPU:{distr.get_rank()}] Retrieving additional info...'}")
            self.annot_df = self._annot()

        # Pandas methods will be used on this dataframe
        assert isinstance(self.annot_df, pd.DataFrame)

        # Similarity metric
        assert metric in ['lidar', 'goal', 'both']
        self.metric = metric
        if metric in ['lidar', 'both']:
            assert mask in ['naive', 'binary', 'soft']

            # Define LiDAR readings mask
            rand_sample = self.annot_df.sample(n=1).iloc[0]
            w = np.zeros(rand_sample['laser_readings']['scan'].squeeze().shape[0])
            match mask:
                case 'naive':
                    w += 1
                case 'binary':
                    # In FOV readings
                    w[64:164] += 1
                case 'soft':
                    assert shift is not None
                    # In FOV readings
                    w[64:164] += 1
                    # Out of FOV readings
                    x = np.linspace(0.0, 1.0, w[164:].shape[0])
                    sigmoid = 1 - 0.9*(1 / (1+np.exp(-x + shift))) # Sigmoid 1.0 -> 0.1
                    w[164:] += sigmoid
                    w[63::-1] += sigmoid
                
            # Mask and Normalizer    
            self.mask = w
            self.norm = np.sqrt(w.sum())

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
    
    def _normalize_angle(self, angle: float):
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
    
    def _init_sim_matrix(self):
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
            shm = shared_memory.SharedMemory(name='sim_scores_mat', create=True, size=size)
            shared_mat = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
            shared_mat[:] = sim_scores_mat[:] # Copy on shared memory
        except FileExistsError:
            shm = shared_memory.SharedMemory(name='sim_scores_mat', create=False)
            shared_mat = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)

        return shm, shared_mat
    
    def _seed_worker(self, worker_id):
        """
        Set the random seed for each worker.
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    def get_DataLoader(self, batch_size: int, num_workers: int=None) -> DataLoader:
        """ 
        Return the torch DataLoader of the dataset.
        """
        if self.multi_gpu:
            batch_size = batch_size if self.mode == 'train' else batch_size // 2
            return DataLoader(
                dataset=self,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(self.mode == 'train'),
                sampler=DistributedSampler(self)
            )
        else:
            return DataLoader(
                dataset=self,
                batch_size=batch_size,
                shuffle=(self.mode == 'train'),
                num_workers=num_workers,
                worker_init_fn=self._seed_worker,
                pin_memory=True,
                drop_last=(self.mode == 'train'),
            )

    def set_shared_sim_mat(self):
        if not hasattr(self, 'sim_scores_mat'):
            try:
                shm = shared_memory.SharedMemory(name='sim_scores_mat', create=False)
                self.sim_scores_mat = np.ndarray(shape=(self.annot_df.shape[0], self.annot_df.shape[0]), dtype=np.float32, buffer=shm.buf)
                self._shm = shm
            except FileNotFoundError:
                print(f'Could not find shared memory block named `sim_scores_mat`.')
    

class WithAugmentationsDataset(ContrastiveDataset):
    def __init__(
            self,
            dir: str,
            metric: str,
            mask: str,
            shift: float,
            n_pos: int,
            pos_thresh: float,
            n_neg: int,
            neg_thresh: float,
            val_room: int,
            transforms: v2,
            augmentations: list=None,
            mode: str='train',
            multi_gpu: bool=False,
            seed: int=42
        ):
        """
        Torch implementation of with-augmentations dataset.
        ----------
        Parameters:
        - dir: str            - directory of the dataset
        - metric: str         - metric for computing sample similarity (lidar, goal, both)
        - mask: str           - LiDAR readings mask type
        - n_pos: int          - number of positive examples
        - pos_thresh: float   - positive similarity threshold
        - n_neg: int          - number of negative examples
        - neg_thresh: float   - negative similarity threshold
        - val_room: int       - validation room
        - transforms: v2      - image transformations to apply
        - augmentations       - additional augmentations for positive examples
        - mode: str           - dataset mode (train or val)
        - multi_gpu: bool     - whether the dataset is used for training a model on multiple GPUs
        - seed: int           - random seed for reproducibility
        """
        super().__init__(
            dir=dir,
            metric=metric,
            mask=mask,
            shift=shift,
            transforms=transforms,
            augmentations=augmentations,
            mode=mode,
            multi_gpu=multi_gpu,
            seed=seed
        )

        # Positive examples
        self.n_pos = n_pos
        self.pos_thresh = pos_thresh
        
        # Negative examples
        self.n_neg = n_neg
        self.neg_thresh = neg_thresh

        # Annotations
        assert val_room in self.annot_df['room'].unique()
        val_sets = np.sort(self.annot_df['setting'].unique())[-5:]
        if self.mode == 'train':
            self.annot_df = self.annot_df[
                (self.annot_df['room'] != val_room) |
                (self.annot_df['setting'].map(lambda x: x not in val_sets))
            ]
        else:
            self.annot_df = self.annot_df[
                (self.annot_df['room'] == val_room) &
                (self.annot_df['setting'].map(lambda x: x in val_sets))
            ]
        self.annot_df.reset_index(inplace=True, drop=True)

        # Initialize similarity matrix
        if self.mode == 'val':
            self.sim_scores_mat = self._init_sim_matrix()
            self.sim_scores_range = self.sim_scores_mat.max() - self.sim_scores_mat.min()

    def __getitem__(self, idx: int):
        # Retrieve image location from the annotations dataframe 
        record = self.annot_df.iloc[idx]
        R = record['room']
        S = record['setting']
        ep = record['episode']
        step = record['step']
        
        # Load anchor image from `augmented_results`
        img = Image.open(f'{self.dir}/Room{R}/Setting{S}/episode_{ep:04}/augmented_results/aug2_rgb_{step:05}.png')
        anchor = self.transforms(img)

        # Augmentations of the anchor image
        augs = [augm(img) for augm in self.augmentations] + self._augs(record)
        pos_ex = torch.stack(augs)
        pos_sim_scores = np.ones(shape=(pos_ex.shape[0],))
        
        # Retrieve additional information for the anchor
        anc_lidar, anc_gd, anc_phi = self._info(record=record)

        if self.n_pos > 0:
            # Load positive examples from any other episode
            df = self.annot_df[
                (self.annot_df['room'] != R) |
                (self.annot_df['setting'] != S) |   
                (self.annot_df['episode'] != ep)         
            ].copy()
            df.reset_index(inplace=True, drop=True)

            # Similarity scores
            sim_scores = np.ones(shape=(df.shape[0],))          

            for i in df.index:
                lidar, gd, phi = self._info(i)
                sim_scores[i] *= self._sim(lidars=[anc_lidar, lidar], gds=[anc_gd, gd], angles=[anc_phi, phi])

            # Sample n_pos negative examples from all samples with score above the threshold
            pos_recs = df[sim_scores >= self.pos_thresh].sample(n=self.n_pos, random_state=self.seed)
            pos_ex = torch.cat([pos_ex, self._load(pos_recs)])
            pos_sim_scores = np.concat([pos_sim_scores, sim_scores[pos_recs.index]])

            # Free space
            del df, sim_scores           
        
        if self.n_neg > 0:
            # Load negative examples from the same setting of the room
            df = self.annot_df[
                (self.annot_df['room'] == R) &
                (self.annot_df['setting'] == S) &   
                (self.annot_df['episode'] != ep)         
            ].copy()
            df.reset_index(inplace=True, drop=True)

            # Similarity scores
            sim_scores = np.ones(shape=(df.shape[0],))          

            for i in df.index:
                lidar, gd, phi = self._info(i)
                sim_scores[i] *= self._sim(lidars=[anc_lidar, lidar], gds=[anc_gd, gd], angles=[anc_phi, phi])

            # Sample n_neg negative examples from all samples with score below the threshold
            neg_recs = df[sim_scores <= self.neg_thresh].sample(n=self.n_neg, random_state=self.seed)
            neg_ex = self._load(neg_recs)
            neg_sim_scores = sim_scores[neg_recs.index]

            # Free space
            del df, sim_scores

            return anchor, pos_ex, pos_sim_scores, neg_ex, neg_sim_scores

        # Return additional information to the anchor for in-batch similarities
        return anchor, pos_ex, pos_sim_scores, anc_lidar, anc_gd, anc_phi
    
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
                for ep_dir in sorted(glob(f'{set_dir}/episode_*')):     

                    ep = ep_dir.split('/')[-1]
                    try:
                        with open(f'{ep_dir}/{ep}.pkl', 'rb') as f:
                            df = pickle.load(f)
                            df.insert(0, 'setting', np.ones(df.shape[0], dtype=int) * setting)
                            df.insert(0, 'room', np.ones(df.shape[0], dtype=int) * room)
                            annot_df.append(df)
                    except FileNotFoundError:
                        print(f'File not found: {ep_dir}/{ep}.pkl')

        annot_df = pd.concat(annot_df)
        annot_df.index = list(range(0, annot_df.shape[0]))

        if not self.multi_gpu:
            annot_df.to_pickle(f'{self.dir}/annotations.pkl')

        return annot_df
    
    def _augs(self, record: pd.Series) -> list[torch.Tensor]:
        """
        Retrieve in-dataset augmentations of an anchor image.
        """
        R = record['room']
        S = record['setting']
        ep = record['episode']
        step = record['step']

        # Path to augmentations directory
        aug_dir = f'{self.dir}/Room{R}/Setting{S}/episode_{ep:04}/augmented_results'

        # No wall
        aug_1 = self.transforms(Image.open(f'{aug_dir}/aug3_rgb_{step:05}.png'))
        # No background
        aug_2 = self.transforms(Image.open(f'{aug_dir}/aug4_rgb_{step:05}.png'))
        # Warehouse 1
        aug_3 = self.transforms(Image.open(f'{aug_dir}/aug5_rgb_{step:05}.png'))
        # Warehouse 2
        aug_4 = self.transforms(Image.open(f'{aug_dir}/aug6_rgb_{step:05}.png'))

        # Training augmentations
        augs = [aug_1, aug_2, aug_3, aug_4]

        if self.mode == 'val':
            # Stadium
            aug_5 = self.transforms(Image.open(f'{aug_dir}/aug1_rgb_{step:05}.png'))
            # Warehouse 3
            aug_6 = self.transforms(Image.open(f'{aug_dir}/aug7_rgb_{step:05}.png'))

            # Include hold-out scenes for validation
            augs.extend([aug_5, aug_6])

        return augs
    
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

        # LiDAR weighted eucledian distances
        lid_dist = np.sqrt(np.sum(self.mask*(lidars[0] - lidars[1])**2)) / self.norm
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
            rec_ep = rec['episode']
            rec_step = rec['step']

            ex_img = Image.open(f'{self.dir}/Room{rec_r}/Setting{rec_s}/episode_{rec_ep:04}/augmented_results/aug2_rgb_{rec_step:05}.png')
            examples.append(self.transforms(ex_img))

        return torch.stack(examples)
    

class RoomAllAgentsDataset(ContrastiveDataset):
    def __init__(
            self,
            dir: str,
            metric: str,
            mask: str,
            shift: float,
            n_pos: int,
            pos_thresh: float,
            n_neg: int,
            neg_thresh: float,
            val_room: int,
            transforms: v2,
            augmentations: list=None,
            mode: str='train',
            multi_gpu: bool=False,
            seed: int=42
        ):
        """
        Torch implementation of with-augmentations dataset.
        ----------
        Parameters:
        - dir: str            - directory of the dataset
        - metric: str         - metric for computing sample similarity (lidar, goal, both)
        - mask: str           - LiDAR readings mask type
        - n_pos: int          - number of positive examples
        - pos_thresh: float   - positive similarity threshold
        - n_neg: int          - number of negative examples
        - neg_thresh: float   - negative similarity threshold
        - val_room: int       - validation room
        - transforms: v2      - image transformations to apply
        - augmentations       - additional augmentations for positive examples
        - mode: str           - dataset mode (train or val)
        - multi_gpu: bool     - whether the dataset is used for training a model on multiple GPUs
        - seed: int           - random seed for reproducibility
        """
        super().__init__(
            dir=dir,
            metric=metric,
            mask=mask,
            shift=shift,
            transforms=transforms,
            augmentations=augmentations,
            mode=mode,
            multi_gpu=multi_gpu,
            seed=seed
        )

        # Positive examples
        self.n_pos = n_pos
        self.pos_thresh = pos_thresh
        
        # Negative examples
        self.n_neg = n_neg
        self.neg_thresh = neg_thresh

        # Annotations
        assert val_room in self.annot_df['room'].unique()
        if self.mode == 'train':
            self.annot_df = self.annot_df[self.annot_df['room'] != val_room]
        else:
            self.annot_df = self.annot_df[self.annot_df['room'] == val_room]
        self.annot_df.reset_index(inplace=True, drop=True)

        # Initialize similarity matrix
        if self.mode == 'val':
            if self.multi_gpu:
                if distr.get_rank() == 0:
                    self._shm, self.sim_scores_mat = self._shared_sim_mat()
                    self.sim_scores_range = self.sim_scores_mat.max() - self.sim_scores_mat.min()
            else:   
                self.sim_scores_mat = self._init_sim_matrix()
                self.sim_scores_range = self.sim_scores_mat.max() - self.sim_scores_mat.min()

    def __getitem__(self, idx: int):
        # Retrieve image location from the annotations dataframe 
        record = self.annot_df.iloc[idx]
        R = record['room']
        S = record['setting']
        agent = record['agent']
        ep = record['episode']
        step = record['step']
        
        # Load anchor image from `augmented_results`
        img = Image.open(f'{self.dir}/Room{R}/Setting{S}/{agent}/episode_{ep:04}/augmented_results/aug2_rgb_{step:05}.png')
        anchor = self.transforms(img)

        # Augmentations of the anchor image
        augs = [augm(img) for augm in self.augmentations] + self._augs(record)
        pos_ex = torch.stack(augs)
        pos_sim_scores = np.ones(shape=(pos_ex.shape[0],))
        
        # Retrieve additional information for the anchor
        anc_lidar, anc_gd, anc_phi = self._info(record=record)

        if self.n_pos > 0:
            # Load positive examples from any other episode
            df = self.annot_df[
                (self.annot_df['setting'] != S) |
                (self.annot_df['agent'] != agent) |
                (self.annot_df['episode'] != ep)         
            ].copy()
            df.reset_index(inplace=True, drop=True)

            # Similarity scores
            sim_scores = np.ones(shape=(df.shape[0],))          

            for i in df.index:
                lidar, gd, phi = self._info(i)
                sim_scores[i] *= self._sim(lidars=[anc_lidar, lidar], gds=[anc_gd, gd], angles=[anc_phi, phi])

            # Find enough negative examples to sample from 
            cur_thresh = self.pos_thresh
            pos_recs = df[sim_scores >= cur_thresh]
            while pos_recs.shape[0] < self.n_pos*2:
                cur_thresh -= 0.01
                pos_recs = df[sim_scores >= cur_thresh]

            # Sample n_pos negative examples from all samples with score above the threshold
            pos_recs = pos_recs.sample(n=self.n_pos, random_state=self.seed)
            pos_ex = torch.cat([pos_ex, self._load(pos_recs)])
            pos_sim_scores = np.concat([pos_sim_scores, sim_scores[pos_recs.index]])          
        
        if self.n_neg > 0:
            # Load negative examples from the same setting of the room
            df = self.annot_df[
                (self.annot_df['setting'] == S) &   
                ((self.annot_df['agent'] != agent) | (self.annot_df['episode'] != ep))         
            ].copy()
            df.reset_index(inplace=True, drop=True)

            # Similarity scores
            sim_scores = np.ones(shape=(df.shape[0],))          

            for i in df.index:
                lidar, gd, phi = self._info(i)
                sim_scores[i] *= self._sim(lidars=[anc_lidar, lidar], gds=[anc_gd, gd], angles=[anc_phi, phi])

            # Find enough negative examples to sample from 
            cur_thresh = self.neg_thresh
            neg_recs = df[sim_scores <= cur_thresh]
            while neg_recs.shape[0] < self.n_neg*2:
                cur_thresh += 0.01
                neg_recs = df[sim_scores <= cur_thresh]

            # Sample n_neg negative examples from all samples with score below the threshold
            neg_recs = neg_recs.sample(n=self.n_neg, random_state=self.seed)
            neg_ex = self._load(neg_recs)
            neg_sim_scores = sim_scores[neg_recs.index]

            # Return both anchors and respective positive/negative partitions
            return anchor, pos_ex, pos_sim_scores, neg_ex, neg_sim_scores

        # Return additional information to the anchor for in-batch similarities
        return anchor, pos_ex, pos_sim_scores, anc_lidar, anc_gd, anc_phi
    
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
    
    def _augs(self, record: pd.Series) -> list[torch.Tensor]:
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

        # Training augmentations
        augs = [aug_1, aug_2, aug_3, aug_4]

        if self.mode == 'val':
            # Stadium
            aug_5 = self.transforms(Image.open(f'{aug_dir}/aug1_rgb_{step:05}.png'))
            # Warehouse 3
            aug_6 = self.transforms(Image.open(f'{aug_dir}/aug7_rgb_{step:05}.png'))

            # Include hold-out scenes for validation
            augs.extend([aug_5, aug_6])

        return augs
    
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

            ex_img = Image.open(f'{self.dir}/Room{rec_r}/Setting{rec_s}/{rec_agent}/episode_{rec_ep:04}/augmented_results/aug2_rgb_{rec_step:05}.png')
            examples.append(self.transforms(ex_img))

        return torch.stack(examples)