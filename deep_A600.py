import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from multiprocessing import Pool, cpu_count
import re
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from lpips import LPIPS
import logging
import nibabel as nib
import kagglehub
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau
from torch.nn.utils import spectral_norm
from torch_optimizer import Lookahead
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.distributions import Normal
import random
from typing import Tuple
from torch.nn import functional as F
from math import sqrt
import copy
from timm.models import resnet
from timm.models import vision_transformer
from torchmetrics import Dice, JaccardIndex
import torch.distributed as dist
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.nn.functional import interpolate
from skimage.restoration import denoise_nl_means
from skimage import img_as_float
from typing import List
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from torchvision.transforms import functional as TF
from torch_optimizer import RAdam
from torch.optim.lr_scheduler import CyclicLR
from itertools import cycle
from collections import deque


class Config:
    # Hyperparameters
    BATCH_SIZE = 16
    IMAGE_SIZE = 128
    LOW_RES_SIZE = IMAGE_SIZE // 4
    CHANNELS = 1
    EPOCHS = 300
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIXED_PRECISION = True
    
    NUM_RESIDUAL_BLOCKS = 8
    GRADIENT_ACCUMULATION_STEPS = 8
    LAMBDA_ADV = 0.001
    LAMBDA_PERC = 0.1
    LAMBDA_CONTENT = 1.0
    LAMBDA_SSIM = 0.1
    LAMBDA_FEATURE = 0.1
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.2
    WARMUP_STEPS = 200
    
    # Model Parameters
    NUM_HEADS = 8
    EMBED_DIM = 64  # Base embedding dim, can be modified
    
    # Data loading
    NUM_WORKERS = 12
    PIN_MEMORY = True
    
    # Augmentation parameters
    AFFINE_DEGREES = 15
    AFFINE_TRANSLATE = (0.1, 0.1)
    AFFINE_SCALE = (0.8, 1.2)
    
    # Directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_FOLDER = os.path.join(BASE_DIR, "brats_dataset")
    TRAIN_HR_FOLDER = os.path.join(DATASET_FOLDER, "train_HR")
    TRAIN_LR_FOLDER = os.path.join(DATASET_FOLDER, "train_LR")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    GENERATED_DIR = os.path.join(BASE_DIR, "generated_medical")
    
    # Distributed Training
    DISTRIBUTED = False
    LOCAL_RANK = None
    WORLD_SIZE = None
    
    # Loss weights
    ADV_WEIGHT = 0.1
    CONTENT_WEIGHT = 1.0
    PERCEPTUAL_WEIGHT = 0.1
    SSIM_WEIGHT = 0.1
    SEG_WEIGHT = 0.1 # Weight for segmentation loss

    # Perceptual loss weights
    PERC_LAYER_WEIGHTS = [1, 0.5, 0.25, 0.125]  # Weights for the different layers

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15  # how many epochs without improvement to stop
    EARLY_STOPPING_MIN_DELTA = 1e-4
    
    # Cross validation
    NUM_FOLDS = 5

    # Self Supervised learning parameters
    SELF_SUPERVISED_EPOCHS = 50
    SELF_SUPERVISED_BATCH_SIZE = 16

    # Data Preprocessing
    DATA_NORM_MEAN = 0.5
    DATA_NORM_STD = 0.5

    # Activation Analysis
    ACTIVATION_LAYERS = [1, 7]  # Layers to visualize in the generator
    ACTIVATION_SAMPLE_COUNT = 4  # Number of images to use for activation visualization

    # Cyclical LR
    CYCLE_LR_BASE_LR = 1e-5
    CYCLE_LR_MAX_LR = 1e-4
    CYCLE_LR_STEP_SIZE_UP = 1000
    CYCLE_LR_MODE = "triangular2"

    # Progressive Growing
    PROGRESSIVE_GROW_STEPS = 5  # Number of steps at each image size.

    # Data Subsetting
    SUBSET_SIZE = 0.2

    # Quantile Clipping
    QUANTILE_CLIP_MIN = 0.01
    QUANTILE_CLIP_MAX = 0.99
    
    @staticmethod
    def set_dataset_folder(path):
        Config.DATASET_FOLDER = path
        Config.TRAIN_HR_FOLDER = os.path.join(path, "train_HR")
        Config.TRAIN_LR_FOLDER = os.path.join(path, "train_LR")

import os
import torch
import numpy as np
import nibabel as nib
import logging
import re
from PIL import Image
from torchvision import transforms
from skimage.restoration import denoise_nl_means
from skimage import img_as_float
from scipy.ndimage import affine_transform
from multiprocessing import Pool, cpu_count  # Fixed import here
from deep_A600 import Config  # Import Config from your original script
import math
from itertools import islice


def resize_with_padding_numpy(image_array, target_size, blur_sigma=0.5):
     """Resizes and pads image array with numpy only"""
     height, width = image_array.shape
     target_height, target_width = target_size
     
     target_ratio = target_width / target_height
     img_ratio = width / height
     
     if img_ratio > target_ratio:
        new_width = target_width
        new_height = int(new_width / img_ratio)
     else:
        new_height = target_height
        new_width = int(new_height * img_ratio)
     
     resized_img = np.array(Image.fromarray(image_array, mode="L").resize((new_width, new_height), Image.BICUBIC)) # Explicit L mode

     padded_image = np.zeros(target_size, dtype=resized_img.dtype)
    
     left = (target_width - new_width) // 2
     top = (target_height - new_height) // 2
     
     padded_image[top:top+new_height, left:left+new_width] = resized_img

     return padded_image


def apply_affine_transform(image_array, affine_matrix):
    """Applies an affine transformation to an image array."""
    transformed_image = affine_transform(image_array, affine_matrix)
    return transformed_image

def process_file_chunk(args):
    (file_path, image_size, use_segmentation, hr_folder, lr_folder) = args
    try:
        logging.info(f"Preprocessing: Started processing: {file_path}")
        img = nib.load(file_path)
        img_data = np.asanyarray(img.get_fdata())
        
        # Normalize data to [0, 1]
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        
        # Extract modality from filename
        modality = os.path.basename(file_path).split("_")[-1].split(".")[0]

        # Ensure 3D array and take all channels if present
        if img_data.ndim == 4:
            num_channels = img_data.shape[3]
        else:
            num_channels = 1
            img_data = np.expand_dims(img_data, axis=3)

        num_slices_processed = 0
        for channel_idx in range(num_channels):
            for slice_idx in range(img_data.shape[2]):
                slice_img = img_data[:, :, slice_idx, channel_idx]

                # Quantile Clipping
                lower_quantile = np.quantile(slice_img, Config.QUANTILE_CLIP_MIN)
                upper_quantile = np.quantile(slice_img, Config.QUANTILE_CLIP_MAX)
                slice_img = np.clip(slice_img, lower_quantile, upper_quantile)

                # Affine Transformation before resizing
                if image_size is not None:
                    angle = np.random.uniform(-Config.AFFINE_DEGREES, Config.AFFINE_DEGREES)
                    scale = np.random.uniform(Config.AFFINE_SCALE[0], Config.AFFINE_SCALE[1])
                    translate = (
                        np.random.uniform(-Config.AFFINE_TRANSLATE[0], Config.AFFINE_TRANSLATE[0]),
                        np.random.uniform(-Config.AFFINE_TRANSLATE[1], Config.AFFINE_TRANSLATE[1])
                    )

                    # Create a transformation matrix
                    center = np.array(slice_img.shape) / 2.0
                    transform_matrix = np.array([
                        [scale * np.cos(np.deg2rad(angle)), -scale * np.sin(np.deg2rad(angle)), translate[0]],
                        [scale * np.sin(np.deg2rad(angle)), scale * np.cos(np.deg2rad(angle)), translate[1]],
                        [0, 0, 1]
                    ])
                    transform_matrix[0, 2] += -center[0] * transform_matrix[0, 0] - center[1] * transform_matrix[0, 1] + center[0]
                    transform_matrix[1, 2] += -center[0] * transform_matrix[1, 0] - center[1] * transform_matrix[1, 1] + center[1]
                    slice_img = apply_affine_transform(slice_img, transform_matrix[:2, :2].T)
                
                if image_size is None:
                    hr_img = resize_with_padding_numpy(slice_img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), blur_sigma=0.5)
                    lr_img = resize_with_padding_numpy(slice_img, (Config.LOW_RES_SIZE, Config.LOW_RES_SIZE), blur_sigma=0.25)
                else:
                    hr_img = resize_with_padding_numpy(slice_img, (image_size, image_size), blur_sigma=0.5)
                    lr_img = resize_with_padding_numpy(slice_img, (image_size // 4, image_size // 4), blur_sigma=0.25)

                # Convert to torch tensors
                hr_tensor = torch.tensor(hr_img, dtype=torch.float32).unsqueeze(0)
                lr_tensor = torch.tensor(lr_img, dtype=torch.float32).unsqueeze(0)

                # Save tensors
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                hr_file_path = os.path.join(hr_folder, f'{base_name}_{slice_idx}_{channel_idx}_HR.pt')
                lr_file_path = os.path.join(lr_folder, f'{base_name}_{slice_idx}_{channel_idx}_LR.pt')
                
                torch.save(hr_tensor, hr_file_path)
                torch.save(lr_tensor, lr_file_path)
                
                # Segmentation masks
                if use_segmentation and modality == "flair":
                    seg_img = slice_img > 0.3  # Create a binary segmentation mask
                    if image_size is None:
                        seg_img = resize_with_padding_numpy(seg_img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), blur_sigma=0.5)
                    else:
                        seg_img = resize_with_padding_numpy(seg_img, (image_size, image_size), blur_sigma=0.5)
                    seg_tensor = torch.tensor(seg_img, dtype=torch.float32).unsqueeze(0)
                    seg_file_path = os.path.join(hr_folder, f'{base_name}_{slice_idx}_{channel_idx}_SEG.pt')
                    torch.save(seg_tensor, seg_file_path)
                
                num_slices_processed += 1
        
        logging.info(f"Preprocessing: Finished processing: {file_path}")
        return num_slices_processed
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return 0
def preprocess_dataset(use_segmentation=False, image_size=None):
    hr_folder = Config.TRAIN_HR_FOLDER
    lr_folder = Config.TRAIN_LR_FOLDER
    os.makedirs(hr_folder, exist_ok=True)
    os.makedirs(lr_folder, exist_ok=True)
    
    num_images = 0
    
    logging.info(f"Preprocessing: Scanning dataset folder: {Config.DATASET_FOLDER}")
    
    # Use all available CPU cores for processing.
    num_processes = cpu_count()
    logging.info(f"Preprocessing: Using {num_processes} processes")
    with Pool(processes = num_processes) as pool:
        tasks = []
        try:
            for dirpath, _, filenames in os.walk(Config.DATASET_FOLDER):
                logging.info(f"Preprocessing: Files in current folder: {dirpath} , Files found: {filenames}") #Added log for every folder found.
                #Only look in training data path
                if "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" not in dirpath:
                    continue
               
                #Gather all files that match the pattern
                file_paths = [os.path.join(dirpath, file_name) for file_name in filenames if re.match(r'^\d+_brain_(flair|t1|t1ce|t2).nii', file_name)]

                for file_path in file_paths: #Iterate throught the list
                      tasks.append((file_path, image_size, use_segmentation, hr_folder, lr_folder)) # Appends single files

            results = pool.imap_unordered(process_file_chunk, tasks) # Changed to imap

            num_images = sum(results)
    
        except Exception as e:
            logging.error(f"Failed to process dataset: {e}")
            raise
            
    logging.info(f"Preprocessing: Created {num_images} total image pairs.")
    return num_images

def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        Config.DISTRIBUTED = True
        Config.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(Config.LOCAL_RANK)
        init_process_group(backend="nccl")
        Config.WORLD_SIZE = torch.distributed.get_world_size()
        logging.info(f"Distributed training enabled. Rank: {Config.LOCAL_RANK}, World size: {Config.WORLD_SIZE}")
    else:
        logging.info("Distributed training not enabled")


def cleanup_distributed():
    if Config.DISTRIBUTED:
        destroy_process_group()


def is_main_process():
    return not Config.DISTRIBUTED or Config.LOCAL_RANK == 0


def setup_logging():
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    log_filename = os.path.join(Config.LOG_DIR, 'training.log') if is_main_process() else None
    logging.basicConfig(
        filename=log_filename,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    if is_main_process():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)


def log_memory_usage():
    if torch.cuda.is_available():
        logging.info(
            f"GPU Memory - Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
            f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB"
        )


def download_brats_dataset():
    if is_main_process():
        logging.info("Downloading BraTS dataset from Kaggle...")
        try:
            path = kagglehub.dataset_download("shakilrana/brats-2023-adult-glioma")
            logging.info(f"Dataset downloaded to: {path}")
            return path
        except Exception as e:
            logging.error(f"Failed to download dataset: {e}")
            raise
    else:
        logging.info("Waiting for main process to download dataset...")
        torch.distributed.barrier()
        return Config.BASE_DIR + "/brats-2023-adult-glioma"


import os
import torch
import numpy as np
import nibabel as nib
import logging
import re
from PIL import Image
from torchvision import transforms
from skimage.restoration import denoise_nl_means
from skimage import img_as_float
from scipy.ndimage import affine_transform
from multiprocessing import Pool, cpu_count  # Fixed import here
from deep_A600 import Config  # Import Config from your original script
import math
from itertools import islice


def apply_transform(tensor, transform):
    if transform is None:
        return tensor
    
    if not isinstance(tensor, torch.Tensor):
        logging.error(f"Invalid type provided for transformation: {type(tensor)}, skipping transformation")
        return tensor
    
    # Ensure the tensor has 3 dimensions (channels, height, width)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # Add channel dimension
    elif tensor.ndim != 3:
        logging.error(f"Invalid tensor dimensions: {tensor.shape}, skipping transformation")
        return tensor
    
    try:
        seed = torch.randint(0, 2**32, (1,)).item()
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)
        tensor = transform(tensor)
    except Exception as e:
        logging.error(f"Error during transformation: {e}, skipping transformation")
    
    if not isinstance(tensor, torch.Tensor):
        logging.error(f"Tensor is no longer a tensor: {type(tensor)}, skipping transformation")
        return tensor
    
    if tensor.ndim != 3:
        logging.error(f"Invalid output tensor dimensions after transform: {tensor.shape}, skipping transformation")
        return tensor

    return tensor


class BraTSDataset(Dataset):
    def __init__(self, hr_folder, lr_folder, transform=True, norm=True, subset = 1.0, progressive = False, image_size = None,use_segmentation=True):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.norm = norm
        self.subset = subset
        self.progressive = progressive
        self.image_size = image_size
        self.use_segmentation = use_segmentation  

        
        hr_files = sorted([f for f in os.listdir(hr_folder) if f.endswith('_HR.pt')])
        lr_files = sorted([f for f in os.listdir(lr_folder) if f.endswith('_LR.pt')])
        
        self.image_pairs = [(
            os.path.join(hr_folder, hr_file),
            os.path.join(lr_folder, hr_file.replace('_HR.pt', '_LR.pt'))
        ) for hr_file in hr_files if hr_file.replace('_HR.pt', '_LR.pt') in lr_files]
        
        if self.subset < 1.0:
             num_subset = int(len(self.image_pairs) * self.subset)
             self.image_pairs = self.image_pairs[:num_subset]
        
        if transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(Config.AFFINE_DEGREES),
                transforms.RandomAffine(degrees=Config.AFFINE_DEGREES, translate=Config.AFFINE_TRANSLATE, scale=Config.AFFINE_SCALE),
                RandomElasticDeformation(alpha=10, sigma=3, p=0.5),
                RandomApply(GaussianBlur(kernel_size=3, sigma=(0.1, 2)), p=0.2),
                RandomApply(AddGaussianNoise(0, 0.1), p=0.3),
                RandomApply(Cutout(length=32), p=0.2),
                RandomMixUp(p = 0.2),
                RandomCutMix(p = 0.2)
            ])
        else:
            self.transform = None
        
        logging.info(f"Found {len(self.image_pairs)} valid image pairs")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        hr_path, lr_path = self.image_pairs[idx]
        
        try:
            hr_tensor = torch.load(hr_path, weights_only=True).float()
            lr_tensor = torch.load(lr_path, weights_only=True).float()
            
            if self.image_size is not None and self.progressive:
                hr_tensor = interpolate(hr_tensor.unsqueeze(0), size=(self.image_size, self.image_size), mode="bicubic", align_corners=False).squeeze(0)
                lr_tensor = interpolate(lr_tensor.unsqueeze(0), size=(self.image_size // 4, self.image_size // 4), mode="bicubic", align_corners=False).squeeze(0)

            if self.norm:
                hr_tensor = (hr_tensor - Config.DATA_NORM_MEAN) / Config.DATA_NORM_STD
                lr_tensor = (lr_tensor - Config.DATA_NORM_MEAN) / Config.DATA_NORM_STD
        
            hr_tensor = apply_transform(hr_tensor, self.transform)
            lr_tensor = apply_transform(lr_tensor, self.transform)
                
            if self.use_segmentation:
                seg_path = hr_path.replace('_HR.pt', '_SEG.pt')
                if os.path.exists(seg_path):
                    seg_tensor = torch.load(seg_path, weights_only=True).float()
                    if self.norm:
                        seg_tensor = (seg_tensor - Config.DATA_NORM_MEAN) / Config.DATA_NORM_STD
                    if self.image_size is not None and self.progressive:
                        seg_tensor = interpolate(seg_tensor.unsqueeze(0), size=(self.image_size, self.image_size), mode="bicubic", align_corners=False).squeeze(0)
                    seg_tensor = apply_transform(seg_tensor, self.transform)
                else:
                    # Generate a placeholder mask (all zeros)
                    seg_tensor = torch.zeros_like(hr_tensor)
                    logging.warning(f"Segmentation mask not found at: {seg_path}. Using a placeholder mask.")
                
                return lr_tensor, hr_tensor, seg_tensor
            else:
                return lr_tensor, hr_tensor
            
        except Exception as e:
            logging.error(f"Error loading images at index {idx}: {e}")
            raise

class RandomElasticDeformation:
    """Applies random elastic deformation to a tensor."""

    def __init__(self, alpha=10, sigma=3, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.rng = np.random.default_rng()

    def __call__(self, tensor):
        if self.rng.random() < self.p:
            image = tensor.squeeze().numpy()
            h, w = image.shape
            dx = self.rng.normal(0, self.alpha, (h, w))
            dy = self.rng.normal(0, self.alpha, (h, w))
            
            # Apply Gaussian filter
            dx = self._gaussian_filter(dx, self.sigma)
            dy = self._gaussian_filter(dy, self.sigma)
            
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            indices = np.stack([x + dx, y + dy], axis=-1).reshape(-1, 2)
            
            # Convert to tensor for grid_sample
            grid = torch.tensor(indices, dtype=torch.float32).reshape(h, w, 2)
            image_tensor = tensor.unsqueeze(0)  # Add batch dimension
            
            # Remap using grid_sample
            deformed_image_tensor = F.grid_sample(
                image_tensor, 
                grid.unsqueeze(0).to(tensor.device),  # Add batch and channel dimensions, move to GPU
                mode='bilinear', 
                padding_mode='reflection', 
                align_corners=True
            ).squeeze(0)  # Remove batch dimension
            return deformed_image_tensor
        
        return tensor
    
    def _gaussian_filter(self, image, sigma):
        """Apply Gaussian filter to an image.
        Args:
            image (np.ndarray): Input image
            sigma (float): Sigma parameter of Gaussian filter

        Returns:
            np.ndarray: Filtered image
        """
        size = int(3 * abs(sigma) + 1)
        x = np.arange(-size, size + 1)
        g = np.exp(-x * x / (2 * sigma * sigma))
        g /= g.sum()
        filtered_image = np.zeros_like(image)
        
        # Convolution along both axis
        for i in range(image.shape[0]):
            filtered_image[i, :] = np.convolve(image[i, :], g, mode="same")
        for j in range(image.shape[1]):
            filtered_image[:, j] = np.convolve(filtered_image[:, j], g, mode="same")

        return filtered_image
    
class RandomApply(torch.nn.Module):
    """Applies a transformation with a given probability."""
    def __init__(self, transform, p=0.5):
        super().__init__()
        self.transform = transform
        self.p = p
        self.rng = np.random.default_rng()

    def forward(self, x):
        if self.rng.random() < self.p:
            return self.transform(x)
        return x
    
class GaussianBlur(torch.nn.Module):
    """Applies gaussian blur to a tensor."""
    def __init__(self, kernel_size, sigma=(0.1, 2)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.rng = np.random.default_rng()
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
            
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        
    def forward(self, x):
        sigma = self.rng.uniform(self.sigma[0], self.sigma[1])
        
        # Generate Gaussian kernel
        k_x = self._get_gaussian_kernel(self.kernel_size[0], sigma)
        k_y = self._get_gaussian_kernel(self.kernel_size[1], sigma)
        kernel = torch.outer(k_x, k_y)
        kernel = kernel / kernel.sum()
        
        # Prepare the kernel for convolution
        kernel = kernel.float().unsqueeze(0).unsqueeze(0).to(x.device)  # [c, 1, h, w]

        # Apply convolution
        return F.conv2d(
            x.unsqueeze(0),
            kernel,
            padding=self.padding,
            groups=1  # since we are applying on each channel separately
        ).squeeze(0)  # Remove batch dim,
    
    def _get_gaussian_kernel(self, kernel_size, sigma):
        """Generate 1D Gaussian kernel."""
        x = torch.arange(-kernel_size // 2, kernel_size // 2 + 1)
        return torch.exp(-x**2 / (2 * sigma**2))
    
class AddGaussianNoise(torch.nn.Module):
    """Applies random gaussian noise to a tensor."""
    def __init__(self, mean=0, std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.rng = np.random.default_rng()

    def forward(self, x):
        std = self.rng.uniform(0, self.std)
        noise = torch.randn_like(x) * std + self.mean
        return x + noise
    
class Cutout(torch.nn.Module):
    """Applies Cutout augmentation to a tensor."""
    def __init__(self, length=32):
        super().__init__()
        self.length = length
        self.rng = np.random.default_rng()

    def forward(self, img):
        h, w = img.shape[-2:]
        mask = torch.ones_like(img)
        y = self.rng.integers(0, h - self.length)
        x = self.rng.integers(0, w - self.length)
        mask[..., y:y+self.length, x:x+self.length] = 0
        return img * mask

class RandomMixUp(torch.nn.Module):
    def __init__(self, p=0.5, alpha=0.4):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.rng = np.random.default_rng()

    def forward(self, x):
        if self.rng.random() < self.p:
            lam = self.rng.beta(self.alpha, self.alpha)
            batch_size = x.shape[0]
            index = torch.randperm(batch_size).to(x.device)
            mixed_x = lam * x + (1 - lam) * x[index]
            return mixed_x
        return x

class RandomCutMix(torch.nn.Module):
    def __init__(self, p=0.5, alpha=0.4):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.rng = np.random.default_rng()

    def forward(self, x):
        if self.rng.random() < self.p:
            lam = self.rng.beta(self.alpha, self.alpha)
            batch_size, _, h, w = x.shape
            rand_h = self.rng.integers(0, h)
            rand_w = self.rng.integers(0, w)
            cut_size_h = int(h * sqrt(1 - lam))
            cut_size_w = int(w * sqrt(1 - lam))

            start_h = max(0, rand_h - cut_size_h // 2)
            end_h = min(h, rand_h + cut_size_h // 2)

            start_w = max(0, rand_w - cut_size_w // 2)
            end_w = min(w, rand_w + cut_size_w // 2)

            mask = torch.ones_like(x)
            mask[:, :, start_h:end_h, start_w:end_w] = 0

            index = torch.randperm(batch_size).to(x.device)
            cutmix_x = mask * x + (1 - mask) * x[index]
            return cutmix_x
        return x

class BayesianLayer(nn.Module):
    """Bayesian Layer for uncertainty estimation."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_mu = nn.Parameter(torch.Tensor(out_features))
        self.b_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.w_mu)
        nn.init.constant_(self.w_rho, -3)
        nn.init.constant_(self.b_mu, 0)
        nn.init.constant_(self.b_rho, -3)

    def forward(self, x, sample=False):
        if self.training or sample:
            w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(x.device)
            b_epsilon = Normal(0, 1).sample(self.b_mu.shape).to(x.device)
            w = self.w_mu + torch.log1p(torch.exp(self.w_rho)) * w_epsilon
            b = self.b_mu + torch.log1p(torch.exp(self.b_rho)) * b_epsilon
        else:
          w = self.w_mu
          b = self.b_mu
        
        return self.dropout(torch.matmul(x, w.t()) + b)
    
class SelfAttention(nn.Module):
    """Self-Attention Layer."""
    def __init__(self, in_channels):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)
        attn_output = torch.bmm(v, attn.permute(0, 2, 1)).view(batch_size, channels, height, width)
        
        return x + self.gamma * attn_output
    
class VisionTransformerBlock(nn.Module):
    """Vision Transformer Block."""
    def __init__(self, num_channels, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
            nn.Dropout(Config.DROPOUT_RATE)
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # (B, H*W, C)
        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        x = x + attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x = x + self.mlp(x.permute(0, 2, 3, 1).view(batch_size, -1, channels)).view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class UNetWithAttention(nn.Module):
    """U-Net with Attention."""
    def __init__(self, in_channels=Config.CHANNELS, num_channels=64):
        super().__init__()
        self.down1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, num_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            spectral_norm(nn.Conv2d(num_channels, num_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2)
        )
        self.attention = SelfAttention(num_channels * 2)
        self.up1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(num_channels * 2, num_channels, kernel_size=4, stride=2, padding=1)),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(num_channels, in_channels, kernel_size=4, stride=2, padding=1)),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x2 = self.attention(x2)
        x = self.up1(x2)
        x = self.up2(x)
        return x

class FeatureExtractor(nn.Module):
    """Extracts intermediate features from the ResNet18 model."""
    def __init__(self):
        super().__init__()
        self.model = resnet.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.eval()

    def forward(self, x):
        with torch.no_grad():
           return self.features(x)

class Generator(nn.Module):
    """Generator with Vision Transformers and U-Net Attention."""
    def __init__(self, in_channels=Config.CHANNELS, num_channels=64):
        super().__init__()
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, num_channels, kernel_size=9, padding=4)),
            nn.PReLU()
        )
        
        self.transformer_blocks = nn.Sequential(*[
            VisionTransformerBlock(num_channels, num_heads=Config.NUM_HEADS) for _ in range(Config.NUM_RESIDUAL_BLOCKS)
        ])
        
        self.conv_block = nn.Sequential(
            spectral_norm(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)),
            nn.BatchNorm2d(num_channels)
        )
        
        self.upsampling = nn.Sequential(
            spectral_norm(nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
            spectral_norm(nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1)),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(num_channels, in_channels, kernel_size=9, padding=4)),
            nn.Tanh()
        )
    
    def forward(self, x, sample = False, layer_activations = False):
        initial = self.initial(x)
        if layer_activations:
            layer_outputs = [initial]
        x = initial
        for i, layer in enumerate(self.transformer_blocks):
             x = layer(x)
             if layer_activations and (i+1) in Config.ACTIVATION_LAYERS:
                  layer_outputs.append(x)
        x = self.conv_block(x) + initial
        x = self.upsampling(x)
        x = self.final(x)
        if layer_activations:
              return x, layer_outputs
        return x

class Discriminator(nn.Module):
    """Discriminator with Spectral Normalization and Self-Attention."""
    def __init__(self, in_channels=Config.CHANNELS, num_channels=64):
        super().__init__()
        def discriminator_block(in_feat, out_feat, normalize=True):
            layers = [spectral_norm(nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, num_channels, normalize=False),
            *discriminator_block(num_channels, num_channels * 2),
            *discriminator_block(num_channels * 2, num_channels * 4),
            *discriminator_block(num_channels * 4, num_channels * 8),
            SelfAttention(num_channels * 8),
            spectral_norm(nn.Conv2d(num_channels * 8, 1, 3, stride=1, padding=1))
        )
    
    def forward(self, x):
                return self.model(x)

class WarmUpScheduler(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warm up scheduler """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps
        return 1

class Trainer:
    def __init__(self, use_segmentation=False, progressive_growing=False):
        self.setup_directories()
        setup_logging()
        setup_distributed()
        
        self.device = Config.DEVICE
        if Config.DISTRIBUTED:
            self.device = torch.device(f"cuda:{Config.LOCAL_RANK}")
        
        self.scaler = GradScaler(enabled=Config.MIXED_PRECISION)
        self.writer = SummaryWriter(os.path.join(Config.LOG_DIR, 'tensorboard')) if is_main_process() else None
        
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        if Config.DISTRIBUTED:
            self.generator = DistributedDataParallel(self.generator, device_ids=[Config.LOCAL_RANK], output_device=Config.LOCAL_RANK)
            self.discriminator = DistributedDataParallel(self.discriminator, device_ids=[Config.LOCAL_RANK], output_device=Config.LOCAL_RANK)
        
        self.g_optimizer = Lookahead(RAdam(self.generator.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY))
        self.d_optimizer = Lookahead(RAdam(self.discriminator.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY))
        
        self.g_scheduler = CosineAnnealingWarmRestarts(self.g_optimizer, T_0=10, T_mult=2)
        self.d_scheduler = CosineAnnealingWarmRestarts(self.d_optimizer, T_0=10, T_mult=2)

        self.g_scheduler_plateau = ReduceLROnPlateau(self.g_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.d_scheduler_plateau = ReduceLROnPlateau(self.d_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        self.g_scheduler_cyclic = CyclicLR(self.g_optimizer, base_lr = Config.CYCLE_LR_BASE_LR, max_lr = Config.CYCLE_LR_MAX_LR, step_size_up=Config.CYCLE_LR_STEP_SIZE_UP, mode = Config.CYCLE_LR_MODE)
        self.d_scheduler_cyclic = CyclicLR(self.d_optimizer, base_lr = Config.CYCLE_LR_BASE_LR, max_lr = Config.CYCLE_LR_MAX_LR, step_size_up=Config.CYCLE_LR_STEP_SIZE_UP, mode = Config.CYCLE_LR_MODE)

        self.g_warmup_scheduler = WarmUpScheduler(self.g_optimizer, Config.WARMUP_STEPS)
        self.d_warmup_scheduler = WarmUpScheduler(self.d_optimizer, Config.WARMUP_STEPS)

        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_content = nn.L1Loss()
        self.lpips = LPIPS(net='alex', verbose=False).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        
        self.best_g_loss = float('inf')
        self.epochs_without_improvement = 0
        self.dice = Dice().to(self.device)
        self.jaccard = JaccardIndex(num_classes=1, task="binary").to(self.device)

        self.use_segmentation = use_segmentation
        self.progressive_growing = progressive_growing

        self.integrated_gradients = IntegratedGradients(self.generator)
        self.current_image_size = Config.LOW_RES_SIZE * 4
    
    def setup_directories(self):
        if is_main_process():
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            os.makedirs(Config.GENERATED_DIR, exist_ok=True)
            os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    def train_step(self, lr_images, hr_images, seg_images = None, epoch = None, total_steps = None):
        batch_size = lr_images.size(0)
        real_label = torch.ones(batch_size, 1, 16, 16).to(self.device)
        fake_label = torch.zeros(batch_size, 1, 16, 16).to(self.device)
        
        # Train Discriminator
        self.discriminator.zero_grad()
        with autocast(device_type=self.device.type, enabled=Config.MIXED_PRECISION):
            fake_images = self.generator(lr_images)
            d_real = self.discriminator(hr_images)
            d_fake = self.discriminator(fake_images.detach())
            
            d_real_loss = self.criterion_gan(d_real, real_label)
            d_fake_loss = self.criterion_gan(d_fake, fake_label)
            d_loss = (d_real_loss + d_fake_loss) / 2
        
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.d_optimizer)
        
        # Train Generator
        self.generator.zero_grad()
        with autocast(device_type=self.device.type, enabled=Config.MIXED_PRECISION):
            g_fake = self.discriminator(fake_images)
            g_gan_loss = self.criterion_gan(g_fake, real_label)
            g_content_loss = self.criterion_content(fake_images, hr_images)
            g_perceptual_loss = self.calculate_perceptual_loss(fake_images, hr_images)
            g_ssim_loss = 1 - self.ssim(fake_images, hr_images)
            
            hr_features = self.feature_extractor(hr_images.repeat(1, 3, 1, 1))
            sr_features = self.feature_extractor(fake_images.repeat(1, 3, 1, 1))
            g_feature_loss = self.criterion_content(sr_features, hr_features)

            if self.use_segmentation and seg_images is not None:
                seg_loss = self.criterion_content(fake_images, seg_images)
                g_loss = (
                    Config.LAMBDA_CONTENT * g_content_loss +
                    Config.LAMBDA_ADV * g_gan_loss +
                    g_perceptual_loss +
                    Config.LAMBDA_SSIM * g_ssim_loss +
                    Config.LAMBDA_FEATURE * g_feature_loss +
                    Config.SEG_WEIGHT * seg_loss
                )
            else:
                 g_loss = (
                    Config.LAMBDA_CONTENT * g_content_loss +
                    Config.LAMBDA_ADV * g_gan_loss +
                    g_perceptual_loss +
                    Config.LAMBDA_SSIM * g_ssim_loss +
                    Config.LAMBDA_FEATURE * g_feature_loss
                )
        
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.g_optimizer)
        self.scaler.update()
        
        torch.cuda.empty_cache()
        
        metrics = {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'g_gan_loss': g_gan_loss.item(),
            'g_content_loss': g_content_loss.item(),
            'g_perceptual_loss': g_perceptual_loss.item(),
            'g_ssim_loss': g_ssim_loss.item(),
            'g_feature_loss' : g_feature_loss.item(),
            'g_psnr': self.psnr(fake_images, hr_images).item(),
            'g_dice': self.dice(fake_images > 0.5, hr_images > 0.5).item(),
            'g_jaccard': self.jaccard(fake_images > 0.5, hr_images > 0.5).item()
        }

        if self.use_segmentation and seg_images is not None:
            metrics['seg_loss'] = seg_loss.item()
        
        if is_main_process():
          for key, value in metrics.items():
              self.writer.add_scalar(key, value, total_steps)

        return metrics
    
    def calculate_perceptual_loss(self, fake_images, hr_images):
          perceptual_loss = 0
          hr_features = self.feature_extractor(hr_images.repeat(1, 3, 1, 1))
          sr_features = self.feature_extractor(fake_images.repeat(1, 3, 1, 1))

          for i, (hr_feat, sr_feat, weight) in enumerate(zip(hr_features, sr_features, Config.PERC_LAYER_WEIGHTS)):
              perceptual_loss += weight * self.criterion_content(sr_feat, hr_feat)
          return Config.LAMBDA_PERC * perceptual_loss

    def save_checkpoint(self, epoch, losses, is_final=False): # Added final param to save last weights
        if is_main_process():
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': self.generator.module.state_dict() if Config.DISTRIBUTED else self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.module.state_dict() if Config.DISTRIBUTED else self.discriminator.state_dict(),
                'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                'losses': losses
            }
            path = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth') if not is_final else os.path.join(Config.CHECKPOINT_DIR, 'final_checkpoint.pth')
            torch.save(checkpoint, path)
            logging.info(f"Saved checkpoint: {path}")
    
    def save_sample_images(self, epoch, lr_images, hr_images, sr_images, is_final = False): # Added final param to save last images
            if is_main_process():
                images = torch.cat([lr_images.cpu(), sr_images.cpu(), hr_images.cpu(), sr_images.cpu()], dim=-1)
                save_path = os.path.join(Config.GENERATED_DIR, f'epoch_{epoch}.png') if not is_final else os.path.join(Config.GENERATED_DIR, 'final_generated_images.png')
                save_image(images, save_path, normalize=True)

    def train(self, train_loader, num_epochs, val_loader = None):
            try:
                total_steps = 0
                
                for epoch in range(1, num_epochs + 1):
                    self.generator.train()
                    self.discriminator.train()
                    
                    epoch_losses = []
                    if Config.DISTRIBUTED:
                       train_loader.sampler.set_epoch(epoch)
                    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', disable=not is_main_process())
                    
                    for batch_idx, batch in enumerate(progress_bar):
                        if self.use_segmentation:
                           lr_images, hr_images, seg_images = batch
                           seg_images = seg_images.to(self.device)
                        else:
                           lr_images, hr_images = batch
                           seg_images = None

                        lr_images = lr_images.to(self.device)
                        hr_images = hr_images.to(self.device)
                        
                        losses = self.train_step(lr_images, hr_images, seg_images, epoch, total_steps)
                        epoch_losses.append(losses)
                        
                        progress_bar.set_postfix(
                            d_loss=f"{losses['d_loss']:.4f}",
                            g_loss=f"{losses['g_loss']:.4f}"
                        )
                        
                        total_steps += 1
                        
                        # Learning rate warmup
                        self.g_warmup_scheduler.step()
                        self.d_warmup_scheduler.step()

                    
                    avg_losses = {
                        key: sum(loss[key] for loss in epoch_losses) / len(epoch_losses)
                        for key in epoch_losses[0].keys()
                    }
                    
                    if is_main_process():
                        logging.info(f"Epoch {epoch} Average Losses:")
                        for key, value in avg_losses.items():
                            logging.info(f"{key}: {value:.4f}")

                    if val_loader is not None:
                        val_metrics = self.validate(val_loader, epoch)
                        if is_main_process():
                            logging.info(f"Epoch {epoch} Validation Metrics:")
                            for key, value in val_metrics.items():
                                logging.info(f"{key}: {value:.4f}")
                    
                    if epoch % 5 == 0 or epoch == num_epochs: # save checkpoint at the end of training
                        self.save_checkpoint(epoch, avg_losses)
                        
                        if is_main_process():
                            self.generator.eval()
                            with torch.no_grad():
                                if self.use_segmentation:
                                    sample_lr, sample_hr, _ = next(iter(train_loader))
                                else:
                                   sample_lr, sample_hr = next(iter(train_loader))
                                sample_lr = sample_lr[:4].to(self.device)
                                sample_hr = sample_hr[:4].to(self.device)
                                sample_sr = self.generator(sample_lr)
                                self.save_sample_images(epoch, sample_lr, sample_hr, sample_sr)
                            self.generator.train()
                    
                    self.g_scheduler.step()
                    self.d_scheduler.step()

                    self.g_scheduler_cyclic.step()
                    self.d_scheduler_cyclic.step()

                    # Reduce Learning Rate on plateau
                    if val_loader is not None:
                      self.g_scheduler_plateau.step(avg_losses['g_loss'])
                      self.d_scheduler_plateau.step(avg_losses['d_loss'])
                    
                    # Early stopping logic
                    current_g_loss = avg_losses['g_loss']
                    if current_g_loss < self.best_g_loss - Config.EARLY_STOPPING_MIN_DELTA:
                            self.best_g_loss = current_g_loss
                            self.epochs_without_improvement = 0
                    else:
                            self.epochs_without_improvement += 1
                    
                    if self.epochs_without_improvement > Config.EARLY_STOPPING_PATIENCE:
                        logging.info(f"Early stopping triggered at epoch {epoch}")
                        break
                
                # Save final model and visualizations
                if is_main_process():
                      self.save_checkpoint(epoch, avg_losses, is_final=True)
                      self.generator.eval()
                      with torch.no_grad():
                          if self.use_segmentation:
                              sample_lr, sample_hr, _ = next(iter(train_loader))
                          else:
                              sample_lr, sample_hr = next(iter(train_loader))
                          sample_lr = sample_lr[:4].to(self.device)
                          sample_hr = sample_hr[:4].to(self.device)
                          sample_sr = self.generator(sample_lr)
                          self.save_sample_images(epoch, sample_lr, sample_hr, sample_sr, is_final = True)
                      self.generator.train()


            except Exception as e:
                logging.error(f"Training error: {str(e)}")
                raise
            finally:
                if is_main_process() and self.writer:
                    self.writer.close()
    def validate(self, val_loader, epoch):
        self.generator.eval()
        all_metrics = []
        with torch.no_grad():
           for batch in tqdm(val_loader, desc = f"Validating Epoch {epoch}", disable = not is_main_process()):
                 if self.use_segmentation:
                     lr_images, hr_images, seg_images = batch
                     seg_images = seg_images.to(self.device)
                 else:
                     lr_images, hr_images = batch
                     seg_images = None
                 
                 lr_images = lr_images.to(self.device)
                 hr_images = hr_images.to(self.device)
                 
                 metrics = self.calculate_metrics(lr_images, hr_images, seg_images)
                 all_metrics.append(metrics)
        avg_metrics = {
            key: sum(metrics[key] for metrics in all_metrics) / len(all_metrics) for key in all_metrics[0].keys()
        }
        self.generator.train()
        return avg_metrics
    
    def calculate_metrics(self, lr_images, hr_images, seg_images = None):
         with torch.no_grad():
            sr_images = self.generator(lr_images)
            psnr = self.psnr(sr_images, hr_images)
            ssim = self.ssim(sr_images, hr_images)
            dice = self.dice(sr_images > 0.5, hr_images > 0.5)
            jaccard = self.jaccard(sr_images > 0.5, hr_images > 0.5)
            content_loss = self.criterion_content(sr_images, hr_images)

            metrics = {
                'val_psnr': psnr.item(),
                'val_ssim': ssim.item(),
                'val_dice': dice.item(),
                'val_jaccard': jaccard.item(),
                'val_content_loss' : content_loss.item()
            }

            if self.use_segmentation and seg_images is not None:
               seg_loss = self.criterion_content(sr_images, seg_images)
               metrics['val_seg_loss'] = seg_loss.item()

            return metrics

    def predict(self, lr_image, num_samples=10, post_process = True):
        self.generator.eval()
        with torch.no_grad():
            if lr_image.ndim == 3: # If its a single image (c,h,w)
              lr_image = lr_image.unsqueeze(0)
            
            lr_image = lr_image.to(self.device)
            
            if num_samples > 1: # Enable monte carlo dropout
              sr_images = torch.stack([self.generator(lr_image, sample = True) for _ in range(num_samples)])
              sr_image = torch.mean(sr_images, dim = 0)
              uncertainty = torch.var(sr_images, dim = 0) # Variance as uncertainty
              if post_process:
                sr_image = self.post_process(sr_image)

              return sr_image, uncertainty
            else:
              sr_image = self.generator(lr_image)
              if post_process:
                  sr_image = self.post_process(sr_image)
              return sr_image, None # No uncertainty returned

    def post_process(self, image, patch_size = 7, patch_distance = 5, fast_mode = True, h = 0.02):
        """
        Applies non-local means filtering to the given image.
        """
        if isinstance(image, torch.Tensor):
             image = image.squeeze(0).cpu().numpy() # remove batch dim and move to cpu
        
        image = img_as_float(image)
        filtered_image = denoise_nl_means(image, patch_size = patch_size, patch_distance = patch_distance, fast_mode = fast_mode, h=h, channel_axis=0)

        return torch.tensor(filtered_image, dtype = torch.float32).unsqueeze(0).to(self.device)
    
    def calculate_activations(self, lr_images):
          self.generator.eval()
          with torch.no_grad():
             _, layer_outputs = self.generator(lr_images, layer_activations = True)
             return layer_outputs
    
    def visualize_activations(self, lr_images, layer_outputs):
         if is_main_process():
           num_layers = len(layer_outputs)
           num_images = lr_images.shape[0]
           fig, axes = plt.subplots(num_layers, num_images, figsize=(15, 4 * num_layers))

           if num_layers == 1:
              axes = [axes]

           for i, activations in enumerate(layer_outputs):
             for j in range(num_images):
                 ax = axes[i][j] if num_layers > 1 else axes[j]
                 activation_map = activations[j].mean(dim=0).cpu().numpy()
                 ax.imshow(activation_map, cmap='viridis')
                 ax.axis('off')
                 ax.set_title(f"Image {j+1} Layer {i+1}")

           plt.tight_layout()
           plt.savefig(os.path.join(Config.GENERATED_DIR, 'activation_maps.png'))

    def calculate_integrated_gradients(self, lr_images, hr_images):
       self.generator.eval()
       lr_images = lr_images.to(self.device)
       hr_images = hr_images.to(self.device)

       integrated_grads = self.integrated_gradients.attribute(lr_images, target = hr_images)

       return integrated_grads
    
    def visualize_integrated_gradients(self, lr_images, integrated_grads):
       if is_main_process():
         num_images = lr_images.shape[0]
         fig, axes = plt.subplots(1, num_images, figsize=(15, 4))

         if num_images == 1:
             axes = [axes]
         
         for j in range(num_images):
             ax = axes[j] if num_images > 1 else axes
             grads = integrated_grads[j].mean(dim=0).cpu().numpy()
             ax.imshow(grads, cmap = "viridis")
             ax.axis('off')
             ax.set_title(f"Image {j+1} Integrated Gradients")

         plt.tight_layout()
         plt.savefig(os.path.join(Config.GENERATED_DIR, 'integrated_gradients.png'))



def create_data_loaders(dataset, fold_idx, num_folds, progressive=False, image_size=None): # Removed duplicated code
    # Calculate indices for validation set
    fold_size = len(dataset) // num_folds
    start_idx = fold_idx * fold_size
    end_idx = (fold_idx + 1) * fold_size
    val_indices = list(range(start_idx, end_idx))

    # Split the remaining dataset into training set indices
    train_indices = [i for i in range(len(dataset)) if i not in val_indices]

    # Create samplers (common logic)
    train_sampler = RandomSampler(train_indices) if not Config.DISTRIBUTED else dist.DistributedSampler(train_indices, shuffle=True)
    val_sampler = SequentialSampler(val_indices) if not Config.DISTRIBUTED else dist.DistributedSampler(val_indices, shuffle=False)
    
    # Create dataloaders (common logic)
    train_loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )
    
    val_loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=val_sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )

    return train_loader, val_loader

def main():
    try:
        setup_logging()
        logging.info("Starting SRGAN training pipeline")
        
        dataset_path = download_brats_dataset()
        if is_main_process():
             Config.set_dataset_folder(dataset_path)
             logging.info(f"Dataset path set to: {dataset_path}")

        use_segmentation = True # Set to True to use the segmentation mask during training and prediction
        progressive_growing = True # set to true to enable progressive growing
        
        if not os.path.exists(Config.TRAIN_HR_FOLDER) or not os.path.exists(Config.TRAIN_LR_FOLDER):
            if is_main_process():
                 logging.info("Dataset not preprocessed. Starting preprocessing...")
                 num_images = preprocess_dataset(use_segmentation = use_segmentation)
                 logging.info(f"Preprocessing complete. Created {num_images} image pairs")
            else:
                  logging.info("Waiting for main process to process the dataset...")
                  torch.distributed.barrier()
        else:
            logging.info("Using previously preprocessed dataset")
        
        hr_files = os.listdir(Config.TRAIN_HR_FOLDER)
        lr_files = os.listdir(Config.TRAIN_LR_FOLDER)
        if len(hr_files) != len(lr_files):
            logging.warning(f"Number of HR files ({len(hr_files)}) does not match the number of LR files ({len(lr_files)}), removing files for equal number")
            
            hr_files = sorted(hr_files)
            lr_files = sorted(lr_files)
            
            if len(hr_files) > len(lr_files):
                 for i in range(len(hr_files) - len(lr_files)):
                     file_to_remove = os.path.join(Config.TRAIN_HR_FOLDER, hr_files[len(hr_files)-1-i])
                     os.remove(file_to_remove)
                     logging.warning(f"Removed extra HR file: {file_to_remove}")
            elif len(lr_files) > len(hr_files):
                 for i in range(len(lr_files) - len(hr_files)):
                     file_to_remove = os.path.join(Config.TRAIN_LR_FOLDER, lr_files[len(lr_files)-1-i])
                     os.remove(file_to_remove)
                     logging.warning(f"Removed extra LR file: {file_to_remove}")

            hr_files = os.listdir(Config.TRAIN_HR_FOLDER)
            lr_files = os.listdir(Config.TRAIN_LR_FOLDER)

            if len(hr_files) != len(lr_files):
                 raise ValueError(f"Number of HR files ({len(hr_files)}) still does not match the number of LR files ({len(lr_files)}) after correction")
            else:
                 logging.info(f"Corrected to {len(hr_files)} HR files and {len(lr_files)} LR files")
        else:
             logging.info(f"Verified {len(hr_files)} HR files and {len(lr_files)} LR files")
        
        
        # Cross Validation Training
        for fold_idx in range(Config.NUM_FOLDS):
            if is_main_process():
                logging.info(f"Starting training for Fold {fold_idx + 1}/{Config.NUM_FOLDS}")
           
            trainer = Trainer(use_segmentation = use_segmentation, progressive_growing=progressive_growing)
            
            if progressive_growing:
                current_image_size = Config.LOW_RES_SIZE * 4
                while current_image_size <= Config.IMAGE_SIZE:
                    if is_main_process():
                        logging.info(f"Removing existing files to save re-processed files at image size: {current_image_size}")
                        for file in os.listdir(Config.TRAIN_HR_FOLDER):
                            if file.endswith(".pt"):
                                os.remove(os.path.join(Config.TRAIN_HR_FOLDER, file))
                        for file in os.listdir(Config.TRAIN_LR_FOLDER):
                            if file.endswith(".pt"):
                                 os.remove(os.path.join(Config.TRAIN_LR_FOLDER, file))
                    else:
                        torch.distributed.barrier()

                    # 2. Re-preprocess dataset with the new image size
                    if is_main_process():
                        num_images = preprocess_dataset(use_segmentation = use_segmentation, image_size = current_image_size) # New image size
                        logging.info(f"Re-processing complete. Created {num_images} image pairs at image size: {current_image_size}")
                    else:
                        torch.distributed.barrier()
                   
                    # 3. Create Dataset and DataLoaders
                    dataset = BraTSDataset(
                        Config.TRAIN_HR_FOLDER,
                        Config.TRAIN_LR_FOLDER,
                        transform=True,
                        use_segmentation = use_segmentation,
                        norm = True,
                        subset = 1,
                        progressive=False,  # Indicate that preprocessing is done.
                        image_size=current_image_size
                    )
                    
                    train_loader, val_loader = create_data_loaders(dataset, fold_idx, Config.NUM_FOLDS, progressive = True, image_size = current_image_size)
                    trainer.current_image_size = current_image_size # updating the current image size to track it.
                    
                    trainer.train(train_loader, Config.PROGRESSIVE_GROW_STEPS, val_loader = val_loader)
                    
                    current_image_size *= 2

            else:
              dataset = BraTSDataset(
                    Config.TRAIN_HR_FOLDER,
                    Config.TRAIN_LR_FOLDER,
                    transform=True,
                    use_segmentation = use_segmentation,
                    norm = True,
                    subset = 1
                )
              train_loader, val_loader = create_data_loaders(dataset, fold_idx, Config.NUM_FOLDS, progressive = False)
              trainer.train(train_loader, Config.EPOCHS, val_loader = val_loader)
        
        if is_main_process():
            logging.info("Training completed successfully")
        
        # Predict on a sample image using Monte Carlo Dropout
        if is_main_process():
            sample_dataset = BraTSDataset(
            Config.TRAIN_HR_FOLDER,
            Config.TRAIN_LR_FOLDER,
            transform = False,
            use_segmentation = use_segmentation,
            norm = False
        )
            
            sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle = True)
            sample_batch = next(iter(sample_loader))

            if use_segmentation:
                sample_lr, sample_hr, *_ = sample_batch
            else:
                sample_lr, sample_hr = sample_batch


            sr_image, uncertainty = trainer.predict(sample_lr, num_samples = 10) # Get the SR image and uncertainty with mc dropout
            save_image(sr_image.cpu(), os.path.join(Config.GENERATED_DIR, 'sample_sr_image.png'), normalize=True)

            if uncertainty is not None:
                save_image(uncertainty.cpu(), os.path.join(Config.GENERATED_DIR, 'sample_uncertainty.png'), normalize=True)
            logging.info(f"Generated Sample SR Image and saved to {Config.GENERATED_DIR}")
            
            layer_outputs = trainer.calculate_activations(sample_lr.to(trainer.device))
            trainer.visualize_activations(sample_lr, layer_outputs)
            logging.info(f"Generated Sample activation maps and saved to {Config.GENERATED_DIR}")

            integrated_grads = trainer.calculate_integrated_gradients(sample_lr, sample_hr.to(trainer.device))
            trainer.visualize_integrated_gradients(sample_lr, integrated_grads)
            logging.info(f"Generated  ample Integrated Gradients and saved to {Config.GENERATED_DIR}")
        
    except Exception as e:
        logging.error(f"Pipeline error: {str(e)}")
        raise
    finally:
        cleanup_distributed()

if __name__ == '__main__':
    main()