import warnings

# Suppress specific FutureWarning from transformers
warnings.filterwarnings(
    "ignore",
    message=r"torch\.utils\._pytree\._register_pytree_node is deprecated\. Please use torch\.utils\._pytree\.register_pytree_node instead\.",
    category=FutureWarning,
    module="transformers"
)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import zipfile
from tqdm import tqdm
import requests
from multiprocessing import freeze_support
import torch.nn.functional as F
import datetime
import logging
from torch.utils.checkpoint import checkpoint  # Import checkpoint for gradient checkpointing
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# Enable CuDNN Benchmark for optimized performance
torch.backends.cudnn.benchmark = True

# Hyperparameters
BATCH_SIZE = 16  # Increased for RTX 4090
IMAGE_SIZE = 256
CHANNELS = 3
B = 32  # Number of residual blocks
LEARNING_RATE = 1e-4
EPOCHS = 100  # Increased for comprehensive training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gradient Accumulation Steps
ACCUMULATION_STEPS = 2  # Reduced due to higher GPU memory

# Dataset Configuration
DATASET_URL = "https://github.com/ヒfigan/SRGAN-PyTorch/releases/download/datasets/celebahq-resized-256x256.zip"
DATASET_FOLDER = "celeba_hq_256"
CELEBA_FOLDER = os.path.join(DATASET_FOLDER, "celeba_hq_256", "celeba_hq_256")

def download_dataset():
    try:
        if not os.path.exists(DATASET_FOLDER):
            os.makedirs(DATASET_FOLDER)
            print("Downloading dataset...")
            response = requests.get(DATASET_URL, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

            zip_path = os.path.join(DATASET_FOLDER, "celebahq-resized-256x256.zip")
            with open(zip_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                raise Exception("ERROR, something went wrong with the download")

            print("Unzipping dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATASET_FOLDER)
        else:
            print("Dataset already exists. Skipping download.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

class SRGANDataset(Dataset):
    def __init__(self, image_dir, low_res_size=IMAGE_SIZE // 4, high_res_size=IMAGE_SIZE, augment=False):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        transforms_list = [
            transforms.Resize(low_res_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
        if augment:
            transforms_list.extend([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        self.low_res_transform = transforms.Compose(transforms_list)
        
        self.high_res_transform = transforms.Compose([
            transforms.Resize(high_res_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*CHANNELS, std=[0.5]*CHANNELS),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        high_res = self.high_res_transform(img)
        low_res = self.low_res_transform(img)
        return low_res, high_res

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.PReLU(),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
        )

    def forward(self, x):
        return self.block(x) + x  # Removed gradient checkpointing for better performance

class UpsampleBlock(nn.Module):
    def __init__(self, features, scale_factor):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features * scale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, channels=CHANNELS, features=64, num_residuals=B):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(features) for _ in range(num_residuals)])
        self.conv_block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
        )
        self.upsamples = nn.Sequential(
            UpsampleBlock(features, scale_factor=2),
            UpsampleBlock(features, scale_factor=2),
        )
        self.final_conv = nn.Conv2d(features, channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial_conv(x)
        x = self.residuals(initial)
        x = self.conv_block(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final_conv(x))

class Discriminator(nn.Module):
    def __init__(self, channels=CHANNELS, features=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = Block(features, features, stride=2)
        self.conv3 = Block(features, features * 2, stride=1)
        self.conv4 = Block(features * 2, features * 2, stride=2)
        self.conv5 = Block(features * 2, features * 4, stride=1)
        self.conv6 = Block(features * 4, features * 4, stride=2)
        self.conv7 = Block(features * 4, features * 8, stride=1)
        self.conv8 = Block(features * 8, features * 8, stride=2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(features * 8 * 7 * 7, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            # Removed nn.Sigmoid() for compatibility with BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return self.classifier(x)

# Initialize SSIM metric outside the loss function
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(DEVICE)
except ImportError:
    ssim_metric = None
    print("torchmetrics not installed. SSIM loss not calculated.")

def get_vgg_features():
    try:
        vgg19 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', weights='IMAGENET1K_V1').features[:10].eval().to(DEVICE)
        for param in vgg19.parameters():
            param.requires_grad = False
        return vgg19
    except Exception as e:
        print(f"Error loading VGG19: {e}")
        return None

def vgg_loss(vgg_model, y_hr, y_sr):
    if vgg_model is None:
        return torch.tensor(0.0).to(DEVICE)
    return nn.MSELoss()(vgg_model(y_sr), vgg_model(y_hr))

def ssim_loss(y_true, y_pred):
    # Simple SSIM implementation using torchmetrics
    if ssim_metric is None:
        return torch.tensor(0.0).to(DEVICE)
    loss = 1 - ssim_metric(y_pred, y_true)
    return loss

def pixel_mse_loss(y_true, y_pred):
    return nn.MSELoss()(y_true, y_pred)

def content_loss(y_true, y_pred, vgg_model):
    return pixel_mse_loss(y_true, y_pred) + 0.006 * vgg_loss(vgg_model, y_true, y_pred) + 0.01 * ssim_loss(y_true, y_pred)

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss decreased to {self.best_loss:.4f}. Resetting counter.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def setup_logging():
    logging.basicConfig(
        filename='training.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def train_srgan():
    try:
        # Setup logging
        setup_logging()
        logging.info("Starting SRGAN training.")
        
        # Initialize TensorBoard
        writer = SummaryWriter('runs/SRGAN_experiment')
        
        # Print and log device information
        print(f"Training on {DEVICE}")
        logging.info(f"Training on {DEVICE}")

        # Download and prepare dataset
        download_dataset()

        # Create dataset and dataloader with augmentation
        dataset = SRGANDataset(CELEBA_FOLDER, augment=True)
        train_loader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=8,  # Increased for RTX 4090
            pin_memory=True,
            prefetch_factor=4  # Increased for RTX 4090
        )

        # Initialize models
        generator = Generator().to(DEVICE)
        discriminator = Discriminator().to(DEVICE)
        
        # Optimizers
        g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        
        # Learning Rate Schedulers
        g_scheduler = StepLR(g_optimizer, step_size=30, gamma=0.1)
        d_scheduler = StepLR(d_optimizer, step_size=30, gamma=0.1)
        
        # Loss functions
        bce_loss = nn.BCEWithLogitsLoss()  # Changed from BCELoss to BCEWithLogitsLoss
        vgg_model = get_vgg_features()

        # Initialize mixed precision scaler
        scaler = torch.cuda.amp.GradScaler()

        # Early Stopping
        early_stopping = EarlyStopping(patience=10, verbose=True)

        # Training loop
        generator.train()
        discriminator.train()

        for epoch in range(1, EPOCHS + 1):
            logging.info(f"Epoch {epoch}/{EPOCHS}")
            print(f"Epoch {epoch}/{EPOCHS}")
            loop = tqdm(train_loader, leave=True)
            for batch_idx, (low_res, high_res) in enumerate(loop):
                low_res = low_res.to(DEVICE, non_blocking=True)
                high_res = high_res.to(DEVICE, non_blocking=True)

                ### Discriminator Training ###
                with torch.cuda.amp.autocast():
                    fake_images = generator(low_res)
                    real_labels = torch.ones((high_res.size(0), 1)).to(DEVICE) * 0.9  # Label smoothing
                    fake_labels = torch.zeros((fake_images.size(0), 1)).to(DEVICE)

                    discriminator_real_output = discriminator(high_res)
                    discriminator_fake_output = discriminator(fake_images.detach())

                    d_loss_real = bce_loss(discriminator_real_output, real_labels)
                    d_loss_fake = bce_loss(discriminator_fake_output, fake_labels)
                    d_loss = (d_loss_real + d_loss_fake) / 2

                discriminator.zero_grad()
                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
                scaler.update()

                ### Generator Training ###
                with torch.cuda.amp.autocast():
                    discriminator_fake_output = discriminator(fake_images)
                    g_loss_gan = bce_loss(discriminator_fake_output, real_labels)
                    g_loss_content = content_loss(high_res, fake_images, vgg_model)
                    g_loss = g_loss_content + 1e-3 * g_loss_gan

                generator.zero_grad()
                scaler.scale(g_loss).backward()
                scaler.step(g_optimizer)
                scaler.update()

                # Gradient Accumulation (if needed)
                # Currently set to 1, can be increased if required
                # if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                #     optimizer_step += 1

                # Logging
                loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())
                writer.add_scalar('Discriminator Loss', d_loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Generator Loss', g_loss.item(), epoch * len(train_loader) + batch_idx)

            # Step the schedulers
            g_scheduler.step()
            d_scheduler.step()

            # Save generated images
            os.makedirs("generated", exist_ok=True)
            with torch.no_grad():
                generator.eval()
                sample_low_res, _ = next(iter(train_loader))
                sample_low_res = sample_low_res.to(DEVICE)
                fake_images = generator(sample_low_res)
                fake_image_grid = torchvision.utils.make_grid(fake_images[:4], normalize=True)
                torchvision.utils.save_image(fake_image_grid, f"generated/fake_images_epoch_{epoch}.png")
                writer.add_image('Generated Images', fake_image_grid, epoch)
                generator.train()

            # Save Checkpoint
            checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
            }, filename=checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

            # Placeholder for validation loss calculation
            # You can implement a separate validation loop and calculate validation_loss
            # For demonstration, we'll skip it and not use early stopping
            # validation_loss = calculate_validation_loss()
            # early_stopping(validation_loss)
            # if early_stopping.early_stop:
            #     logging.info("Early stopping triggered.")
            #     print("Early stopping triggered.")
            #     break

            # Free up memory
            torch.cuda.empty_cache()

        # Save final models
        torch.save(generator.state_dict(), 'srgan_generator_pytorch.pth')
        torch.save(discriminator.state_dict(), 'srgan_discriminator_pytorch.pth')

        logging.info("Training finished and models saved.")
        print("Training finished and models saved!")
        writer.close()

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    freeze_support()
    train_srgan()