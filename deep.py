import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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
from torch.optim.lr_scheduler import StepLR

class Config:
    BATCH_SIZE = 32  # Increased batch size for faster training
    IMAGE_SIZE = 512
    LOW_RES_SIZE = IMAGE_SIZE // 4
    CHANNELS = 1
    EPOCHS = 200
    LEARNING_RATE = 2e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIXED_PRECISION = True
    GRADIENT_ACCUMULATION_STEPS = 4  # Gradient accumulation for larger effective batch size

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_FOLDER = os.path.join(BASE_DIR, "brats_dataset")
    TRAIN_HR_FOLDER = os.path.join(DATASET_FOLDER, "train_HR")
    TRAIN_LR_FOLDER = os.path.join(DATASET_FOLDER, "train_LR")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    GENERATED_DIR = os.path.join(BASE_DIR, "generated_medical")

    @staticmethod
    def set_dataset_folder(path):
        Config.DATASET_FOLDER = path
        Config.TRAIN_HR_FOLDER = os.path.join(path, "train_HR")
        Config.TRAIN_LR_FOLDER = os.path.join(path, "train_LR")

def setup_logging():
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(Config.LOG_DIR, 'training.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

def download_brats_dataset():
    logging.info("Downloading BraTS dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("shakilrana/brats-2023-adult-glioma")
        logging.info(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        raise

def resize_with_padding(image_array, target_size):
    """Resize numpy array to target size while maintaining aspect ratio and padding."""
    image = Image.fromarray((image_array * 255).astype(np.uint8))
    target_ratio = target_size[0] / target_size[1]
    width, height = image.size
    img_ratio = width / height
    
    if img_ratio > target_ratio:
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    
    image = image.resize((new_width, new_height), Image.BICUBIC)
    padded_image = Image.new('L', target_size, 0)
    left = (target_size[0] - new_width) // 2
    top = (target_size[1] - new_height) // 2
    padded_image.paste(image, (left, top))
    
    return np.array(padded_image) / 255.0

class BraTSDataset(Dataset):
    def __init__(self, hr_folder, lr_folder, transform=True):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        
        hr_files = sorted([f for f in os.listdir(hr_folder) if f.endswith('_HR.pt')])
        lr_files = sorted([f for f in os.listdir(lr_folder) if f.endswith('_LR.pt')])
        
        self.image_pairs = [(
            os.path.join(hr_folder, hr_file),
            os.path.join(lr_folder, hr_file.replace('_HR.pt', '_LR.pt'))
        ) for hr_file in hr_files if hr_file.replace('_HR.pt', '_LR.pt') in lr_files]
        
        if transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
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
            
            if self.transform:
                seed = torch.randint(0, 2**32, (1,)).item()
                torch.manual_seed(seed)
                hr_tensor = self.transform(hr_tensor)
                torch.manual_seed(seed)
                lr_tensor = self.transform(lr_tensor)
            
            return lr_tensor, hr_tensor
            
        except Exception as e:
            logging.error(f"Error loading images at index {idx}: {e}")
            raise

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=Config.CHANNELS, num_channels=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        self.residuals = nn.Sequential(*[
            ResidualBlock(num_channels) for _ in range(8)  # Reduced from 16 to 8 for faster training
        ])
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        
        self.upsampling = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, kernel_size=9, padding=4),
            nn.Tanh()
        )
    
    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.conv_block(x) + initial
        x = self.upsampling(x)
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=Config.CHANNELS):
        super().__init__()
        def discriminator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 3, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

class Trainer:
    def __init__(self):
        self.setup_directories()
        setup_logging()
        
        self.device = Config.DEVICE
        self.scaler = GradScaler(enabled=Config.MIXED_PRECISION)
        self.writer = SummaryWriter(os.path.join(Config.LOG_DIR, 'tensorboard'))
        
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=Config.LEARNING_RATE)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=Config.LEARNING_RATE)
        
        self.g_scheduler = StepLR(self.g_optimizer, step_size=50, gamma=0.5)  # Learning rate scheduler
        self.d_scheduler = StepLR(self.d_optimizer, step_size=50, gamma=0.5)
        
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_content = nn.L1Loss()
        self.lpips = LPIPS(net='alex', verbose=False).to(self.device)
    
    def setup_directories(self):
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.GENERATED_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    def train_step(self, lr_images, hr_images):
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
            g_perceptual_loss = self.lpips(fake_images, hr_images).mean()
            g_loss = g_content_loss + 0.001 * g_gan_loss + 0.1 * g_perceptual_loss
        
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.g_optimizer)
        self.scaler.update()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'g_gan_loss': g_gan_loss.item(),
            'g_content_loss': g_content_loss.item(),
            'g_perceptual_loss': g_perceptual_loss.item()
        }
    
    def save_checkpoint(self, epoch, losses):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'losses': losses
        }
        path = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint: {path}")
    
    def save_sample_images(self, epoch, lr_images, hr_images, sr_images):
        from torch.nn.functional import interpolate
        lr_images_resized = interpolate(lr_images, size=hr_images.shape[2:], mode='bilinear', align_corners=False)
        sr_images_resized = interpolate(sr_images, size=hr_images.shape[2:], mode='bilinear', align_corners=False)
        images = torch.cat([lr_images_resized.cpu(), sr_images_resized.cpu(), hr_images.cpu()], dim=-1)
        save_path = os.path.join(Config.GENERATED_DIR, f'epoch_{epoch}.png')
        save_image(images, save_path, normalize=True)

    def train(self, train_loader, num_epochs):
        """Train the SRGAN model."""
        try:
            total_steps = 0
            for epoch in range(1, num_epochs + 1):
                self.generator.train()
                self.discriminator.train()
                
                epoch_losses = []
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
                
                for batch_idx, (lr_images, hr_images) in enumerate(progress_bar):
                    lr_images = lr_images.to(self.device)
                    hr_images = hr_images.to(self.device)
                    
                    losses = self.train_step(lr_images, hr_images)
                    epoch_losses.append(losses)
                    
                    progress_bar.set_postfix(
                        d_loss=f"{losses['d_loss']:.4f}",
                        g_loss=f"{losses['g_loss']:.4f}"
                    )
                    
                    if batch_idx % 100 == 0:
                        for key, value in losses.items():
                            self.writer.add_scalar(key, value, total_steps)
                    
                    total_steps += 1
                
                avg_losses = {
                    key: sum(loss[key] for loss in epoch_losses) / len(epoch_losses)
                    for key in epoch_losses[0].keys()
                }
                
                logging.info(f"Epoch {epoch} Average Losses:")
                for key, value in avg_losses.items():
                    logging.info(f"{key}: {value:.4f}")
                
                # Save checkpoint and generate sample images after every epoch
                self.save_checkpoint(epoch, avg_losses)
                
                self.generator.eval()
                with torch.no_grad():
                    sample_lr = next(iter(train_loader))[0][:4].to(self.device)
                    sample_hr = next(iter(train_loader))[1][:4].to(self.device)
                    sample_sr = self.generator(sample_lr)
                    self.save_sample_images(epoch, sample_lr, sample_hr, sample_sr)
                self.generator.train()
                
                self.g_scheduler.step()
                self.d_scheduler.step()
        
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise
        finally:
            self.writer.close()

def main():
    try:
        setup_logging()
        logging.info("Starting SRGAN training pipeline")

        dataset_path = download_brats_dataset()
        Config.set_dataset_folder(dataset_path)
        logging.info(f"Dataset path set to: {dataset_path}")

        if not os.path.exists(Config.TRAIN_HR_FOLDER) or not os.path.exists(Config.TRAIN_LR_FOLDER):
            logging.info("Dataset not preprocessed. Starting preprocessing...")
            num_images = preprocess_dataset()
            logging.info(f"Preprocessing complete. Created {num_images} image pairs")
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

        dataset = BraTSDataset(
            Config.TRAIN_HR_FOLDER,
            Config.TRAIN_LR_FOLDER,
            transform=True
        )
        
        if len(dataset) == 0:
            raise ValueError("No valid image pairs found in the dataset")
        
        train_loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=8,  # Increased number of workers for faster data loading
            pin_memory=True  # Faster data transfer to GPU
        )
        
        logging.info(f"DataLoader created with {len(dataset)} images")
        
        trainer = Trainer()
        trainer.train(train_loader, Config.EPOCHS)
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline error: {str(e)}")
        raise

if __name__ == '__main__':
    main()