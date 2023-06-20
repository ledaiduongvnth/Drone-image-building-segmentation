import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage import io

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from unet import UNET

class SpaceNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        # All necessary information for training from Dataset method
        # is located here. .tif directory, masked image directory,
        # transform method
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index].replace("mask.tif", ".jpeg"))
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = io.imread(img_path)
        mask = io.imread(mask_path)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


"""# Utils"""


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = SpaceNetDataset(  # Specifies everything about the dataset
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SpaceNetDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model,
                   device="cuda"):  # For semantic segmentation, we need the output for each individual pixel.
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()  # While implementing this check_accuracy def from some source, it didn't have .float() so I added that.
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
            # Dice score is only for binary evaluation. For multi-class version of this code, this prediction will change.
                    (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


def save_predictions_as_imgs(
        loader, model, folder="", device="cuda"
):
    model.eval()
    # try:
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"

        )
        torchvision.utils.save_image(y.unsqueeze(1).float(), f"{folder}{idx}.png")


"""# Training"""

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 200
NUM_WORKERS = 2
IMAGE_HEIGHT = 650  # 650 originally
IMAGE_WIDTH = 650  # 650 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/mnt/hdd/Datasets/AOI_5_Khartoum_Train/RGB-PanSharpen-8bit/"
TRAIN_MASK_DIR = "/mnt/hdd/Datasets/AOI_5_Khartoum_Train/masktif/"
VAL_IMG_DIR = "/mnt/hdd/Datasets/AOI_5_Khartoum_Train/RGB-PanSharpen-8bit/"
VAL_MASK_DIR = "/mnt/hdd/Datasets/AOI_5_Khartoum_Train/masktif/"


def train_fn(loader, model, optimizer, loss_fn, scaler):  # General Structure
    # Create dataloader
    loop = tqdm(loader)  # tqdm... what a life saver. Observing the process is as much important as the training.

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)  # For binary cross entropy loss.

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    # Transform class which has resize, rotate, horizontal and vertical flip and normalization
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,  # We need a number between 0-1 so we divide the result to 255.0
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(
        DEVICE)  # Our out_channels = 1 just for now. But for the rest of the project, digitization will have apprx. at least 100 classes.
    loss_fn = nn.BCEWithLogitsLoss()  # On our output, we don't have sigmoid function. So I used binary cross entropy with logits loss.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(  # It is ugly, but let's bring our all loaders.
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        print("=> Loading checkpoint")
        model.load_state_dict(torch.load("my_checkpoint.pth.tar")["state_dict"])
        # load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)  # Let's send everything to train function.

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        filename = "model"+"_"+str(IMAGE_HEIGHT)+".pt"
        save_checkpoint(checkpoint, filename)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model,
            folder="/mnt/hdd/PycharmProjects/Drone-image-building-segmentation/output/",
            device=DEVICE
        )


if __name__ == "__main__":
    main()