from unet import UNET
from skimage import io
import torch
import torchvision
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import cv2
import numpy as np

IMAGE_HEIGHT = 325  # 650 originally
IMAGE_WIDTH = 325  # 650 originally

model = UNET(in_channels=3, out_channels=1).to(DEVICE)

model.load_state_dict(torch.load("/mnt/hdd/PycharmProjects/Drone-image-building-segmentation/model_325_epoch73_.pt")["state_dict"])
model.eval()

loader1 = A.Compose(
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


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = io.imread(image_name)

    transform = loader1

    augmentations = transform(image=image)
    image = augmentations["image"]

    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image.cuda()  # assumes that you're using GPU




import time
# Creating a datetime object so we can test.


import glob
for img_path in glob.glob("/mnt/hdd/PycharmProjects/Drone-image-building-segmentation/split_images/input/*.png"):
    print(img_path)
    opencv_img = cv2.imread(img_path)
    image = image_loader(img_path)
    preds = None
    with torch.no_grad():
        preds = torch.sigmoid(model(image))
        preds = (preds > 0.5).float()

    preds = preds[0][0].detach().cpu().numpy()
    preds = preds * 255
    masked_img = preds.astype(np.uint8)
    masked_img = cv2.resize(masked_img, (320, 320))

    contours = cv2.findContours(masked_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = list(filter(lambda cnt: cv2.contourArea(cnt) > 200, contours))
    cv2.drawContours(opencv_img, contours, -1, (0, 0, 255), 2)
    now = time.time()
    output_img_path = "/mnt/hdd/PycharmProjects/Drone-image-building-segmentation/split_images/output/" + str(now) + ".png"
    cv2.imwrite(output_img_path, opencv_img)