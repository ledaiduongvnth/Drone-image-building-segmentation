from unet import UNET
from skimage import io
import torch
import torchvision
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

IMAGE_HEIGHT = 325  # 650 originally
IMAGE_WIDTH = 325  # 650 originally

model = UNET(in_channels=3, out_channels=1).to(DEVICE)

model.load_state_dict(torch.load("my_checkpoint.pth.tar")["state_dict"])
# load_checkpoint(torch.load("my_checkpoint.pth.tar"), model2)
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


image = image_loader(
    "/mnt/hdd/PycharmProjects/Drone-image-building-segmentation/i2.png")

with torch.no_grad():
    preds = torch.sigmoid(model(image))
    preds = (preds > 0.5).float()
torchvision.utils.save_image(
    preds, f"/mnt/hdd/PycharmProjects/Drone-image-building-segmentation/a.png"
)

# import matplotlib.image as mpimg
#
# img = mpimg.imread(
#     "/content/drive/MyDrive/Deep_Learning/DI504/Term_Project/Datasets/SpaceNet2/Shanghai/pred_deneme.png")
# imgplot = plt.imshow(img)
# plt.show()