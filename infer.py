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

IMAGE_HEIGHT = 650
IMAGE_WIDTH = 650

model = UNET(in_channels=3, out_channels=1).to(DEVICE)

model.load_state_dict(torch.load("model0_325.pt")["state_dict"])
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




img_path = "/mnt/hdd/PycharmProjects/Drone-image-building-segmentation/split_images/output/0_11160.png"
opencv_img = cv2.imread(img_path)
image = image_loader(img_path)
preds = None
with torch.no_grad():
    preds = torch.sigmoid(model(image))
    preds = (preds > 0.5).float()


preds = preds[0][0].detach().cpu().numpy()
preds = preds * 255
masked_img = preds.astype(np.uint8)
masked_img = cv2.resize(masked_img, (360, 360))

contours = cv2.findContours(masked_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
contours = list(filter(lambda cnt: cv2.contourArea(cnt)> 400, contours))
cv2.drawContours(opencv_img, contours, -1, (0, 0, 255), 2)
cv2.imshow("img", opencv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# torchvision.utils.save_image(
#     preds, f"/mnt/hdd/PycharmProjects/Drone-image-building-segmentation/0_11160_out_0.png"
# )

# import matplotlib.image as mpimg
#
# img = mpimg.imread(
#     "/content/drive/MyDrive/Deep_Learning/DI504/Term_Project/Datasets/SpaceNet2/Shanghai/pred_deneme.png")
# imgplot = plt.imshow(img)
# plt.show()


# im_gray = cv2.imread('/mnt/hdd/PycharmProjects/Drone-image-building-segmentation/0_11160_out_0.png', cv2.IMREAD_GRAYSCALE)
# print(im_gray.shape)