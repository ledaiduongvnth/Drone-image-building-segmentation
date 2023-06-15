import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# For using U-Net properly, as it seen below in the architecture, it is always used two convolutions one another after.
# That's why we will create the class DoubleConv for double convolution.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # Kernel size 3, stride to 1, padding size of 1 which helps to obtain same convolution.
            nn.BatchNorm2d(out_channels),  # input height and width will be same after convolution.
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # We will copy-paste the first convolution with only difference that
            nn.BatchNorm2d(out_channels),  # changing in_channels, out_channels to out_channels, out_channels
            nn.ReLU(inplace=True),  # For second convolution. And we had bias=False because
        )  # we will use Batch Normalization.

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],  #In Unet paper, out channels are two but we are going to do
    ):                                                                          #binary image segmentation. So, we can output a single channel.
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))                 #We will fill the modulelist created above with features.
            in_channels = feature                                               #This will create a loop inside of conv layers.

        # Up part of UNET
        for feature in reversed(features):                                      #It will be the reversed version of features
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,                #We will add the skip connections after, so we need to concenate
                )                                                               #For example, it will going to be 512 * 2 = 1024, so it will be the case for all of the transpose elements.
            )                                                                   #Kernel_size and stride parameters are = 2 so this will double the height of the image.
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)              #Last in features list is the bottom part of the architecture so with doing features [-1]
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)   #The final conv should not change the height and width of the image. Just change the number of channels.

    def forward(self, x):
        skip_connections = []                                                   #We will store all of the skip connections in here.

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)                                          #We add skip connections right before downsampling.
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]                               #First sample has the highest resolution, so we will reverse it.

        for idx in range(0, len(self.ups), 2):                                  #Up, double conv, up, double conv. That's why we have the range here.
            x = self.ups[idx](x)                                                #Probably, there are better ways to doing this. But I came out with this solution.
            skip_connection = skip_connections[idx//2]                          #Because of step of 2, we used index / 2 for skip connection.

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)                #We upsample, have skip connection and then concetenate.
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)