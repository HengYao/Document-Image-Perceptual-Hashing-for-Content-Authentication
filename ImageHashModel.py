import torch
import torch.nn as nn
from torchvision.models import *
from einops.layers.torch import Rearrange
from transformer import Transformer
from options import HiDDenConfiguration
from einops import rearrange, repeat

class ResNetVit(nn.Module):

    def __init__(self, config: HiDDenConfiguration,img_size=256, nodown=True, patch_height=16,
                 patch_width=16, n_class=50, dim=256,
                 depth=4, heads=8, droupout=0.5):
        super().__init__()
        self.nodown = nodown
        self.img_size = img_size
        self.dim = dim
        self.depth=depth
        self.heads=heads
        self.dropouts=droupout


        self.net = resnet18(pretrained=True)
        self.filters = [64, 64, 128, 256, 512]

        if self.nodown:
            self.firstconv = nn.Conv2d(in_channels=3, out_channels=self.filters[0], kernel_size=3, stride=1, padding=0,
                                       bias=False)
        else:
            self.firstconv = nn.Conv2d(in_channels=3, out_channels=self.filters[0], kernel_size=7, stride=2, padding=3,
                                       bias=False)
        self.firstbn = self.net.bn1
        self.firstrelu = self.net.relu
        self.firstmaxpool = self.net.maxpool
        self.encoder1 = self.net.layer1
        self.encoder2 = self.net.layer2
        self.encoder3 = self.net.layer3
        self.encoder4 = self.net.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --------------------------------------------------------------------------------
        # ----------------------- define transformer parameters --------------------------
        # --------------------------------------------------------------------------------
        # patch_dim = patch_height * patch_width * 3
        num_patches = (self.img_size // patch_height) * (self.img_size // patch_height)
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) c p1 p2', p1=patch_height, p2=patch_width)
            #Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_height, p2=patch_width)
        )
        self.to_image = nn.Sequential(
            Rearrange('b (p1 p2) h w -> b (p1 h) (p2 w)',
                      p1=self.img_size // patch_height, p2=self.img_size // patch_width)
        )
        self.patch_embedding = nn.Linear(self.filters[4] * patch_height * patch_width,
                                         self.dim)
        self.patch_decoding = nn.Linear(dim, self.filters[4])
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(self.dropouts)

        self.transformer = Transformer(dim, depth=self.depth, heads=self.heads,
                                      dim_head=self.dim, mlp_dim=self.dim, dropout=self.dropouts)


    # resnet forward
    def forward_one(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)

        x = self.firstmaxpool(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        return x


    def forward(self, achor):
        # get feature map by resnet
        achor = self.forward_one(achor)

        x = self.to_patch(achor)
        (batch, num_block, c, blook_h, blook_w) = x.shape

        # Dimension Conversion
        x = x.view(batch, num_block, -1)
        x = self.patch_embedding(x)

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(num_block + 1)]
        x = self.dropout(x)

        # vision transformer extract feature
        x = self.transformer(x)


        cls_x = x[:batch, 0, :]
        return cls_x



if __name__ == '__main__':
    a = torch.rand(1,3,256,256)
    c = HiDDenConfiguration(256,256,50)
    model = ResNetVit(c)
    print(model)
    b = model(a)
    print(b)