import torch 
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class upSample(nn.Sequential):
    def __init__(self, sInputs, output):
        super(upSample, self).__init__()
        self.conv1 = nn.Conv2d(sInputs, output,kernel_size=3, stride=1, padding=1)
        self.Lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(output, output,kernel_size=3, stride=1, padding=1)
        self.Lrelu2 = nn.LeakyReLU(0.2)
    
    def forward(self, x , cWidth):
        upX = F.interpolate(x, size=[cWidth.size(2), cWidth.size(3)], mode='bilinear', align_corners=True)
        return self.Lrelu2(self.conv2(self.conv1(torch.cat([upX,cWidth ], dim=1))))

class Decoder(nn.Module):
    def __init__(self, n_feat=1280, decoder_w = 0.6):
        super(Decoder, self).__init__()
        feat = int(n_feat*decoder_w)

        self.conv2 = nn.Conv2d(n_feat, feat, kernel_size=1, stride=1, padding=1)
        
        self.up_0 = upSample(sInputs=feat//1+320, output=feat//2)
        self.up_1 = upSample(sInputs=feat//2+160, output=feat//2)
        self.up_2 = upSample(sInputs=feat//2+64, output=feat//4)
        self.up_3 = upSample(sInputs=feat//4+32, output=feat//8)
        self.up_4 = upSample(sInputs=feat//8+24, output=feat//8)
        self.up_5 = upSample(sInputs=feat//8+16, output=feat//16)
        

        self.conv3 = nn.Conv2d(feat//16,1, kernel_size=3, stride=1, padding=1)

    def forward(self, feat):
        x_b0, x_b1, x_b2, x_b3, x_b4, x_b5, x_b6 = feat[2], feat[4], feat[6], feat[9], feat[15], feat[18], feat[19]
        x_d0 = self.conv2(x_b6)
        x_d1 = self.up_0(x_d0,x_b5)
        x_d2 = self.up_1(x_d1,x_b4)
        x_d3 = self.up_2(x_d2,x_b3)
        x_d4 = self.up_3(x_d3,x_b2)
        x_d5 = self.up_4(x_d4,x_b1)
        x_d6 = self.up_5(x_d5,x_b0)
        return self.conv3(x_d6)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.mobile_net = models.mobilenet_v2(pretrained=True)

    def forward(self, x):
        features = [x]
        for i,j in self.mobile_net.features._modules.items():
            features.append(j(features[-1]))
        return features

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))
