import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


#----------------------------------------------------------------
#                         efficientNet_2d
#----------------------------------------------------------------

# First block (low-level feature extractor)
class FeatureExtrator(nn.Module):
  def __init__(self, in_ch, num_filter, bias=False):
    super(FeatureExtrator,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=num_filter//2, 
                           kernel_size=7, dilation=(1,1), padding=3, bias=bias)
    self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=num_filter//2, 
                           kernel_size=7, dilation=(2,2), padding=6, bias=bias)
    self.conv3 = nn.Conv2d(in_channels=in_ch, out_channels=num_filter//2, 
                           kernel_size=7, dilation=(4,4), padding=12,bias=bias)
    
    self.bn1 = nn.BatchNorm2d(int(1.5*num_filter))
    self.bn2 = nn.BatchNorm2d(num_filter)
    self.relu = nn.ReLU(inplace=False)

    self.conv4 = nn.Conv2d(in_channels=int(1.5*num_filter), out_channels=num_filter, 
                            kernel_size=7, stride=1, padding =3, bias=bias)
    self.conv5 = nn.Conv2d(in_channels=in_ch, out_channels=num_filter, 
                           kernel_size=1, stride=1, padding =0, bias=bias)

  def forward(self, inp):
    x = torch.cat((self.conv1(inp), self.conv2(inp), self.conv3(inp)), dim=1)
    x = self.relu(self.bn1(x))

    x = self.bn2(self.conv4(x))

    # Shortcut Connection
    s = self.conv5(inp)

    # Addition
    x = x + s
    return x

# Residual block
class Residual_block(nn.Module):
  def __init__(self, in_ch, out_ch, stride=1, bias=False):
    super(Residual_block,self).__init__()
    self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding =1, bias=bias)
    self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding =(1,1), bias=bias)
    self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding =(0,0), bias=bias)

    self.bn = nn.BatchNorm2d(out_ch)
    self.relu = nn.ReLU(inplace=False)

  def forward(self, inp):
    x = self.relu(self.bn(self.conv1(inp)))
    x = self.relu(self.bn(self.conv2(x)))

    # Shortcut Connection
    s = self.conv3(inp)

    # Addition 
    x = x + s
    
    # Activation
    x = self.relu(x)
    return x



# Decoder block
class Decoder_block(nn.Module):
  def __init__(self, in_ch, skip_ch, out_ch):
    super(Decoder_block,self).__init__()


    self.up_sampling = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, 
                                          padding=1, output_padding=1) #nn.Upsample(scale_factor=2)
    self.resblock = Residual_block(out_ch+skip_ch, out_ch, stride=1, bias=False)

  def forward(self, inp, skip_features):
    x = self.up_sampling(inp)

    x = torch.cat((x, skip_features), dim=1)
    x = self.resblock(x)
    return x

# Transformer for denoiser
class Transformer(nn.Module):
  def __init__(self, in_ch, out_ch, repeat, stride=1, bias=False):
    super(Transformer,self).__init__()

    layers = []
    ch_tmp = in_ch
    for r in range(0, repeat):
        layers.append(
                Residual_block(ch_tmp, ch_tmp//2, stride=1, bias=False)
            )
        ch_tmp = ch_tmp // 2
    
    self.layers = nn.Sequential(*layers)

    self.classifier = nn.Sequential(
        nn.Conv2d(ch_tmp, out_ch, kernel_size=1,stride=1, bias=False),
        nn.Sigmoid(),
    )

  def forward(self, x):
    x =  self.layers(x)
    x = self.classifier(x)
    return x

# Transformer for discriminator
class Disc_Transformer(nn.Module):
  def __init__(self, in_ch, out_ch, repeat, negative_slope= 0.01, stride=1, bias=False):
    super(Disc_Transformer,self).__init__()

    layers = []
    ch_tmp = in_ch
    for r in range(0, repeat):
        layers.append(
                Residual_block(ch_tmp, ch_tmp//2, stride=1, bias=False)
            )
        ch_tmp = ch_tmp // 2
    
    self.layers = nn.Sequential(*layers)

    self.negative_slope = negative_slope
    self.classifier = nn.Sequential(
        nn.Conv2d(ch_tmp, out_ch, kernel_size=1,stride=1, bias=False),
    )


  def forward(self, x):
    x =  self.layers(x)
    x = F.leaky_relu(x, negative_slope=self.negative_slope, inplace=False)
    x = self.classifier(x)
    return x


class efficient_Unet(nn.Module):
  def __init__(self, in_ch, out_ch, filter_base=32, bias=False):
    super(efficient_Unet, self).__init__()
    f1 = 3*filter_base
    f2 = f1 + filter_base
    f3 = f2 + filter_base
    fb = f3 + filter_base

    #Feature extractor
    self.feature_extractor = FeatureExtrator(in_ch, f1) #[B,C,W,H]-->[B,f1,W,H]

    #Encoder1
    self.down_0 = Residual_block(in_ch, f1, stride=1) #[B,C,W,H]-->[B,f1,W,H]

    #Encoder2
    self.down_1 = Residual_block(f1, f2, stride=2) #[B,f1,W,H]-->[B,f2,W/2,H/2]

    #Encoder3
    self.down_2 = Residual_block(f2, f3, stride=2) #[B,f2,W/2,H/2]-->[B,f3,W/4,H/4]

    #Bridge
    self.bridge = Residual_block(f3, fb, stride=2) #[B,f3,W/4,H/4]-->[B,fb,W/8,H/8]

    #Decoder1
    self.Decoder_block1 = Decoder_block(fb, f3, f3) #[B,fb,W/8,H/8]-->[B,f3,W/4,H/4]

    #Decoder2
    self.Decoder_block2 = Decoder_block(f3, f2, f2) #[B,f3-->[B,f2,W/2,H/2]

    #Decoder3
    self.Decoder_block3 = Decoder_block(f2, 2*f1, f1) #[B,f2,W/4,H/4]-->[B,f1,W,H]


    # Transformers
    self.Transformer1 = Transformer(f1, out_ch, 1, stride=1, bias=bias) 
    self.Transformer2 = Transformer(f2, out_ch, 2, stride=1, bias=bias) 
    self.Transformer3 = Transformer(f3, out_ch, 3, stride=1, bias=bias) 

  def forward(self,inp):
    #Feature extractor
    s0 = self.feature_extractor(inp)

    #Encoder1
    s1 = self.down_0(inp) 

    #Encoder2
    s2 = self.down_1(s1) 

    #Encoder3
    s3 = self.down_2(s2) 

    #Bridge
    b = self.bridge(s3)

    #Decoder1
    x = self.Decoder_block1(b, s3)
    yhat_x4 = self.Transformer3(x.clone())


    #Decoder2
    x = self.Decoder_block2(x, s2) 
    yhat_x2 = self.Transformer2(x.clone())


    #Decoder3
    s1_con = torch.cat((s1, s0), dim=1)
    x = self.Decoder_block3(x, s1_con)
    yhat = self.Transformer1(x)

    return yhat, yhat_x2, yhat_x4






class Efficient_Unet_disc(nn.Module):
  def __init__(self, in_ch, out_ch, negative_slope, filter_base=16, bias=False):
    super(Efficient_Unet_disc, self).__init__()
    f1 = filter_base
    f2 = 2*f1
    f3 = 2*f2
    f4 = 2*f3
    f5 = f4
    fb = f5

    #Encoder1
    self.down_0 = Residual_block(in_ch, f1, stride=1) #[B,C,W,H]-->[B,f1,W,H]

    #Encoder2
    self.down_1 = Residual_block(f1, f2, stride=2) #[B,f1,W,H]-->[B,f2,W/2,H/2]

    #Encoder3
    self.down_2 = Residual_block(f2, f3, stride=2) #[B,f2,W/2,H/2]-->[B,f3,W/4,H/4]

    #Bridge
    self.bridge = Residual_block(f3, fb, stride=2) #[B,f3,W/4,H/4]-->[B,fb,W/8,H/8]
    self.bridge_map = nn.Conv2d(fb, fb, kernel_size=1, stride=1, padding =(0,0), bias=bias)
    

    #Decoder1
    self.Decoder_block1 = Decoder_block(fb, f3, f3) #[B,fb,W/8,H/8]-->[B,f3,W/4,H/4]

    #Decoder2
    self.Decoder_block2 = Decoder_block(f3, f2, f2) #[B,f3-->[B,f2,W/2,H/2]

    #Decoder3
    self.Decoder_block3 = Decoder_block(f2, f1, f1) #[B,f2,W/4,H/4]-->[B,f1,W,H]


    # Transformers
    self.Disc_Transformer1 = Disc_Transformer(f1, out_ch, 1, stride=1, negative_slope=negative_slope, bias=bias) 
    self.Disc_Transformer2 = Disc_Transformer(f2, out_ch, 2, stride=1, negative_slope=negative_slope, bias=bias) 
    self.Disc_Transformer3 = Disc_Transformer(f3, out_ch, 3, stride=1, negative_slope=negative_slope, bias=bias) 

  def forward(self,inp):

    #Encoder1
    s1 = self.down_0(inp) 

    #Encoder2
    s2 = self.down_1(s1) 

    #Encoder3
    s3 = self.down_2(s2) 

    #Bridge
    b = self.bridge(s3)
    bridge_map = self.bridge_map(b)

    #Decoder1
    x = self.Decoder_block1(b, s3)
    yhat_x4_map = self.Disc_Transformer3(x.clone())


    #Decoder2
    x = self.Decoder_block2(x, s2) 
    yhat_x2_map = self.Disc_Transformer2(x.clone())


    #Decoder3
    x = self.Decoder_block3(x, s1)
    yhat_x0_map = self.Disc_Transformer1(x.clone())

    return bridge_map, yhat_x0_map, yhat_x2_map, yhat_x4_map





if __name__ == "__main__":
    B = 2
    ch = 3
    image = torch.rand((B,ch,128,128))
    model = efficient_Unet(ch, ch)
    x1,x2,x3 = model(image)
    print('out: ', x1.shape )
    print(x2.shape)
    print(x3.shape)
