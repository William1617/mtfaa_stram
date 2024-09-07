
import torch
import torch.nn as nn


class TFCM_Block(nn.Module):
    def __init__(self,
                 cin=24,
                 K=(3, 3),
                 dila=1,
                 ):
        super(TFCM_Block, self).__init__()
        self.pconv1 = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=(1, 1)),
            nn.BatchNorm2d(cin),
            nn.PReLU(cin),
        )
        dila_pad = dila * (K[1] - 1)
      #  self.pad=nn.ConstantPad2d((dila_pad, 0, 1, 1), 0.0),
        self.dila_conv = nn.Sequential(
            nn.ConstantPad2d((dila_pad, 0, 1, 1), 0.0),
            nn.Conv2d(cin, cin, K, 1, dilation=(1, dila), groups=cin),
            nn.BatchNorm2d(cin),
            nn.PReLU(cin)
        )
    
        self.pconv2 = nn.Conv2d(cin, cin, kernel_size=(1, 1))
       

    def forward(self, inps):
        """
            inp: B x C x F x T
        """
        outs = self.pconv1(inps)
        outs = self.dila_conv(outs)
        outs = self.pconv2(outs)
        return outs +inps



class TFCM(nn.Module):
    def __init__(self,
                 cin=24,
                 K=(3, 3),
                 tfcm_layer=6,

                 ):
        super(TFCM, self).__init__()
        self.tfcm = nn.ModuleList()
        for idx in range(tfcm_layer):
            self.tfcm.append(
                TFCM_Block(cin, K, 2**idx)
            )

    def forward(self, inp):
        out = inp
        for idx in range(len(self.tfcm)):
            out = self.tfcm[idx](out)
        return out
    
if __name__ == "__main__":
    net=TFCM(48)
    inps=torch.randn(1,48,64,126)
    out=net(inps)

