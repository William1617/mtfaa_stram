import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal=True,
        complex_axis=1,
    ):
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels//2
        self.out_channels = out_channels//2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.cache_len=kernel_size[-1]-1
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[
                                   self.padding[0], 0], dilation=self.dilation, groups=self.groups)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[
                                   self.padding[0], 0], dilation=self.dilation, groups=self.groups)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs,cache=None):

        if self.complex_axis == 0:
            if(self.cache_len>0):
                inputs=torch.cat([cache,inputs],-1)
                new_cache=inputs[:,:,:,-self.cache_len:]
            else:
                new_cache=None
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)
                if(self.cache_len>0):
                    real_cache,imag_cache=torch.chunk(cache,2,self.complex_axis)
                    real=torch.cat([real_cache,real],-1)
                    imag=torch.cat([imag_cache,imag],-1)
                    newreal_cache=real[:,:,:,-self.cache_len:]
                    newimag_cache=imag[:,:,:,-self.cache_len:]
                    new_cache=torch.cat([newreal_cache,newimag_cache],self.complex_axis)
                else:
                    new_cache=None

            real2real = self.real_conv(real,)
            imag2imag = self.imag_conv(imag,)

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)
        
        return out,new_cache


def complex_cat(inps, dim=1):
    reals, imags = [], []
    for inp in inps:
        real, imag = inp.chunk(2, dim)
        reals.append(real)
        imags.append(imag)
    reals = torch.cat(reals, dim)
    imags = torch.cat(imags, dim)
    return reals, imags


class ComplexLinearProjection(nn.Module):
    def __init__(self, cin):
        super(ComplexLinearProjection, self).__init__()
        self.clp = ComplexConv2d(cin, cin)

    def forward(self, real, imag):
        """
        real, imag: B C F T
        """
        inputs = torch.cat([real, imag], 1)
        outputs,_ = self.clp(inputs)
        real, imag = outputs.chunk(2, dim=1)
        outputs = torch.sqrt(real**2+imag**2+1e-8)
        return outputs


class PhaseEncoder(nn.Module):
    def __init__(self, cout, n_sig, cin=2, alpha=0.5):
        super(PhaseEncoder, self).__init__()
        self.complexnn = nn.ModuleList()
        for _ in range(n_sig):
            self.complexnn.append(
                ComplexConv2d(cin, cout, (1, 3)))
        self.clp = ComplexLinearProjection(cout*n_sig)
        self.alpha = alpha

    def forward(self, cspecs,conv_cachses):
        """
        cspec: B C F T
        """
        outs = []
        out_caches=[]
        for idx, layer in enumerate(self.complexnn):
            out,new_cache=layer(cspecs[idx],conv_cachses[idx:idx+1])
            outs.append(out)
            out_caches.append(new_cache)
        real, imag = complex_cat(outs, dim=1)
        
        amp = self.clp(real, imag)
        out_cache=torch.cat(out_caches,dim=0)
        return amp**self.alpha,out_cache


if __name__ == "__main__":
    # 32ms@48kHz, concatenation of [real, imag], dim=1
    nnet=PhaseEncoder(4,1)
    inp=torch.randn(1,2,769,1)
    conv_cache=torch.randn(1,2,769,2)
    out,new_cache=nnet([inp],conv_cache)
  