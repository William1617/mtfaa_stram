"""
linear FBank instead of ERB scale.
NOTE To to reduce the reconstruction error, the linear fbank is used.
shmzhang@aslp-npu.org, 2022
"""

import torch as th
import torch.nn as nn
from spafe.fbanks import linear_fbanks


class Banks(nn.Module):
    def __init__(self, nfilters, nfft, fs, low_freq, high_freq, learnable=False):
        super(Banks, self).__init__()
        self.nfilters, self.nfft, self.fs = nfilters, nfft, fs
        filter, _ = linear_fbanks.linear_filter_banks(
            nfilts=self.nfilters,
            nfft=self.nfft,
            low_freq=low_freq,
            high_freq=high_freq,
            fs=self.fs,
        )
        filter = th.from_numpy(filter).float()
        if not learnable:
            #  30% energy compensation.
            self.register_buffer('filter', filter*1.3)
            self.register_buffer('filter_inv', th.pinverse(filter))
        else:
            self.filter = nn.Parameter(filter)
            self.filter_inv = nn.Parameter(th.pinverse(filter))

    def amp2bank(self, amp):
        amp_feature = th.einsum("bcft,kf->bckt", amp, self.filter)
        return amp_feature

    def bank2amp(self, inputs):
        return th.einsum("bckt,fk->bcft", inputs, self.filter_inv)





if __name__ == '__main__':
    net = Banks(256, 32*48, 48000,0,24000)

