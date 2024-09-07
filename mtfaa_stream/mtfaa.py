"""
Multi-scale temporal frequency axial attention neural network (MTFAA).

shmzhang@aslp-npu.org, 2022
"""

import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as tf
from typing import List

from tfcm import TFCM
from asa import ASA
from phase_encoder import PhaseEncoder
from f_sampling import FD, FU
from erb import Banks


def parse_1dstr(sstr: str) -> List[int]:
    return list(map(int, sstr.split(",")))


def parse_2dstr(sstr: str) -> List[List[int]]:
    return [parse_1dstr(tok) for tok in sstr.split(";")]


eps = 1e-10


class MTFAANet(nn.Module):

    def __init__(self,
                 n_sig=1,
                 PEc=4,
                 Co="48,96,192",
                 O="1,1,1",
                 bottleneck_layer=2,
                 tfcm_layer=6,
                 mag_f_dim=3,
                 win_len=32*48,
                 nerb=256,
                 sr=48000,
                 ):
        super(MTFAANet, self).__init__()
        self.PE = PhaseEncoder(PEc, n_sig)
        # 32ms @ 48kHz
        self.ERB = Banks(nerb, win_len, sr,0,sr//2)
        self.encoder_fd = nn.ModuleList()
        self.encoder_tfcm = nn.ModuleList()
        self.encoder_asa = nn.ModuleList()
        self.bottleneck_tfcm = nn.ModuleList()
        self.bottleneck_asa = nn.ModuleList()
        self.decoder_fu = nn.ModuleList()
        self.decoder_tfcm = nn.ModuleList()
        self.decoder_asa = nn.ModuleList()
        C_en = [PEc//2*n_sig] + parse_1dstr(Co)
        C_de = [4] + parse_1dstr(Co)
        O = parse_1dstr(O)
        for idx in range(len(C_en)-1):
            self.encoder_fd.append(
                FD(C_en[idx], C_en[idx+1]),
            )
            self.encoder_tfcm.append( TFCM(C_en[idx+1], (3, 3),
                         tfcm_layer=tfcm_layer))
            self.encoder_asa.append( ASA(C_en[idx+1]))

        for idx in range(bottleneck_layer):
            self.bottleneck_tfcm.append(TFCM(C_en[-1], (3, 3),
                         tfcm_layer=tfcm_layer))
            self.bottleneck_asa.append(ASA(C_en[-1]))

        for idx in range(len(C_de)-1, 0, -1):
            self.decoder_fu.append(
                FU(C_de[idx], C_de[idx-1], O=(O[idx-1], 0)),
            )
            self.decoder_tfcm.append(TFCM(C_de[idx-1], (3, 3),
                         tfcm_layer=tfcm_layer) )
            self.decoder_asa.append(ASA(C_de[idx-1]))
    
        # MEA is causal, so mag_t_dim = 1.
        self.mag_mask = nn.Conv2d(
            4, mag_f_dim, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        kernel = th.eye(mag_f_dim)
        kernel = kernel.reshape(mag_f_dim, 1, mag_f_dim, 1)
        self.register_buffer('kernel', kernel)
        self.mag_f_dim = mag_f_dim


    def forward(self, D_cspec,pe_cache,tfcm_cache1,tfcm_cache2,tfcm_cache3,tfcm_cache4,asa_cache1,asa_cache2,asa_cache3,asa_cache4):
        """
        sigs: list [B N] of len(sigs)
        """

        mag = th.norm(D_cspec, dim=1)
        pha = torch.atan2(D_cspec[:, -1, ...], D_cspec[:, 0, ...])
        pe_out,new_pecahe=self.PE([D_cspec],pe_cache)
        out = self.ERB.amp2bank(pe_out)
      
        encoder_out = []
        
        tfcm_encache1,tfcm_decache2=torch.split(tfcm_cache1,1,dim=0)
        tfcm_encache2,tfcm_decache1=torch.split(tfcm_cache2,1,dim=0)
        tfcm_encache3,tfcm_bocahce1,tfcm_bocahce2=torch.split(tfcm_cache3,1,dim=0)
        tfcm_decache3=tfcm_cache4
        encoder_tfcmcache=[tfcm_encache1,tfcm_encache2,tfcm_encache3]
        decoder_tfcmcache=[tfcm_decache1,tfcm_decache2,tfcm_decache3]
        bottle_tfcmcache=[tfcm_bocahce1,tfcm_bocahce2]

        asa_encache1,asa_decache2=torch.split(asa_cache1,1,dim=0)
        asa_encache2,asa_decache1=torch.split(asa_cache2,1,dim=0)
        asa_encache3,asa_bocache1,asa_bocache2=torch.split(asa_cache3,1,dim=0)
        asa_decache3=asa_cache4

        encoder_asacache=[asa_encache1,asa_encache2,asa_encache3]
        decoder_asacache=[asa_decache1,asa_decache2,asa_decache3]
        bottle_asacache=[asa_bocache1,asa_bocache2]

        for idx in range(len(self.encoder_fd)):
            out = self.encoder_fd[idx](out)
            encoder_out.append(out)
            out ,encoder_tfcmcache[idx]= self.encoder_tfcm[idx](out,encoder_tfcmcache[idx])
            
            out,encoder_asacache[idx]= self.encoder_asa[idx](out,encoder_asacache[idx])
            #print('encoder',out.shape)

        for idx in range(len(self.bottleneck_asa)):
            out ,bottle_tfcmcache[idx]= self.bottleneck_tfcm[idx](out,bottle_tfcmcache[idx])
          
            out,bottle_asacache[idx] = self.bottleneck_asa[idx](out,bottle_asacache[idx])
            #print('biott',out.shape)

        for idx in range(len(self.decoder_fu)):
            out = self.decoder_fu[idx](out, encoder_out[-1-idx])
            out ,decoder_tfcmcache[idx]= self.decoder_tfcm[idx](out,decoder_tfcmcache[idx])
            #print('decoder',out.shape)
            out,decoder_asacache[idx]=self.decoder_asa[idx](out,decoder_asacache[idx])
           

        new_tfcmcache1=th.cat([encoder_tfcmcache[0],decoder_tfcmcache[1]],0)
        new_tfcmcache2=th.cat([encoder_tfcmcache[1],decoder_tfcmcache[0]],0)
        new_tfcmcache3=th.cat([encoder_tfcmcache[2],bottle_tfcmcache[0],bottle_tfcmcache[1]],0)
        new_tfcmcache4=decoder_tfcmcache[2]

        new_asacache1=th.cat([encoder_asacache[0],decoder_asacache[1]],0)
        new_asacache2=th.cat([encoder_asacache[1],decoder_asacache[0]],0)
        new_asacache3=th.cat([encoder_asacache[2],bottle_asacache[0],bottle_asacache[1]],1)
        new_asacache4=decoder_asacache[2]
        out = self.ERB.bank2amp(out)
        # stage 1
        mag_mask = self.mag_mask(out)
        mag_pad = tf.pad(
            mag[:, None], [0, 0, (self.mag_f_dim-1)//2, (self.mag_f_dim-1)//2])
        mag = tf.conv2d(mag_pad, self.kernel)
        mag = mag * mag_mask.sigmoid()
        mag = mag.sum(dim=1)
        # stage 2
        real_mask = self.real_mask(out).squeeze(1)
        imag_mask = self.imag_mask(out).squeeze(1)

        mag_mask = th.sqrt(th.clamp(real_mask**2+imag_mask**2, eps))
        pha_mask = th.atan2(imag_mask+eps, real_mask+eps)
        real = mag * mag_mask.tanh() * th.cos(pha+pha_mask)
        imag = mag * mag_mask.tanh() * th.sin(pha+pha_mask)
       # print(mag.shape,imag.shape)
        return real,imag,new_pecahe,new_tfcmcache1,new_tfcmcache2,new_tfcmcache3,new_tfcmcache4,new_asacache1,new_asacache2,new_asacache3,new_asacache4




if __name__ == "__main__":
    # test_nnet()
    nnet = MTFAANet(n_sig=1)
    inp=torch.randn(1,2,769,1)
    pe_cache=torch.randn(1,2,769,2)
    tfcm_cache1=torch.randn(2,48,64,126)
    tfcm_cache2=torch.randn(2,96,16,126)
    tfcm_cache3=torch.randn(3,192,4,126)
    tfcm_cache4=torch.randn(1,4,256,126)

    asa_cache1=torch.randn(2,24,64,200)
    asa_cache2=torch.randn(2,48,16,200)
    asa_cache3=torch.randn(3,96,4,200)
    asa_cache4=torch.randn(1,2,256,200)
    nnet.eval()
    #torch.save(nnet.state_dict(),'./mtfaa2.pt')
    nnet.load_state_dict(torch.load('./mtfaa.pt'))
    torch.onnx.export(nnet,(inp,pe_cache,tfcm_cache1,tfcm_cache2,tfcm_cache3,tfcm_cache4,asa_cache1,asa_cache2,asa_cache3,asa_cache4),
                      "./mtfaa_stream.onnx",
                      input_names=['in_fea','pe_cache','tfcm_cache1','tfcm_cache2','tfcm_cache3','tfcm_cache4','asa_cache1','asa_cache2','asa_cache3','asa_cache4'],
                      output_names=['enh_real','enh_imag','mew_pecache','newtfcm_cache1','newtfcm_cache2','newtfcm_cache3','newtfcm_cache4','newasa_cache1','newasa_cache2','newasa_cache3','newasa_cache4'],
                      opset_version=12)
  #  torch.save(nnet.state_dict(),'./mtfaa.pt')