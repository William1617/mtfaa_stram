import einops
import torch as th
import torch.nn as nn


class ASA(nn.Module):
    def __init__(self, c=64, causal=True):
        super(ASA, self).__init__()
        self.d_c = c//4
        self.f_qkv = nn.Sequential(
            nn.Conv2d(c, self.d_c*3, kernel_size=(1, 1),bias=False),
            nn.BatchNorm2d(self.d_c*3),
            nn.PReLU(self.d_c*3),
        )
        self.t_qk = nn.Sequential(
            nn.Conv2d(c, self.d_c*2, kernel_size=(1, 1),bias=False),
            nn.BatchNorm2d(self.d_c*2),
            nn.PReLU(self.d_c*2),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(self.d_c, c,kernel_size=(1, 1),  bias=False),
            nn.BatchNorm2d(c),
            nn.PReLU(c),
        )
        self.causal = causal

    def forward(self, inp,cache):
        """
        inp: B C F T
        """
        # f-attention
        f_qkv = self.f_qkv(inp)
        k_cache, v_cache = th.split(cache,
                                                 cache.size(1) // 2,
                                                 dim=1)
        qf, kf, v = tuple(einops.rearrange(
            f_qkv, "b (c k) f t->k b c f t", k=3))
        
        f_score = th.einsum("bcft,bcyt->btfy", qf, kf) / (self.d_c**0.5)

        f_score = f_score.softmax(dim=-1)
        f_out = th.einsum('btfy,bcyt->bcft', [f_score, v])
        f_out=th.cat([v_cache,f_out],-1)
        new_vcache=f_out[:,:,:,1:]
        # t-attention
        t_qk = self.t_qk(inp)
        qt, kt = tuple(einops.rearrange(t_qk, "b (c k) f t->k b c f t", k=2))
        kt=th.cat([k_cache,kt],-1)
        new_kcache=kt[:,:,:,1:]
        
        t_score = th.einsum('bcft,bcfy->bfty', [qt, kt]) / (self.d_c**0.5)
        
        t_score = t_score.softmax(dim=1)
       
        t_out = th.einsum('bfty,bcfy->bcft', [t_score, f_out])
        out = self.proj(t_out)
        new_cache=th.cat([new_kcache,new_vcache],1)
        return out+inp,new_cache


def test_asa():
    nnet = ASA(c=64)
    inp = th.randn(1, 64, 4, 1)
    cache=th.randn(1, 32, 4, 4)
    
    out,new_cache = nnet(inp,cache)
    print(out.shape,new_cache.shape)
    


if __name__ == "__main__":
    test_asa()