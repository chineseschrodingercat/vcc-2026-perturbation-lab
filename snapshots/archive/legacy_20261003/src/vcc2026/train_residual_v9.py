"""Two additional context-fusion architectures on the frozen training engine."""
from pathlib import Path
import math,json
import torch
from torch import nn
import train_residual_v7 as engine
from train_residual_v8 import model_for as previous_factory

MODEL=Path('models/residual_v9');PROTOCOL=Path('configs/residual_v9_protocol.json')

class ResidualHead(nn.Module):
    def head_init(self,ng):
        self.decoder=nn.Parameter(torch.zeros(ng,32));self.background_decoder=nn.Parameter(torch.zeros(ng,32))
        self.gate=nn.Linear(194,1);nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,math.log(.1/.9))
    def forward(self,z,g,h,seen,ownmask=None):
        q=self.latent(z,h);base=q@self.decoder.T;back=(q@self.background_decoder.T)*g
        s=torch.sigmoid(self.gate(torch.cat([z,h],dim=1)));r=s*.75*torch.tanh((base+back)/.75)*seen
        if ownmask is not None:r=r*ownmask
        return r,s,back*s

class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__();self.norms=nn.ModuleList([nn.LayerNorm(128) for _ in range(4)])
        self.self_attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
        self.cross_attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128));self.drop=nn.Dropout(.1)
    def forward(self,q,context):
        t=self.norms[0](q);q=q+self.drop(self.self_attention(t,t,t,need_weights=False)[0])
        t=self.norms[1](q);c=self.norms[2](context);q=q+self.drop(self.cross_attention(t,c,c,need_weights=False)[0])
        return q+self.drop(self.ff(self.norms[3](q)))

class CrossAttention(ResidualHead):
    def __init__(self,ng):
        super().__init__();self.target_projections=nn.ModuleList([nn.Linear(66,128),nn.Linear(64,128)])
        self.context_projection=nn.Linear(8,128);self.target_type=nn.Parameter(torch.randn(1,2,128)*.02);self.context_type=nn.Parameter(torch.randn(1,8,128)*.02)
        self.layers=nn.ModuleList([CrossBlock() for _ in range(2)]);self.norm=nn.LayerNorm(128);self.readout=nn.Linear(128,32);self.head_init(ng)
    def latent(self,z,h):
        blocks=[torch.cat([z[:,:64],z[:,128:]],dim=1),z[:,64:128]]
        q=torch.stack([p(b) for p,b in zip(self.target_projections,blocks)],dim=1)+self.target_type
        context=self.context_projection(h.reshape(-1,8,8))+self.context_type
        for layer in self.layers:q=layer(q,context)
        return self.readout(self.norm(q).mean(1))

class DenseExpertFF(nn.Module):
    def __init__(self):
        super().__init__();self.experts=nn.ModuleList([nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,128)) for _ in range(4)])
        self.router=nn.Linear(128,4);nn.init.zeros_(self.router.weight);nn.init.zeros_(self.router.bias)
    def forward(self,x):
        w=torch.softmax(self.router(x),dim=-1);return (torch.stack([e(x) for e in self.experts],dim=-2)*w[...,None]).sum(-2)

class AttentionExpertBlock(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(128);self.norm2=nn.LayerNorm(128)
        self.attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True);self.ff=DenseExpertFF();self.drop=nn.Dropout(.1)
    def forward(self,x):
        z=self.norm1(x);x=x+self.drop(self.attention(z,z,z,need_weights=False)[0]);return x+self.drop(self.ff(self.norm2(x)))

class AttentionMixture(ResidualHead):
    def __init__(self,ng):
        super().__init__();self.projections=nn.ModuleList([nn.Linear(66,128),nn.Linear(64,128),nn.Linear(64,128)])
        self.token_type=nn.Parameter(torch.randn(1,3,128)*.02);self.layers=nn.ModuleList([AttentionExpertBlock() for _ in range(2)])
        self.norm=nn.LayerNorm(128);self.readout=nn.Linear(128,32);self.head_init(ng)
    def latent(self,z,h):
        blocks=[torch.cat([z[:,:64],z[:,128:]],dim=1),z[:,64:128],h]
        t=torch.stack([p(b) for p,b in zip(self.projections,blocks)],dim=1)+self.token_type
        for layer in self.layers:t=layer(t)
        return self.readout(self.norm(t).mean(1))

def model_for(kind,ng):
    if kind=='cross_attention':return CrossAttention(ng)
    if kind=='attention_moe':return AttentionMixture(ng)
    return previous_factory(kind,ng)

def configure():engine.MODEL=MODEL;engine.PROTOCOL=PROTOCOL;engine.model_for=model_for
def fit(*args,**kwargs):configure();return engine.fit(*args,**kwargs)
def predict(*args,**kwargs):configure();return engine.predict(*args,**kwargs)
predict_features=engine.predict_features
init_runtime=engine.init_runtime

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps({k:sum(p.numel() for p in model_for(k,10916).parameters()) for k in ['cross_attention','attention_moe']},indent=2))
