"""V8 architectures configured on the unchanged V7 training engine.

Configuration below changes process-local engine dependencies only; no V7 files,
checkpoints, data or results are modified. Loss, batches and generator stay fixed.
"""
from pathlib import Path
import math,json
import torch
from torch import nn
import train_residual_v7 as engine

MODEL=Path('models/residual_v8');PROTOCOL=Path('configs/residual_v8_protocol.json')
_v7_factory=engine.model_for

class MediumTransformer(engine.JointResidual):
    def __init__(self,ng):
        nn.Module.__init__(self);self.kind='medium_transformer'
        self.projections=nn.ModuleList([nn.Linear(66,256),nn.Linear(64,256),nn.Linear(64,256)])
        self.token_type=nn.Parameter(torch.randn(1,3,256)*.02)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(256,8,512,.1,'gelu',batch_first=True,norm_first=True) for _ in range(4)])
        self.norm=nn.LayerNorm(256);self.readout=nn.Linear(256,32)
        self.decoder=nn.Parameter(torch.zeros(ng,32));self.background_decoder=nn.Parameter(torch.zeros(ng,32))
        self.gate=nn.Linear(194,1);nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,math.log(.1/.9))

class SoftMixture(nn.Module):
    def __init__(self,ng):
        super().__init__()
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(130,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,32)) for _ in range(4)])
        self.router=nn.Sequential(nn.Linear(194,64),nn.SiLU(),nn.Linear(64,4))
        nn.init.zeros_(self.router[-1].weight);nn.init.zeros_(self.router[-1].bias)
        self.decoder=nn.Parameter(torch.zeros(ng,32));self.background_decoder=nn.Parameter(torch.zeros(ng,32))
        self.gate=nn.Linear(194,1);nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,math.log(.1/.9))
    def routing(self,z,h):return torch.softmax(self.router(torch.cat([z,h],dim=1)),dim=1)
    def forward(self,z,g,h,seen,ownmask=None):
        weights=self.routing(z,h);q=(torch.stack([e(z) for e in self.experts],dim=1)*weights[:,:,None]).sum(1)
        base=q@self.decoder.T;back=(q@self.background_decoder.T)*g
        s=torch.sigmoid(self.gate(torch.cat([z,h],dim=1)))
        r=s*.75*torch.tanh((base+back)/.75)*seen
        if ownmask is not None:r=r*ownmask
        return r,s,back*s

def model_for(kind,ng):
    if kind=='medium_transformer':return MediumTransformer(ng)
    if kind=='soft_moe':return SoftMixture(ng)
    return _v7_factory(kind,ng)

def configure():
    engine.MODEL=MODEL;engine.PROTOCOL=PROTOCOL;engine.model_for=model_for

def fit(*args,**kwargs):
    configure();return engine.fit(*args,**kwargs)

def predict(*args,**kwargs):
    configure();return engine.predict(*args,**kwargs)

predict_features=engine.predict_features
init_runtime=engine.init_runtime

if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps({k:sum(p.numel() for p in model_for(k,10916).parameters()) for k in ['tiny_transformer','medium_transformer','soft_moe']},indent=2))
