from pathlib import Path
import json,time,random
import numpy as np
import torch
from torch import nn
U=Path(__file__).resolve().parent
DEVICE=torch.device('cuda:0');assert torch.cuda.is_available()
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
exec((U/'architecture.txt').read_text())
checks=verify_ea_architecture();print('ARCHITECTURE CHECKS',checks,flush=True)
class Plain(nn.Module):
    def __init__(self):
        super().__init__();self.lstm=nn.LSTM(27,256,batch_first=True);self.output_dropout=nn.Dropout(.2);self.head=nn.Linear(256,1)
    def forward(self,x):
        z,_=self.lstm(x);return self.head(self.output_dropout(z[:,-1])).squeeze(-1)
def measure(cls):
    model=cls().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=.001)
    dynamic=torch.randn(256,365,7,device=DEVICE);static=torch.randn(256,1,20,device=DEVICE)
    x=torch.cat([dynamic,static.expand(-1,365,-1)],-1);y=torch.randn(128,device=DEVICE)
    def step():
        model.train();opt.zero_grad(set_to_none=True);q=model(x[:128]);loss=((q-y)**2).mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();float(loss.detach())
    for _ in range(3):step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    for _ in range(12):step()
    torch.cuda.synchronize();train=(time.perf_counter()-t)/12
    model.eval();t=time.perf_counter()
    with torch.no_grad():
        for _ in range(8):model(x)
    torch.cuda.synchronize();val=(time.perf_counter()-t)/8
    if hasattr(model,'assert_structural_zeros'):model.assert_structural_zeros()
    out=dict(training_batch=128,validation_batch=256,train_seconds_per_batch=train,validation_seconds_per_batch=val,projected_compute_seconds_per_epoch=2652*train+645*val,peak_allocated_bytes=torch.cuda.max_memory_allocated(),stored_parameters=sum(p.numel() for p in model.parameters()))
    if hasattr(model,'effective_parameter_count'):out['effective_parameters']=model.effective_parameter_count()
    del model,opt,x,y,dynamic,static;torch.cuda.empty_cache();return out
plain=measure(Plain);ea=measure(MeteorologyLSTM)
ratio=ea['projected_compute_seconds_per_epoch']/plain['projected_compute_seconds_per_epoch']
result=dict(status='passed',checks=checks,plain=plain,ea=ea,relative_runtime=ratio,estimated_epoch_seconds_using_measured_baseline=68*ratio,estimated_50_epoch_hours=50*68*ratio/3600,gpu=torch.cuda.get_device_name(0))
(U/'benchmark_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
