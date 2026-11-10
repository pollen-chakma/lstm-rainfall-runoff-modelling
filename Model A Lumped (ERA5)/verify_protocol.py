from pathlib import Path
import runpy,numpy as np
b=Path(__file__).resolve().parent
m=runpy.run_path(str(b/'train_unseen.py'))
d=np.arange(np.datetime64('1998-01-01'),np.datetime64('2002-01-01'))
c=dict(training_basins=['a','b'],unseen_basins=['held'],validation_start='2000-01-01',test_start='2001-01-01',lookback=30)
def record(name,k):
 return dict(id=name,dates=d.copy(),x=np.tile(np.arange(7,dtype='float32'),(len(d),1))+k,y=np.arange(len(d),dtype='float32')/100+k,static=np.arange(5,dtype=float)+k)
rs=[record('a',1),record('b',2)]; rs[0]['static'][2]=np.nan
p=m['fit_preprocessing'](c,rs)
changed=[dict(r,y=r['y'].copy()) for r in rs]
for r in changed:r['y'][d>=np.datetime64('2000-01-01')]=999999
p2=m['fit_preprocessing'](c,changed)
assert all(np.allclose(p[k],p2[k]) for k in p),'Later targets affected preprocessing'
try:
 m['fit_preprocessing'](c,rs+[record('held',3)])
 raise RuntimeError('Heldout inclusion was accepted')
except AssertionError: pass
train=m['Windows'](c,rs,p,'train'); sampler=m['EqualBasinSampler'](train,17,42)
ix=list(sampler); bs=np.searchsorted(train.ends,np.array(ix),side='right')
assert np.array_equal(np.bincount(bs),[17,17]),'Unequal basin sampling'
assert np.isfinite(train[0][1].numpy()).all(),'Static imputation failed'
held=record('held',3); u=m['Windows'](c,[held],p,'unseen')
held2=dict(held,y=np.full(len(d),np.nan)); u2=m['Windows'](c,[held2],p,'unseen')
assert np.array_equal(u.indices[0],u2.indices[0]),'Test labels changed prediction dates'
assert u[0][2].item()==0.,'Unseen target exposed in model input batch'
assert all(r['dates'][ts].max()<np.datetime64(c['validation_start']) for r,ts in zip(rs,train.indices))
print('PASS: donor-only fits, later-target invariance, equal-basin sampling, training-only imputation, unseen-label independence, temporal boundaries')
