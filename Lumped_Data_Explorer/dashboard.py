from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json, argparse, math, csv, io, functools, gzip
import numpy as np

BASE=Path(__file__).resolve().parent
@functools.lru_cache(maxsize=6000)
def arrays(source,basin):
    p=BASE/'cache'/source
    return np.load(p/(basin+'_dates.npy'),mmap_mode='r'),np.load(p/(basin+'_values.npy'),mmap_mode='r')

def query(payload):
    m=json.loads((BASE/'manifest.json').read_text()); features=m['features']
    sources=payload.get('sources',['Daymet']); ids=payload.get('basins',[]); params=payload.get('parameters',['prcp'])
    assert sources and all(s in m['sources'] for s in sources)
    assert ids and len(ids)==len(set(ids)) and set(ids)<={b['basin_id'] for b in m['basins']}
    assert 1<=len(params)<=3 and all(p in features for p in params)
    start=np.datetime64(payload['start'],'D').astype(int);end=np.datetime64(payload['end'],'D').astype(int);assert start<=end
    budget=max(24,min(2400,80000//(len(ids)*len(sources))))
    series=[];stats=[]; index={b['basin_id']:b for b in m['basins']}
    for s in sources:
        for b in ids:
            dd,vv=arrays(s,b);lo=np.searchsorted(dd,start);hi=np.searchsorted(dd,end,side='right');d=dd[lo:hi];v=vv[lo:hi]
            for p in dict.fromkeys(params):
                y=np.asarray(v[:,features.index(p)]);finite=np.isfinite(y);a=y[finite]
                row=dict(source=s,basin=b,group=index[b]['seen_unseen'],parameter=p,unit=m['units'][p],days=len(y),valid=len(a),missing=int(len(y)-len(a)))
                row.update({k:None for k in ['mean','median','q1','q3','std','minimum','maximum','zero_percent']})
                if len(a):row.update(mean=float(a.mean()),median=float(np.median(a)),q1=float(np.quantile(a,.25)),q3=float(np.quantile(a,.75)),std=float(a.std(ddof=1)) if len(a)>1 else None,minimum=float(a.min()),maximum=float(a.max()),zero_percent=float((a==0).mean()*100))
                stats.append(row)
                # Min/max bins retain peaks and low values; values/hover stay actual daily values.
                if len(y)>budget:
                    chosen={0,len(y)-1}
                    for chunk in np.array_split(np.arange(len(y)),max(1,budget//3)):
                        ok=chunk[np.isfinite(y[chunk])]
                        if len(ok):chosen.update([int(ok[np.argmin(y[ok])]),int(ok[np.argmax(y[ok])])])
                        bad=chunk[~np.isfinite(y[chunk])]
                        if len(bad):chosen.add(int(bad[0]))
                    ix=np.array(sorted(chosen))
                else:ix=np.arange(len(y))
                missing=np.r_[0,np.cumsum(~finite)]
                breaks=[True]+[bool(missing[right]-missing[left+1]) for left,right in zip(ix[:-1],ix[1:])] if len(ix) else []
                series.append(dict(source=s,basin=b,parameter=p,dates=d[ix].astype('datetime64[D]').astype(str).tolist(),values=[float(x) if np.isfinite(x) else None for x in y[ix]],breaks=breaks,full_days=len(y),sampled=len(ix)<len(y)))
    return dict(series=series,statistics=stats,start=payload['start'],end=payload['end'],plot_sampling='Min/max sampling when needed; statistics use every valid daily value in the selected period.')

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(BASE),**kw)
    def do_GET(self):
        if self.path.split('?')[0]=='/api/manifest':
            m=json.loads((BASE/'manifest.json').read_text())
            for s in m['sources'].values():s.pop('basins',None)
            self.send_json(m)
        elif self.path.split('?')[0] in ['/','/index.html']:self.path='/index.html';super().do_GET()
        elif self.path.split('?')[0] in ['/app.js','/style.css']:super().do_GET()
        else:self.send_error(404)
    def send_json(self,obj):
        data=json.dumps(obj,allow_nan=False,separators=(',',':')).encode();self.send_response(200);self.send_header('Content-Type','application/json')
        if 'gzip' in self.headers.get('Accept-Encoding',''):
            data=gzip.compress(data,compresslevel=3);self.send_header('Content-Encoding','gzip')
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_POST(self):
        if self.path!='/api/query':return self.send_error(404)
        try:
            n=int(self.headers.get('Content-Length',0));assert n<300000
            self.send_json(query(json.loads(self.rfile.read(n))))
        except (ValueError,AssertionError,KeyError) as e:self.send_error(400,'Invalid selection: '+str(e))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    print('Dashboard ready at http://127.0.0.1:'+str(a.port),flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
