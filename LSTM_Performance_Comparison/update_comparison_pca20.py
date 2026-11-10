from pathlib import Path
import csv, json, hashlib, shutil, datetime, math, base64, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
B=ROOT/'LSTM_Performance_Comparison'
P=B/'basin_lstm_performance_comparison.csv'
S=ROOT/'Model A Lumped (Daymet + PCA20)'/'basin_model_performance.csv'
O=B/'PCA20_updated_boxplots'; O.mkdir(exist_ok=True)
def read(p):
    with p.open(newline='') as f:
        r=csv.DictReader(f); return r.fieldnames,list(r)
def write(p,rows,fields=None):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
fields,old=read(P); sf,src=read(S)
assert len(old)==len(src)==1426
assert len({r['basin_id'] for r in old})==len({r['basin_id'] for r in src})==1426
index={r['basin_id']:r for r in src}; assert set(index)=={r['basin_id'] for r in old}
prefix='Daymet_PCA20_Test2005_2009_'
mapping={'NSE':'NSE','KGE':'KGE','RMSE':'RMSE','MAE':'MAE','Bias_percent':'bias_percent','N_scored_days':'n','Status':'status'}
newfields=[prefix+x for x in mapping]
merged=[]
for r in old:
    s=index[r['basin_id']]
    assert r['seen_unseen']==s['seen_unseen'] and r['catchment_split']==s['catchment_split']
    assert s['test_start']=='2005-01-01' and s['test_end']=='2009-12-31'
    assert r['Daymet_PCA10_Test2005_2009_N_scored_days']==s['n']
    merged.append(dict(r))
    merged[-1].update({prefix+k:s[v] for k,v in mapping.items()})
backup=B/'backups'/('basin_comparison_before_PCA20_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.csv')
backup.parent.mkdir(exist_ok=True); shutil.copy2(P,backup)
allfields=fields+[c for c in newfields if c not in fields]
tmp=P.with_suffix('.csv.tmp');write(tmp,merged,allfields)
ff,rr=read(tmp)
assert ff==allfields and len(rr)==1426
for before,after in zip(old,rr):
    assert all(before[k]==after[k] for k in fields if k not in newfields)
    assert all(after[prefix+k]==index[before['basin_id']][v] for k,v in mapping.items())
tmp.replace(P)
audit=dict(status='passed',rows=1426,columns=len(allfields),seen=sum(r['seen_unseen']=='Seen' for r in rr),unseen=sum(r['seen_unseen']=='Unseen' for r in rr),old_values_and_order_preserved=True,all_PCA20_values_match_source=True,backup=str(backup),sha256=hashlib.sha256(P.read_bytes()).hexdigest())
(O/'merge_verification.json').write_text(json.dumps(audit,indent=2))

models=[('Daymet meteorology only','Daymet_MetOnly_Test2005_2009_','2005–2009','Daymet\nForcing only\n2005–2009'),('Daymet + original attributes','Daymet_MetAttr_Test2005_2009_','2005–2009','Daymet\n+ original attributes\n2005–2009'),('Daymet + PCA10 mean/std','Daymet_PCA10_Test2005_2009_','2005–2009','Daymet\n+ PCA10 mean/std\n2005–2009'),('Daymet + PCA20 mean/std',prefix,'2005–2009','Daymet\n+ PCA20 mean/std\n2005–2009'),('ERA5 + original attributes','ERA5_MetAttr_Test2005_2009_','2005–2009','ERA5\n+ original attributes\n2005–2009'),('ERA5 extended + attributes','ERA5_MetAttr_Test2015_2023_','2015–early 2023','ERA5 extended\n+ original attributes\n2015–early 2023')]
def val(r,c):
    try: return float(r[c])
    except (ValueError,TypeError):return float('nan')
def array(group,p,m):return np.array([val(r,p+m) for r in rr if r['seen_unseen']==group and math.isfinite(val(r,p+m))])
colors={'Seen':'#3569a8','Unseen':'#dc8a36'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':14,'svg.fonttype':'none','pdf.fonttype':42})
stats=[]
for metric in ['NSE','KGE']:
    fig,ax=plt.subplots(figsize=(15.5,7.8));fig.subplots_adjust(left=.07,right=.985,bottom=.25,top=.80)
    for i,(label,p,period,_) in enumerate(models):
        for group,offset in [('Seen',-.19),('Unseen',.19)]:
            a=array(group,p,metric);assert len(a)>0
            raw=boxplot_stats(a,whis=1.5)[0];d=dict(raw);d['fliers']=raw['fliers'][raw['fliers']>=-1]
            x=i+1+offset;c=colors[group]
            ax.bxp([d],positions=[x],widths=.30,shownotches=True,patch_artist=True,showmeans=True,manage_ticks=False,boxprops=dict(facecolor=c,edgecolor=c,alpha=.27,linewidth=1.6),whiskerprops=dict(color=c,linewidth=1.4),capprops=dict(color=c,linewidth=1.4),medianprops=dict(color='#c7372e',linewidth=1.8),meanprops=dict(marker='D',markerfacecolor='#218842',markeredgecolor='white',markersize=5.5),flierprops=dict(marker='+',markeredgecolor=c,markersize=5,alpha=.55))
            below=int(sum(a<-1))
            if below:ax.scatter([x],[-1.015],marker='v',color=c,s=32,zorder=5);ax.text(x,-1.105,str(below),ha='center',fontsize=9,color=c)
            ax.text(x,1.035,f'n={len(a):,}',ha='center',fontsize=9,color=c)
            stats.append(dict(model=label,test_period=period,group=group,metric=metric,n=len(a),mean=float(a.mean()),median=float(np.median(a)),q1=float(np.quantile(a,.25)),q3=float(np.quantile(a,.75)),std=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max()),negative=int(sum(a<0)),below_minus_one=below))
    ax.axvspan(5.5,6.5,color='#f2f2f2',zorder=-3);ax.axvline(5.5,color='#999999',ls='--',lw=1)
    ax.set(xlim=(.45,6.55),ylim=(-1.14,1.10),ylabel=metric);ax.set_yticks(np.arange(-1,1.01,.5));ax.set_xticks(np.arange(1,7),[x[3] for x in models]);ax.tick_params(axis='x',length=0,pad=13,labelsize=10.5)
    ax.grid(axis='y',color='#d7dce2',lw=.8);ax.axhline(0,color='#8a939c',lw=1);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'{metric} performance across LSTM experiments',x=.07,y=.965,ha='left',fontweight='bold',fontsize=19)
    fig.text(.07,.916,'Seen and unseen catchments • basin-level test scores • higher is better',color='#505963',fontsize=12)
    handles=[Patch(facecolor=c,edgecolor=c,alpha=.4,label=g) for g,c in colors.items()]+[Line2D([],[],color='#c7372e',lw=1.8,label='Median'),Line2D([],[],color='#218842',marker='D',linestyle='None',markersize=5,label='Mean')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.065,.89),frameon=False,ncol=4)
    fig.text(.07,.075,'Boxes: middle 50%; notches: approximate median intervals; whiskers: 1.5 × IQR; +: outliers.',fontsize=10,color='#505963')
    fig.text(.07,.045,'▼ at −1 aggregates lower scores; numbers below are counts. All statistics use uncapped scores.',fontsize=10,color='#505963')
    fig.text(.07,.015,'Shaded ERA5 extended setup uses a different test period and is not a controlled same-period comparison.',fontsize=10,color='#505963')
    for ext in ['png','pdf','svg']:fig.savefig(O/f'{metric}_seen_unseen_boxplot_with_PCA20.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
write(O/'model_summary_statistics.csv',stats)
# Same-period, common-basin medians prevent rankings caused by different missing scores.
common=[];paired=[]
for group in colors:
    for metric in ['NSE','KGE']:
        rows=[r for r in rr if r['seen_unseen']==group and all(math.isfinite(val(r,p+metric)) for _,p,_,_ in models[:5])]
        for label,p,period,_ in models[:5]:
            a=np.array([val(r,p+metric) for r in rows]);common.append(dict(group=group,metric=metric,model=label,n=len(a),mean=float(a.mean()),median=float(np.median(a))))
        a=np.array([val(r,models[2][1]+metric) for r in rows]);b=np.array([val(r,prefix+metric) for r in rows]);d=b-a
        paired.append(dict(group=group,metric=metric,n=len(a),improved=int(sum(d>0)),worsened=int(sum(d<0)),tied=int(sum(d==0)),median_paired_change=float(np.median(d)),mean_paired_change=float(d.mean())))
write(O/'same_period_common_basin_summary.csv',common);write(O/'PCA20_vs_PCA10_paired_summary.csv',paired)
def table(rows):
    keys=list(rows[0]);return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
report='<html><head><meta charset="utf-8"><style>body{font:16px sans-serif;margin:24px}img{width:100%}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}th{background:#eef2f6}</style></head><body><h1>LSTM comparisons including Daymet + PCA20</h1><p>CSV updated: 1,426 basins, 40 columns. Existing values and basin order preserved. 1,326 seen and 100 unseen basins.</p>'
for metric in ['NSE','KGE']:
    report+='<h2>'+metric+'</h2><img alt="'+metric+' seen and unseen boxplot" src="data:image/png;base64,'+base64.b64encode((O/f'{metric}_seen_unseen_boxplot_with_PCA20.png').read_bytes()).decode()+'">'
report+='<h2>Summary statistics</h2>'+table(stats)+'<h2>Same-period common-basin comparisons</h2>'+table(common)+'<h2>PCA20 versus PCA10 paired changes</h2>'+table(paired)+'<p>Single seed, fixed budget. Descriptive comparisons do not establish statistical superiority. ERA5 extended uses different training and testing years. NSE and KGE remain uncapped in CSVs.</p></body></html>'
(O/'comparison_report.html').write_text(report)
print(json.dumps(audit),flush=True)
for group in colors:
    for label,p,period,_ in models:
        s=[r for r in stats if r['group']==group and r['model']==label]
        print(group,label,[(r['metric'],round(r['mean'],4),round(r['median'],4),r['negative'],r['n']) for r in s],flush=True)
print('PAIRED',json.dumps(paired),flush=True)
print('COMPLETE',str(O),flush=True)
