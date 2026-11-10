from pathlib import Path
import csv,json,math,html,base64
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
B=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
O=B/'analysis_outputs';O.mkdir(exist_ok=True)
SEEDS=[42,43,44,45,46]
with (B/'multiseed_basin_analysis_dataset.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==len({r['basin_id'] for r in rows})==1426
def num(x):
    try:return float(x)
    except (ValueError,TypeError):return float('nan')
def vec(rs,c):return np.array([num(r[c]) for r in rs])
def write(name,rs):
    with (O/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def desc(a):
    a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()) if len(a) else np.nan,median=float(np.median(a)) if len(a) else np.nan,q1=float(np.quantile(a,.25)) if len(a) else np.nan,q3=float(np.quantile(a,.75)) if len(a) else np.nan,negative=int(sum(a<0)))
summary=[];ensemble=[];repeatability=[];paired=[];consistency=[];subgroup=[]
for g in ['Seen','Unseen']:
    rs=[r for r in rows if r['seen_unseen']==g]
    for metric in ['NSE','KGE']:
        for pcs in [10,20]:
            group_seed=[]
            for seed in SEEDS:
                s=dict(group=g,metric=metric,model=f'PCA{pcs}',seed=seed,**desc(vec(rs,f'PCA{pcs}_seed{seed}_{metric}')))
                summary.append(s);group_seed.append(s)
            med=np.array([s['median'] for s in group_seed]);means=np.array([s['mean'] for s in group_seed]);neg=np.array([s['negative'] for s in group_seed])
            repeatability.append(dict(group=g,metric=metric,model=f'PCA{pcs}',seeds=5,mean_of_seed_medians=float(med.mean()),std_of_seed_medians=float(med.std(ddof=1)),minimum_seed_median=float(med.min()),maximum_seed_median=float(med.max()),mean_of_seed_means=float(means.mean()),mean_negative_basin_count=float(neg.mean()),minimum_negative_count=int(neg.min()),maximum_negative_count=int(neg.max())))
            ensemble.append(dict(group=g,metric=metric,model=f'PCA{pcs}',**desc(vec(rs,f'PCA{pcs}_ensemble_{metric}'))))
        a=np.array([s['median'] for s in summary if s['group']==g and s['metric']==metric and s['model']=='PCA10'])
        b=np.array([s['median'] for s in summary if s['group']==g and s['metric']==metric and s['model']=='PCA20'])
        d=b-a
        pvalue=float(wilcoxon(d,method='exact').pvalue) if (d!=0).all() else (float(wilcoxon(d,method='approx').pvalue) if np.any(d) else 1.)
        paired.append(dict(group=g,metric=metric,paired_seeds=5,PCA20_seed_median_wins=int(sum(d>0)),PCA10_seed_median_wins=int(sum(d<0)),mean_change_in_seed_median=float(d.mean()),minimum_change=float(d.min()),maximum_change=float(d.max()),wilcoxon_two_sided_p=pvalue,holm_adjusted_p=np.nan))
order=np.argsort([r['wilcoxon_two_sided_p'] for r in paired]);previous=0
for rank,i in enumerate(order):previous=max(previous,min(1.,paired[i]['wilcoxon_two_sided_p']*(len(paired)-rank)));paired[i]['holm_adjusted_p']=previous
# Descriptive subgroup definitions are fixed before inspecting new seed results.
reference=[r for r in rows if r['seen_unseen']=='Seen' and num(r['scored_coverage'])>=.8 and math.isfinite(num(r['test_q_mean_mm_day']))]
low_threshold=float(np.quantile(vec(reference,'test_q_mean_mm_day'),.25))
def classes(r):
    adequate=num(r['scored_coverage'])>=.8
    out=['All scored']
    if not adequate:return out
    out+=['Adequate coverage']
    q=num(r['test_q_mean_mm_day']);zero=num(r['test_zero_flow_fraction']);arid=num(r['test_aridity_pet_over_prcp'])
    if math.isfinite(q):out+=['Low flow' if q<=low_threshold else 'Other flow']
    if math.isfinite(zero):out+=['Intermittent' if zero>=.01 else 'Less than 1% zero flow']
    if math.isfinite(arid):out+=['Dry' if arid>1 else 'Humid or energy limited']
    return out
thresholds=dict(low_flow_definition='Observed test-period mean discharge at or below donor reference 25th percentile',low_flow_threshold_mm_day=low_threshold,low_flow_reference_n=len(reference),intermittent_definition='At least 1% of available test observations equal zero',dry_definition='Test-period PET/precipitation > 1',coverage_requirement='At least 80% scored test-day coverage for hydrological subgroups',interpretation='Descriptive, overlapping test-period groups. Not model predictors or evidence of causality. Groups were specified before reading new seed metrics.')
(O/'subgroup_definitions.json').write_text(json.dumps(thresholds,indent=2))
for r in rows:
    for metric in ['NSE','KGE']:
        a=np.array([num(r[f'PCA10_seed{s}_{metric}']) for s in SEEDS]);b=np.array([num(r[f'PCA20_seed{s}_{metric}']) for s in SEEDS]);valid=np.isfinite(a)&np.isfinite(b);d=b[valid]-a[valid]
        consistency.append(dict(basin_id=r['basin_id'],seen_unseen=r['seen_unseen'],metric=metric,valid_seed_pairs=int(sum(valid)),PCA20_wins=int(sum(d>0)),PCA10_wins=int(sum(d<0)),ties=int(sum(d==0)),PCA20_wins_all_five=bool(len(d)==5 and (d>0).all()),PCA20_wins_at_least_four=bool(len(d)==5 and sum(d>0)>=4),mean_paired_change=float(d.mean()) if len(d) else np.nan,median_paired_change=float(np.median(d)) if len(d) else np.nan,PCA10_negative_seeds=int(sum(a[valid]<0)),PCA20_negative_seeds=int(sum(b[valid]<0)),PCA10_ensemble=num(r[f'PCA10_ensemble_{metric}']),PCA20_ensemble=num(r[f'PCA20_ensemble_{metric}']),hydrological_groups='; '.join(classes(r))))
groupnames=['All scored','Adequate coverage','Low flow','Other flow','Intermittent','Less than 1% zero flow','Dry','Humid or energy limited']
for g in ['Seen','Unseen']:
    for name in groupnames:
        rs=[r for r in rows if r['seen_unseen']==g and name in classes(r)]
        for metric in ['NSE','KGE']:
            for seed in SEEDS+['ensemble']:
                tail=f'seed{seed}' if seed!='ensemble' else 'ensemble'
                a=vec(rs,f'PCA10_{tail}_{metric}');b=vec(rs,f'PCA20_{tail}_{metric}');valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid];d=b-a
                subgroup.append(dict(group=g,hydrological_group=name,metric=metric,seed=seed,n=len(a),PCA10_median=float(np.median(a)) if len(a) else np.nan,PCA20_median=float(np.median(b)) if len(b) else np.nan,median_basin_change=float(np.median(d)) if len(d) else np.nan,mean_basin_change=float(d.mean()) if len(d) else np.nan,PCA20_improved_basins=int(sum(d>0)),PCA20_improved_percent=float(100*np.mean(d>0)) if len(d) else np.nan,PCA10_negative=int(sum(a<0)),PCA20_negative=int(sum(b<0))))
write('per_seed_group_performance.csv',summary);write('repeatability_summary.csv',repeatability);write('paired_seed_comparison.csv',paired);write('basin_improvement_consistency.csv',consistency);write('ensemble_performance_summary.csv',ensemble);write('hydrological_subgroup_comparison.csv',subgroup)
distribution=[]
for g in ['Seen','Unseen']:
    for metric in ['NSE','KGE']:
        cr=[r for r in consistency if r['seen_unseen']==g and r['metric']==metric and r['valid_seed_pairs']==5]
        for wins in range(6):distribution.append(dict(group=g,metric=metric,PCA20_wins_out_of_five=wins,basins=sum(r['PCA20_wins']==wins for r in cr)))
write('consistency_distribution.csv',distribution)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
for metric in ['NSE','KGE']:
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,g in zip(axes,['Seen','Unseen']):
        for pcs,color in [(10,'#3569a8'),(20,'#dc8a36')]:
            a=[s['median'] for s in summary if s['group']==g and s['metric']==metric and s['model']==f'PCA{pcs}']
            ax.plot(SEEDS,a,'o-',label=f'PCA{pcs}',color=color,lw=1.7)
            en=next(s['median'] for s in ensemble if s['group']==g and s['metric']==metric and s['model']==f'PCA{pcs}')
            ax.axhline(en,color=color,ls='--',alpha=.7,label=f'PCA{pcs} ensemble')
        ax.set(title=g,xlabel='Training seed');ax.set_xticks(SEEDS);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel(f'Median basin {metric}');axes[1].legend(fontsize=8)
    fig.suptitle(f'{metric}: repeatability across five paired seeds',fontweight='bold');fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(O/f'{metric}_five_seed_repeatability.{ext}',dpi=250,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,g in zip(axes,['Seen','Unseen']):
        ds=[r for r in distribution if r['group']==g and r['metric']==metric]
        ax.bar(range(6),[r['basins'] for r in ds],color='#3569a8');ax.set(title=g,xlabel='Seeds in which PCA20 beats PCA10',ylabel='Basins',xticks=range(6));ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'{metric}: consistency of basin-level improvements',fontweight='bold');fig.tight_layout()
    fig.savefig(O/f'{metric}_basin_consistency.png',dpi=250,bbox_inches='tight');plt.close(fig)
def table(rs):
    ks=list(rs[0]);return '<table><tr>'+''.join('<th>'+html.escape(k.replace('_',' '))+'</th>' for k in ks)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in ks)+'</tr>' for r in rs)+'</table>'
report='<html><head><meta charset="utf-8"><style>body{font:15px sans-serif;margin:24px;max-width:1400px}img{max-width:100%}td,th{border:1px solid #ddd;padding:7px}table{border-collapse:collapse;font-size:13px;width:100%}th{background:#eef2f6;overflow-wrap:anywhere}</style></head><body><h1>Five-seed Daymet PCA10 versus PCA20</h1><p>Seeds 42–46. Training 1980–1999, validation 2000–2004, testing 2005–2009. Same 1,326 seen and 100 unseen basin IDs, fixed 20-epoch budget, 64-unit LSTM. Seed 42 reused; eight new runs completed.</p><h2>Repeatability</h2>'+table(repeatability)+'<h2>Paired seed comparisons</h2>'+table(paired)+'<p>Five paired seeds give limited statistical power. The Wilcoxon tests are exploratory; Holm adjustment covers the four group/metric comparisons. Basins are not five independent replications per seed.</p><h2>Five-model discharge ensembles</h2>'+table(ensemble)
for metric in ['NSE','KGE']:
    for suffix in ['five_seed_repeatability','basin_consistency']:
        f=O/f'{metric}_{suffix}.png';report+='<img alt="'+metric+' '+suffix+'" src="data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode()+'">'
report+='<h2>Hydrological subgroups: ensembles</h2>'+table([r for r in subgroup if r['seed']=='ensemble'])+'<p>'+html.escape(json.dumps(thresholds))+'</p><p>The original window sampler uses seed + epoch. Adjacent seeds shift overlapping epoch-level window schedules. This experiment measures repeatability of the original training pipeline, including initialization changes; it does not create five independent hydrological datasets.</p></body></html>'
(O/'five_seed_report.html').write_text(report)
print('REPEATABILITY',json.dumps(repeatability),flush=True);print('PAIRED',json.dumps(paired),flush=True);print('ENSEMBLES',json.dumps(ensemble),flush=True)
try:
    from IPython.display import display,HTML
    display(HTML('<h2>Five-seed repeatability</h2>'+table(repeatability)+'<h2>Ensemble performance</h2>'+table(ensemble)))
except ImportError:pass
print('All comparison tables and figures reproduced from the single basin CSV.',flush=True)
