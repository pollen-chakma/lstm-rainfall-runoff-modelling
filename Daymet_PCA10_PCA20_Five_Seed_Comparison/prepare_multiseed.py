from pathlib import Path
import ast,copy,csv,hashlib,json,os
ROOT=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
B=ROOT/'Daymet_PCA10_PCA20_Five_Seed_Comparison'
B.mkdir(exist_ok=True)
seeds=[42,43,44,45,46]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2))
def read(p):return list(csv.DictReader(p.open()))
class Seeds(ast.NodeTransformer):
    def __init__(self,seed):self.seed=seed
    def visit_Constant(self,n):
        if type(n.value) is int and n.value==42:return ast.copy_location(ast.Constant(self.seed),n)
        if n.value=='results_seed42':return ast.copy_location(ast.Constant(f'results_seed{self.seed}'),n)
        return n
manifest=[]
for pcs in [10,20]:
    source=ROOT/f'Model A Lumped (Daymet + PCA{pcs})'
    nbpath=source/f'Daymet_PCA{pcs}_LSTM.ipynb'
    nb=json.loads(nbpath.read_text())
    assert json.loads((source/'completion.json').read_text())['status']=='complete'
    h=json.loads((source/'results_seed42/history.json').read_text());assert [x['epoch'] for x in h]==list(range(1,21))
    rows=read(source/'basin_model_performance.csv');assert len(rows)==1426
    modeldir=B/f'PCA{pcs}';modeldir.mkdir(exist_ok=True)
    ref=modeldir/'seed_42'
    if not ref.exists():ref.symlink_to(source,target_is_directory=True)
    assert ref.resolve()==source.resolve()
    baseline_hash=sha(nbpath)
    for seed in seeds:
        if seed==42:
            manifest.append(dict(pcs=pcs,seed=seed,reused=True,directory=str(ref),notebook=str(nbpath),source_notebook_sha256=baseline_hash,performance_sha256=sha(source/'basin_model_performance.csv')));continue
        dest=modeldir/f'seed_{seed}';dest.mkdir(exist_ok=True)
        inp=dest/f'daymet_pca{pcs}_discharge_1980_2009.csv'
        if not inp.exists():inp.symlink_to(source/inp.name)
        assert inp.resolve()==(source/inp.name).resolve()
        n=copy.deepcopy(nb);audit=[]
        for i,c in enumerate(n['cells']):
            if c['cell_type']=='code':
                s=''.join(c['source']);original=ast.parse(s)
                transformed=Seeds(seed).visit(ast.parse(s));ast.fix_missing_locations(transformed)
                out=ast.unparse(transformed)+'\n'
                reverse=Seeds(42)
                # Restore only intended seed changes for a strict structure comparison.
                class Back(ast.NodeTransformer):
                    def visit_Constant(self,x):
                        if type(x.value) is int and x.value==seed:return ast.copy_location(ast.Constant(42),x)
                        if x.value==f'results_seed{seed}':return ast.copy_location(ast.Constant('results_seed42'),x)
                        return x
                assert ast.dump(Back().visit(ast.parse(out)),include_attributes=False)==ast.dump(original,include_attributes=False)
                c['source']=out.splitlines(keepends=True);c['outputs']=[];c['execution_count']=None
                audit.append(dict(cell=i,only_seed_and_run_name_changed=True,source_sha256=hashlib.sha256(s.encode()).hexdigest()))
        n['cells'][0]['source']=[f'# Daymet + PCA{pcs}, seed {seed}\n', '\n', 'Fixed-budget repeatability run. Same basin IDs, years, model architecture, preprocessing and training budget as the verified seed-42 baseline. Only the random seed and result directory change. The CSV beside this notebook is the sole training data source.\n']
        target=dest/f'Daymet_PCA{pcs}_seed{seed}.ipynb'
        if target.exists():
            existing=json.loads(target.read_text());assert [c['source'] for c in existing['cells'] if c['cell_type']=='code']==[c['source'] for c in n['cells'] if c['cell_type']=='code']
        else:dump(target,n)
        dump(dest/'protocol_equivalence.json',dict(status='passed',source=str(nbpath),source_sha256=baseline_hash,seed=seed,cells=audit))
        manifest.append(dict(pcs=pcs,seed=seed,reused=False,directory=str(dest),notebook=str(target),source_notebook_sha256=baseline_hash))
dump(B/'run_manifest.json',manifest)
(B/'README.md').write_text('''# Daymet PCA10 versus PCA20: five-seed repeatability test

Seeds: 42, 43, 44, 45, 46, paired across both models. Seed 42 reuses the verified completed baseline through read-only-by-convention links; the runner never executes these linked notebooks. Eight new runs use their own folders and outputs.

Training 1980–1999; validation 2000–2004; testing 2005–2009. Same 1,326 donor and 100 unseen basin IDs. One-layer LSTM, hidden size 64, 365-day inputs, 20 epochs, 256 windows per basin per epoch, batch 128, Adam 0.001. Best validation checkpoint evaluated in both test groups. PCA10 provides 20 static mean/std inputs; PCA20 provides 40. All other settings remain unchanged.

Each new training notebook can be run independently with its linked CSV beside it and the Model A PyTorch GPU kernel. CSV links avoid duplicating the large original files. Copy the underlying CSV if moving to another computer.

run_all.py executes only incomplete new runs, one GPU process at a time, logs each run and resumes from saved epoch checkpoints. It then performs the cross-seed and ensemble analysis. Original experiment files are never overwritten. Analysis outputs distinguish variation across five training seeds from variation across basins. Five seeds do not establish universal superiority or isolate the causal contribution of heterogeneity.
''')
print('PREPARED',len(manifest),'runs;',sum(not x['reused'] for x in manifest),'new; strict protocol equivalence passed',flush=True)
