"""Generate manuscript numbers, comparable tables, plots and metric rechecks."""
from common4 import *
from figure_style import journal_layout
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

G=ROOT/'paper/generated';F=ROOT/'paper/figures';A=ROOT/'results/article';G.mkdir(exist_ok=True);F.mkdir(exist_ok=True);A.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.dpi':600})
names={'uniform':'Uniform','endpoint':'Endpoint excess','guard_dense':'Guard B-2','guard_half':'Guard half'}
models={'ResidualTCN':'Residual TCN','IRM':'IEM','unconditional':'Full-state tree','hold_fallback':'Hold + fallback'}

def table(path,caption,header,rows,align=None,label=None,long=False):
    align=align or 'l'+'r'*(len(header)-1)
    if long:
        lines=['\\begingroup\\footnotesize','\\setlength{\\tabcolsep}{3pt}','\\begin{longtable}{'+align+'}','\\caption{'+caption+'}'+(('\\label{'+label+'}') if label else '')+'\\\\','\\toprule',' & '.join(header)+'\\\\\\midrule','\\endfirsthead','\\toprule',' & '.join(header)+'\\\\\\midrule','\\endhead','\\bottomrule\\endfoot','\\bottomrule\\endlastfoot']
    else:
        lines=['\\begin{table}[htbp]\\centering\\small','\\setlength{\\tabcolsep}{3pt}','\\caption{'+caption+'}'+(('\\label{'+label+'}') if label else ''),'\\begin{tabular}{'+align+'}\\toprule',' & '.join(header)+'\\\\\\midrule']
    lines+=[' & '.join(map(str,row))+'\\\\' for row in rows]
    lines+=['\\end{longtable}','\\endgroup'] if long else ['\\bottomrule\\end{tabular}','\\end{table}']
    (G/path).write_text('\n'.join(lines)+'\n',encoding='utf-8')

def pair_interval(frame,one,two,column,factor=1):
    keys=['flight_id','source_id','platform'];p=frame[frame.policy.eq(one)].set_index(keys)[column];q=frame[frame.policy.eq(two)].set_index(keys)[column]
    z=(p-q).rename('difference').reset_index();assert not z.difference.isna().any();s=z.groupby(['platform','source_id']).difference.mean();rng=np.random.default_rng(2026102704);boot=np.zeros(5000)
    for platform in s.index.get_level_values(0).unique():
        v=s.loc[platform].to_numpy();boot+=v[rng.integers(0,len(v),size=(5000,len(v)))].mean(axis=1)/s.index.get_level_values(0).nunique()
    return dict(estimate=macro(z,'difference')*factor,low=float(np.quantile(boot,.025))*factor,high=float(np.quantile(boot,.975))*factor,sources=len(s))

def calibration():
    ca=ROOT/'results/new_calibration';cov=pd.read_csv(ca/'source_coverage.csv');c=pd.read_csv(ca/'calibration.csv');m=pd.read_csv(ca/'record_metrics.csv');rows=[];recalc=[];maxdiff=0.
    for seed in [42,137,2026,815,991]:
        d=pd.read_parquet(ca/f'curves_{seed}.parquet')
        for key,z in d.groupby(['model','scenario','flight_id'],sort=False):
            model,scenario,fid=key;e=z.predicted_prefix_Wh.to_numpy();truth=z.actual_prefix_Wh.to_numpy();score=float(np.max(abs(e-truth)/(e+1)));old=m[m.seed.eq(seed)&m.model.eq(model)&m.scenario.eq(scenario)&m.flight_id.eq(fid)].iloc[0];maxdiff=max(maxdiff,abs(score-old.score));assert maxdiff<1e-12
            q=float(c[c.seed.eq(seed)&c.model.eq(model)].q.iloc[0]);lo=np.maximum(0,e-q*(e+1));hi=e+q*(e+1)
            recalc.append(dict(seed=seed,model=model,scenario=scenario,flight_id=fid,source_id=z.source_id.iloc[0],role=z.role.iloc[0],score=score,mean_width_Wh=float(np.mean(hi-lo)),whole_curve_covered=bool(np.all((truth>=lo-1e-12)&(truth<=hi+1e-12)))))
    f=pd.DataFrame(recalc);f.to_csv(A/'calibration_recomputed.csv',index=False)
    detail=[]
    for (seed,model),z in cov[cov.role.eq('test')].groupby(['seed','model']):
        q=c[c.seed.eq(seed)&c.model.eq(model)].q.iloc[0]
        for scenario in ['original','burst','missing']:
            w=f[f.seed.eq(seed)&f.model.eq(model)&f.scenario.eq(scenario)&f.role.eq('test')].groupby('source_id').mean_width_Wh.mean().mean();t=m[m.seed.eq(seed)&m.model.eq(model)&m.scenario.eq(scenario)&m.role.eq('test')].groupby('source_id')[['RMSLE','peak_prefix_Wh']].mean().mean()
            detail.append(dict(seed=seed,model=model,scenario=scenario,q=q,source_coverage=float(z.whole_family_covered.mean()),width=w,**t.to_dict()))
    detail=pd.DataFrame(detail);detail.to_csv(A/'calibration_detail.csv',index=False)
    for model in ['unconditional','hold_fallback']:
        for scenario in ['original','burst','missing']:
            z=detail[detail.model.eq(model)&detail.scenario.eq(scenario)].mean(numeric_only=True)
            rows.append([models[model],{'original':'Original','burst':'30 s outages','missing':'All wind absent'}[scenario],f'{z.RMSLE:.3f}',f'{z.peak_prefix_Wh:.2f}',f'{z.width:.2f}',f'{100*z.source_coverage:.0f}'])
    table('calibration_main.tex','New randomized-source experiments. Means weight test recordings equally within split, then the five splits equally. Widths use the source-macro mean over prefixes and segments. Coverage protects the complete three-view source family, so it repeats across scenario rows. These 50 test source--split incidences reuse 38 observed recordings.',['Estimator','Input view','RMSLE','Peak error','Width','Coverage \\%'],rows,label='tab:newcal')
    rows=[]
    for seed in [42,137,2026,815,991]:
        for model in ['unconditional','hold_fallback']:
            for scenario in ['original','burst','missing']:
                z=detail[detail.seed.eq(seed)&detail.model.eq(model)&detail.scenario.eq(scenario)].iloc[0]
                rows.append([seed,models[model],{'original':'Original','burst':'Outages','missing':'Missing'}[scenario],f'{z.q:.3f}',f'{100*z.source_coverage:.0f}',f'{z.RMSLE:.3f}',f'{z.width:.2f}'])
    table('full_calibration.tex','All five new source splits and both fixed estimators. Each split has ten calibration and ten test sources. Coverage is whole-source, all-view coverage; width is in Wh. All ten exact random-design expected coverages are 90.91\\%.',['Seed','Estimator','View','$q$','Coverage \\%','RMSLE','Width'],rows,long=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5));x=np.arange(3)
    for model,color,marker in [('unconditional','#9c3b34','o'),('hold_fallback','#176f78','s')]:
        z=detail[detail.model.eq(model)].groupby('scenario').mean(numeric_only=True).reindex(['original','burst','missing']);axes[0].plot(x,z.RMSLE,marker=marker,color=color,label=models[model]);
        z=detail[detail.model.eq(model)&detail.scenario.eq('original')].set_index('seed').loc[[42,137,2026,815,991]];axes[1].plot(np.arange(5),100*z.source_coverage,marker=marker,color=color,label=models[model])
    axes[0].set(xticks=x,xticklabels=['Original','30 s outages','All wind absent'],ylabel='Source-macro RMSLE');axes[0].legend(fontsize=8)
    axes[1].axhline(1000/11,color='#666666',ls='--',lw=1,label='Design expectation: 90.91%');axes[1].set(xticks=np.arange(5),xticklabels=['42','137','2026','815','991'],xlabel='Random-assignment seed',ylabel='Realized whole-source coverage (%)',ylim=(65,103));axes[1].legend(fontsize=7,loc='lower left')
    journal_layout(fig);fig.tight_layout();fig.savefig(F/'random_calibration.png');plt.close(fig)
    return dict(max_score_replay_error=maxdiff,observed_family_coverage={model:float(cov[cov.model.eq(model)&cov.role.eq('test')].whole_family_covered.mean()) for model in models if model in cov.model.unique()},split_coverage=detail[detail.scenario.eq('original')][['seed','model','source_coverage','width','q']].to_dict('records'))

def workflow():
    fig,ax=plt.subplots(figsize=(10,3.6));ax.set(xlim=(0,10),ylim=(0,3.6));ax.axis('off')
    nodes=[(.15,2.05,'State or interpolation','Shape prior; no hidden\nreference values'),(3.55,2.05,'Sparse requests','Uniform primary;\nguarded controls'),(6.95,2.05,'Returned readings','Units, reading bounds,\ntime bounds'),(3.55,.2,'Sharp recovery class','One counter and\none observation clock'),(6.95,.2,'Clipped minimax center','Admissible origin;\nunchanged radius')]
    for x,y,title,subtitle in nodes:
        ax.add_patch(FancyBboxPatch((x,y),2.85,1.15,boxstyle='round,pad=0.08',facecolor='#eaf2f3',edgecolor='#225d68',lw=1.2));ax.text(x+1.425,y+.78,title,ha='center',va='center',weight='bold',fontsize=10);ax.text(x+1.425,y+.34,subtitle,ha='center',va='center',fontsize=8,wrap=True)
    for p,q in [((3.08,2.6),(3.44,2.6)),((6.48,2.6),(6.84,2.6)),((8.35,1.95),(5.65,1.45)),((6.48,.78),(6.84,.78))]:ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=12,color='#333333',lw=1.3))
    ax.text(.15,.85,'The prior guides;\nreadings define the\nadmissible class.',fontsize=10,color='#333333')
    journal_layout(fig);fig.tight_layout(pad=.2);fig.savefig(F/'integrated_workflow.png');plt.close(fig)

def query():
    f=pd.read_csv(ROOT/'results/new_query/query_cases.csv');s=pd.read_csv(ROOT/'results/new_query/query_summary.csv');rows=[];clean=s[s.budget.eq(8)&s.cap_scenario.eq('q0.9')&s.epsilon_Wh.eq(0)]
    for h in [0,.1,.5]:
        for policy in names:
            z=clean[clean.halfwidth_s.eq(h)&clean.policy.eq(policy)].iloc[0];u=clean[clean.halfwidth_s.eq(h)&clean.policy.eq('uniform')].iloc[0]
            rows.append([f'{h:g}',names[policy],f'{100*z.missed:.2f}',f'{100*(u.missed-z.missed):.2f}',f'{z.volume_Wh_s:.1f}',f'{z.radius_Wh:.3f}',f'{z.midpoint_RMSE_Wh:.3f}'])
    table('query_main.tex','New guarded-query comparison, B8/Q90 and clean counters. All summaries weight sources within platform and the three labels equally. Gain is reduction in missed burden against uniform under the same clock boxes. Recovery uses a separate 2000 W cap.',['$h$, s','Policy','Missed \\%','Gain, pp','$J$, Wh s','$R$, Wh','RMSE, Wh'],rows,label='tab:newquery')
    primary=f[f.budget.eq(8)&f.cap_scenario.eq('q0.9')&f.epsilon_Wh.eq(0)&f.halfwidth_s.eq(.5)];contrast={c:pair_interval(primary,'guard_dense','endpoint',c,100 if c=='missed' else 1) for c in ['missed','volume_Wh_s','radius_Wh']};write(A/'paired_query_contrasts.json',contrast)
    rows=[]
    for key,z in s.groupby(['budget','cap_scenario','halfwidth_s','epsilon_Wh','mechanism'],sort=True):
        B,cap,h,eps,kind=key;cells=[B,{'q0.9':'Q90','q0.999':'Q99.9'}[cap],f'{h:g}',('clean' if eps==0 else f'{eps:g}/{kind}')]
        for policy in names:
            a=z[z.policy.eq(policy)].iloc[0];cells.append(f'{100*a.missed:.1f}/{a.volume_Wh_s:.0f}')
        rows.append(cells)
    table('full_query.tex','Every registered query condition: each policy cell is missed burden in percent / continuous recovery volume in Wh s. Provider names give error halfwidth in Wh and mechanism. All 440 records remain; no condition is selected by its outcome. Radius, midpoint error and actual maximum gaps for the identical conditions are retained in the CSV summaries.',['$B$','Cap','$h$','Provider','Uniform','Excess','Guard B-2','Guard half'],rows,long=True)
    rows=[]
    for h in [0,.1,.5]:
        for policy in names:
            z=clean[clean.halfwidth_s.eq(h)&clean.policy.eq(policy)].iloc[0];rows.append([f'{h:g}',names[policy],f'{z.maximum_gap_s:.2f}',f'{z.radius_Wh:.3f}',f'{z.radius_bound_Wh:.3f}'])
    table('radius_design.tex','Actual mean maximum query gap and mean sharp and sufficient radii at B8/Q90, clean counters. The sufficient bound is evaluated per record before averaging; it is not the sharp radius or a calibrated sensor error.',['$h$, s','Policy','$\\Delta$, s','$R$, Wh','Bound, Wh'],rows,label='tab:radiusdesign')
    fig,ax=plt.subplots(1,2,figsize=(10,3.6));colors=['#576670','#a5522a','#176f78','#8c629d'];markers=['o','s','D','^']
    for policy,color,marker in zip(names,colors,markers):
        z=clean[clean.policy.eq(policy)].sort_values('halfwidth_s');ax[0].plot(z.halfwidth_s,100*z.missed,marker=marker,color=color,label=names[policy]);ax[1].plot(z.volume_Wh_s,z.midpoint_RMSE_Wh,marker=marker,color=color,label=names[policy])
    ax[0].set(xlabel='Clock halfwidth (s)',ylabel='Missed burden (%)',xticks=[0,.1,.5]);ax[0].legend(fontsize=8);ax[1].set(xlabel='Recovery volume (Wh s)',ylabel='Midpoint RMSE (Wh)');journal_layout(fig);fig.tight_layout();fig.savefig(F/'guarded_query.png');plt.close(fig)
    return dict(contrasts=contrast,registered_histories=json.loads((ROOT/'results/new_query/query_gate.json').read_text()),main=clean.to_dict('records'))

def projection():
    s=pd.read_csv(ROOT/'results/new_projection/projection_summary.csv');t=pd.read_csv(ROOT/'results/new_projection/projection_runtime.csv');rows=[]
    selected=s[s.policy.eq('guard_dense')&s.halfwidth_s.eq(.5)]
    for eps in [0,.1]:
        for model in ['ResidualTCN','IRM']:
            for view in ['original','wind_missing']:
                z=selected[selected.epsilon_Wh.eq(eps)&selected.model.eq(model)&selected.input_scenario.eq(view)].set_index('weight');a=z.loc[0]
                rows.append([f'{eps:g}',models[model],('Original' if view=='original' else 'No wind'),f'{a.prior_RMSE_Wh:.3f}',f'{z.loc[0,"RMSE_Wh"]:.3f}',f'{z.loc[.5,"RMSE_Wh"]:.3f}',f'{z.loc[1,"RMSE_Wh"]:.3f}',f'{a.radius_Wh:.3f}'])
    table('projection_main.tex','Feasible learned recovery on identical guarded B8/Q90 readings at $h=0.5$ s. Means weight recordings within each IDF configuration and the two labels equally. Columns compare the raw state prior, midpoint, half blend and full projected prior. The conditional worst-case ceilings are $R$, $1.5R$ and $2R$, respectively. All errors and radii are in Wh.',['$\\epsilon$','Prior','Input','Raw','Midpoint','Half','Projected','$R$'],rows,label='tab:newprojection')
    rows=[]
    for key,z in s.groupby(['policy','epsilon_Wh','halfwidth_s','model','input_scenario'],sort=True):
        policy,eps,h,model,view=key;a=z.set_index('weight');r=a.loc[0]
        rows.append([names[policy],f'{eps:g}',f'{h:g}',models[model],('Original' if view=='original' else 'No wind'),f'{r.prior_RMSE_Wh:.3f}',f'{a.loc[0,"RMSE_Wh"]:.3f}',f'{a.loc[.5,"RMSE_Wh"]:.3f}',f'{a.loc[1,"RMSE_Wh"]:.3f}',f'{r.radius_Wh:.3f}'])
    table('full_projection.tex','Every projection condition and all fixed blend weights. All RMSEs and radii are in Wh. The shared-origin LP and original constraints are checked for each prior/view/reading history. Weights are fixed rather than selected by the best observed result.',['Policy','$\\epsilon$','$h$','Prior','Input','Raw','Mid','Half','Proj.','$R$'],rows,long=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,eps in zip(axes,[0,.1]):
        for model,color in [('ResidualTCN','#176f78'),('IRM','#a5522a')]:
            for view,ls in [('original','-'),('wind_missing','--')]:
                z=selected[selected.epsilon_Wh.eq(eps)&selected.model.eq(model)&selected.input_scenario.eq(view)].sort_values('weight');ax.plot(z.weight,z.RMSE_Wh,ls=ls,marker='o',color=color,label=models[model]+(' / no wind' if view=='wind_missing' else ' / original'))
        ax.set(xlabel='Learned blend weight',ylabel='Source-macro curve RMSE (Wh)',xticks=[0,.5,1],title=f'Reading halfwidth: {eps:g} Wh')
    axes[0].legend(fontsize=7);journal_layout(fig);fig.tight_layout();fig.savefig(F/'feasible_projection.png');plt.close(fig)
    runtime=dict(LPs=len(t),median_seconds=float(t.seconds.median()),p95_seconds=float(t.seconds.quantile(.95)),max_seconds=float(t.seconds.max()),max_variables=int(t.variables.max()),max_nonzeros=int(t.nonzeros.max()))
    write(A/'projection_runtime_summary.json',runtime)
    return dict(runtime=runtime,main=selected.to_dict('records'))

def main():
    check();workflow();report={'calibration':calibration(),'query':query(),'projection':projection()};write(A/'article_numbers.json',report)
    q=report['query']['registered_histories'];p=json.loads((ROOT/'results/new_projection/projection_gate.json').read_text());c=json.loads((ROOT/'results/new_calibration/calibration_gate.json').read_text());v=json.loads((ROOT/'results/new_validation/validation_gate.json').read_text());rt=report['projection']['runtime']
    text=(f'The new extension completes {q["histories"]:,} query histories and {q["cases"]:,} clock-conditioned cases, {p["LPs"]:,} shared-origin projections with {p["metric_rows"]:,} fixed-weight evaluations, and {c["new_fits"]} genuinely new tree fits. Independent sparse/dense validation uses {v["independent_dense_and_sparse_LP_solves"]} solves and {v["feasible_blends_checked"]} checked blends. On this CPU, projection median and 95th-percentile times are {rt["median_seconds"]:.4f} and {rt["p95_seconds"]:.4f} s, with a maximum of {rt["max_variables"]:,} variables and {rt["max_nonzeros"]:,} nonzeros. Timings exclude flight sensing, communication, file loading and aircraft energy. The randomized calibration calculation takes {c["seconds"]:.1f} s including fitting, prediction, storage and enumeration; its ten model fits are logged separately. No new neural training or flight collection occurs.\n')
    (G/'execution.tex').write_text(text,encoding='utf-8')
    write(A/'metric_gate.json',dict(passed=True,calibration_score_replay_max=report['calibration']['max_score_replay_error'],all_new_gates_pass=all(g['passed'] for g in [q,p,c,v])))
    print(json.dumps(dict(query=q,projection=p,calibration=c,runtime=rt,contrasts=report['query']['contrasts']),indent=2))

if __name__=='__main__':main()
