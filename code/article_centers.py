"""Complete center-extension summaries; primary contrast fixed in its protocol."""
from common4 import *
from figure_style import journal_layout
from make_article import table,G,F,A,pair_interval
import numpy as np,pandas as pd,matplotlib.pyplot as plt,sqlite3,io

def contrast(f,left,right):
    keys=['flight_id','source_id','platform'];a=f[f.prior.eq(left)].set_index(keys).continuous_RMSE_Wh;b=f[f.prior.eq(right)].set_index(keys).continuous_RMSE_Wh
    z=(a-b).rename('difference').reset_index();rng=np.random.default_rng(2026102704);source=z.groupby(['platform','source_id']).difference.mean();boot=np.zeros(5000)
    for platform in source.index.get_level_values(0).unique():
        values=source.loc[platform].to_numpy();boot+=values[rng.integers(0,len(values),(5000,len(values)))].mean(1)/source.index.get_level_values(0).nunique()
    return dict(estimate=macro(z,'difference'),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),sources=len(source))

def main():
    s=pd.read_csv(ROOT/'results/new_centers/center_summary.csv');f=pd.read_csv(ROOT/'results/new_centers/center_cases.csv');gate=json.loads((ROOT/'results/new_centers/centers_gate.json').read_text());primary=f[f.budget.eq(8)&f.policy.eq('uniform')&f.epsilon_Wh.eq(0)&f.halfwidth_s.eq(.5)&f.risk_factor.eq(1)];ci=contrast(primary,'HGB','PCHIP');write(A/'center_primary_contrast.json',ci)
    rows=[]
    for eps in [0,.1]:
        for name in ['HGB','Linear','PCHIP']:
            z=s[s.budget.eq(8)&s.policy.eq('uniform')&s.epsilon_Wh.eq(eps)&s.halfwidth_s.eq(.5)&s.prior.eq(name)].set_index('risk_factor');a=z.loc[1]
            rows.append([f'{eps:g}',name,f'{a.raw_RMSE_Wh:.3f}',f'{a.midpoint_RMSE_Wh:.3f}',f'{a.continuous_RMSE_Wh:.3f}',f'{z.loc[2,"continuous_RMSE_Wh"]:.3f}',f'{a.radius_Wh:.3f}'])
    table('centers_main.tex','Minimax-preserving recovery on all 440 records, uniform B8 and $h=0.5$ s. All errors are continuous time-normalized RMSE in Wh. $k=1$ keeps the sharp worst-case radius $R$; $k=2$ projects into the full feasible polyline class with worst-case ceiling $2R$. The same measurement history, optimizer and source/platform weights are used for each prior.',['$\\epsilon$','Prior','Raw','Midpoint','$k=1$','$k=2$','$R$'],rows,label='tab:centers')
    rows=[];platforms=[]
    for platform,z in primary.groupby('platform'):
        a=z.groupby(['source_id','prior'])[['continuous_RMSE_Wh','midpoint_RMSE_Wh','radius_Wh']].mean().groupby('prior').mean();v={name:float(a.loc[name,'continuous_RMSE_Wh']) for name in ['HGB','Linear','PCHIP']};v.update(platform=platform,midpoint=float(a.midpoint_RMSE_Wh.iloc[0]),sources=z.source_id.nunique(),records=z.flight_id.nunique());platforms.append(v)
        rows.append([{'cmu_quadrotor':'CMU','pixhawk':'Pixhawk','speedy':'SpeedyBee'}[platform],v['records'],v['sources'],f'{v["midpoint"]:.3f}',*[f'{v[name]:.3f}' for name in ['HGB','Linear','PCHIP']]])
    table('centers_platforms.tex','Prespecified primary setting separated by platform label: uniform B8, clean values, $h=0.5$ s, $k=1$. Errors are source-macro continuous RMSE in Wh. Platform labels do not represent three independent aircraft populations.',['Label','Records','Sources','Midpoint','HGB','Linear','PCHIP'],rows,label='tab:centerplatform')
    rows=[]
    for key,z in s.groupby(['policy','budget','epsilon_Wh','halfwidth_s','risk_factor'],sort=True):
        policy,budget,eps,h,k=key;a=z.set_index('prior');r=a.loc['HGB'];rows.append([('Uniform' if policy=='uniform' else 'Guard'),budget,f'{eps:g}',f'{h:g}',int(k),f'{r.midpoint_RMSE_Wh:.3f}',*[f'{a.loc[name,"continuous_RMSE_Wh"]:.3f}' for name in ['HGB','Linear','PCHIP']],f'{r.radius_Wh:.3f}'])
    table('full_centers.tex','Every registered minimax-center setting. Continuous RMSE and sharp radii are in Wh. Each setting retains all 440 records and all three priors. $k=1$ preserves $R$; $k=2$ permits the full feasible class. Provider is clean at zero error and bounded drift at 0.1 Wh.',['Policy','$B$','$\\epsilon$','$h$','$k$','Mid','HGB','Linear','PCHIP','$R$'],rows,long=True)
    baseline=float(primary.midpoint_RMSE_Wh.groupby([primary.platform,primary.source_id]).mean().groupby(level=0).mean().mean());learned=macro(primary[primary.prior.eq('HGB')],'continuous_RMSE_Wh');pchip=macro(primary[primary.prior.eq('PCHIP')],'continuous_RMSE_Wh');reduction=100*(baseline-learned)/baseline;vs=100*(pchip-learned)/pchip
    word='lower' if ci['estimate']<0 else 'higher';interpret=(f'In the prespecified center-extension contrast, learned HGB center recovery has mean continuous RMSE {learned:.3f} Wh, compared with {pchip:.3f} Wh for the matched PCHIP center and {baseline:.3f} Wh for the sharp midpoint. HGB minus PCHIP is {ci["estimate"]:.4f} Wh, paired descriptive source-bootstrap interval [{ci["low"]:.4f}, {ci["high"]:.4f}] Wh. Its RMSE is {abs(vs):.1f}\\% {word} than the matched PCHIP center in this pooled summary. The learned center changes midpoint RMSE by {reduction:.1f}\\% while keeping the same continuous worst-case radius. The primary setting and all sensitivity outcomes are shown in Tables~\\ref{{tab:centers}} and~\\ref{{tab:centerplatform}}; no platform-specific winner is substituted for the pooled contrast.\n\n')
    interpret+=(f'Across all {gate["QP_evaluations"]:,} QP evaluations, {gate["optimized"]:,} meet both optimization and original-constraint checks, with {gate["fallbacks"]:,} midpoint fallbacks. Maximum accepted primal and risk residuals are {gate["max_primal_residual_Wh"]:.2g} and {gate["max_risk_residual_Wh"]:.2g} Wh. Median and 95th-percentile solve times, including any fixed retry, are {gate["median_seconds"]:.4f} and {gate["p95_seconds"]:.4f} s on this CPU. These are local computation times; sensing, communication and onboard energy cost remain unmeasured.\n')
    (G/'centers_interpretation.tex').write_text(interpret,encoding='utf-8');write(A/'center_article_numbers.json',dict(primary=ci,midpoint_RMSE=baseline,HGB_RMSE=learned,PCHIP_RMSE=pchip,percent_reduction_vs_midpoint=reduction,percent_reduction_vs_PCHIP=vs,platforms=platforms,execution=gate))
    fig,axes=plt.subplots(1,2,figsize=(10,3.6));colors={'HGB':'#176f78','Linear':'#a5522a','PCHIP':'#8263a2'}
    for eps,ax in zip([0,.1],axes):
        z=s[s.policy.eq('uniform')&s.epsilon_Wh.eq(eps)&s.halfwidth_s.eq(.5)&s.risk_factor.eq(1)]
        for name in colors:
            a=z[z.prior.eq(name)].sort_values('budget');ax.plot(a.budget,a.continuous_RMSE_Wh,marker='o',color=colors[name],label=name+' center')
        a=z[z.prior.eq('HGB')].sort_values('budget');ax.plot(a.budget,a.midpoint_RMSE_Wh,'--',color='#555555',label='Sharp midpoint');ax.set(xticks=[4,8,16],xlabel='Query budget, including endpoints',ylabel='Continuous RMSE (Wh)',title=f'Reading halfwidth: {eps:g} Wh')
    axes[0].legend(fontsize=8);journal_layout(fig);fig.tight_layout();fig.savefig(F/'minimax_centers.png');plt.close(fig)
    # Representative record selected by median observed duration, never by error.
    con=sqlite3.connect(ROOT/'results/new_centers/center_curves.sqlite3');examples=[]
    for key,metadata,blob in con.execute('SELECT * FROM curves'):
        meta=json.loads(metadata)
        if meta['budget']==8 and meta['policy']=='uniform' and meta['epsilon_Wh']==0 and meta['halfwidth_s']==.5:
            with np.load(io.BytesIO(blob)) as z:examples.append((float(z['t'][-1]),meta['flight_id'],key))
    _,fid,key=sorted(examples)[len(examples)//2];metadata,blob=con.execute('SELECT metadata,arrays FROM curves WHERE problem_id=?',(key,)).fetchone();meta=json.loads(metadata)
    with np.load(io.BytesIO(blob)) as z:
        t=z['t'];R=meta['radius_Wh'];fig,ax=plt.subplots(figsize=(9,3.6));ax.fill_between(t,z['lower'],z['upper'],color='#d7dce0',label='Feasible truth envelope');ax.fill_between(t,z['upper']-R,z['lower']+R,color='#b3d5d3',alpha=.8,label='Minimax-center band');ax.plot(t,z['truth'],color='#252525',lw=1.5,label='Retained reference');ax.plot(t,z['midpoint'],'--',color='#6e7275',label='Sharp midpoint');ax.plot(t,z['HGB_k1'],color=colors['HGB'],lw=1.4,label='HGB minimax center');ax.set(xlabel='Elapsed time (s)',ylabel='Consumed energy (Wh)');ax.legend(fontsize=8,ncol=2);journal_layout(fig);fig.tight_layout();fig.savefig(F/'center_example.png');plt.close(fig)
    con.close();write(A/'center_example_selection.json',dict(rule='median duration across all 440 records; deterministic flight-ID tie order; no error ranking',flight_id=fid,problem_id=key))
    print(json.dumps(dict(primary=ci,midpoint=baseline,HGB=learned,PCHIP=pchip,execution=gate),indent=2))

if __name__=='__main__':main()
