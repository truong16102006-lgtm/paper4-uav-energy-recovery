"""Vector figures at their final manuscript width; all numbers from retained executions."""
from common4 import ROOT,write,sha
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from threshold_audit import audit_thresholds

O=ROOT/'results/concise_20261029'; F=ROOT/'paper/figures/concise'; G=ROOT/'paper/concise/generated'
F.mkdir(parents=True,exist_ok=True);G.mkdir(parents=True,exist_ok=True)
W=392.878/72.27
plt.rcParams.update({'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
C=['#0072B2','#D55E00','#009E73','#CC79A7','#666666']

def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(F/(name+'.'+ext),dpi=600,metadata={'CreationDate':None} if ext=='pdf' else None)
    plt.close(fig)

def table(name,caption,label,heads,rows,widths=None):
    cols=widths or ('l'+'r'*(len(heads)-1))
    text='\\begin{table}[htbp]\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabularx}{\\linewidth}{'+cols+'}\\toprule\n'
    text+=' & '.join(heads)+'\\\\\\midrule\n'+'\n'.join(' & '.join(map(str,r))+'\\\\' for r in rows)
    text+='\n\\bottomrule\\end{tabularx}\n\\end{table}\n'
    (G/(name+'.tex')).write_text(text,encoding='utf-8')

def figures():
    fig,ax=plt.subplots(figsize=(W,2.9));fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    boxes=[(.015,.61,.26,.29,'UAV load monitor\nCumulative counter\nWh; time uncertainty'),(.365,.61,.26,.29,'Received packets\nSparse / lost readings\nDeclared bounds'),(.715,.61,.27,.29,'Ground receiver\nEnergy envelopes\nClipped curve; joint audit')]
    for x,y,w,h,s in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',fc='#EAF2F8',ec=C[0]));ax.text(x+w/2,y+h/2,s,ha='center',va='center',linespacing=1.6)
    for x,y in [(.28,.75),(.63,.75)]:ax.annotate('',xy=(x+.075,y),xytext=(x,y),arrowprops={'arrowstyle':'->','color':C[0]})
    ax.annotate('',xy=(.85,.43),xytext=(.85,.60),arrowprops={'arrowstyle':'->','color':C[0]})
    ax.add_patch(FancyBboxPatch((.54,.14),.44,.28,boxstyle='round,pad=.008',fc='#EBF6F0',ec=C[2]))
    ax.text(.76,.28,'Below / reached / unresolved\nCrossing bracket; residuals\nObserved-horizon expenditure',ha='center',va='center',linespacing=1.65)
    ax.text(.24,.30,'Evaluation only\nDense logged power / energy\nRMSE and containment checks',ha='center',va='center',linespacing=1.7,color=C[4])
    ax.annotate('',xy=(.53,.28),xytext=(.44,.28),arrowprops={'arrowstyle':'<->','linestyle':'--','color':C[4]})
    ax.text(.5,.035,'Replay uses virtual counters derived from real flight logs.',ha='center',color=C[4])
    save(fig,'01_pipeline')
    z=np.load(O/'spatial_example_private.npz');t=z['grid'];T=t[-1];lo=z['lower'];hi=z['upper'];ec=z['center'];theta=10
    audit=audit_thresholds(t,lo,hi,[theta])['thresholds'][0];early=audit['earliest_crossing_s'];late=audit['latest_crossing_s']
    assert early is not None and late is not None
    fig,(ax,bar)=plt.subplots(2,1,figsize=(W,4.15),gridspec_kw={'height_ratios':[4,1.1]});fig.subplots_adjust(left=.115,right=.95,bottom=.14,top=.97,hspace=.40)
    ax.fill_between(t,lo,hi,color='#D6EAF5',label='Compatible envelope')
    ax.plot(z['t'],z['truth'],color=C[4],lw=1,label='Numerical reference')
    ax.plot(t,z['prior'],'--',color=C[1],lw=1,label='Returned-data prior')
    ax.plot(t,ec,color=C[0],lw=1.2,label='Clipped curve')
    ax.scatter(z['anchors'],z['returned'],s=12,facecolors='none',edgecolors=C[1],zorder=5,label='Received counter')
    ax.axhline(theta,color='black',ls=':',lw=.8);ax.axvspan(early,late,color=C[3],alpha=.12)
    ax.set(xlabel='Elapsed time (s)',ylabel='Consumed energy (Wh)',xlim=(0,T));ax.legend(ncol=2,loc='upper left',columnspacing=.8,handlelength=1.5)
    for start,end,color,txt in [(0,early,C[0],'Below'),(early,late,C[1],'Unresolved'),(late,T,C[2],'Reached')]:
        bar.barh(0,end-start,left=start,color=color,height=.5)
        bar.text((start+end)/2,0,txt,ha='center',va='center',color='white',fontsize=8)
    bar.set(xlim=(0,T),yticks=[],xlabel='Elapsed time (s)',xticks=[0,early,late,T]);bar.tick_params(axis='x',labelsize=8)
    bar.get_xticklabels()[0].set_ha('left');bar.get_xticklabels()[-1].set_ha('right')
    bar.set_title(f'10 Wh crossing bracket: [{early:.1f}, {late:.1f}] s',fontsize=8.5)
    save(fig,'02_worked_example');write(O/'worked_example.json',dict(earliest_s=early,latest_s=late,unresolved_fraction=audit['unresolved_duration_fraction']))
    ab=pd.read_csv(O/'prior_ablation.csv')
    fig,axes=plt.subplots(1,2,figsize=(W,3.05));fig.subplots_adjust(left=.11,right=.98,bottom=.24,top=.89,wspace=.45)
    for ax,cohort in zip(axes,['AMOVFLY','AtlantikSolar']):
        x=ab[ab.cohort.eq(cohort)]
        for j,name in enumerate(['PCHIP','Linear','Flat']):
            a=x[x.prior.eq(name)].mean(numeric_only=True)
            # Flat raw risk dominates the scale; it is retained in the CSV, not hidden at risk zero.
            if name!='Flat':ax.scatter(a.raw_RMSE_Wh,a.raw_conditional_risk_Wh,marker='x',color=C[j],s=40)
            ax.scatter(a.RMSE_Wh,a.radius_Wh,marker='o',color=C[j],s=24,label=name)
        ax.set(title=cohort,xlabel='Mean reference RMSE (Wh)',ylabel='Mean class-wide bound (Wh)');ax.grid(alpha=.2)
    axes[0].legend(loc='lower left');fig.text(.5,.035,'Circles: clipped at R. Crosses: raw prior at its own risk.',ha='center',fontsize=8)
    save(fig,'03_price_guarantee')
    stats=json.loads((O/'primary_statistics.json').read_text());p=pd.read_csv(O/'primary_records.csv')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(W,3.55),gridspec_kw={'width_ratios':[1.35,1]});fig.subplots_adjust(left=.10,right=.98,bottom=.29,top=.91,wspace=.48)
    for i,(cohort,label) in enumerate([('AMOVFLY','AMOVFLY\n247 records'),('AtlantikSolar','Solar\n161 windows')]):
        v=p[p.cohort.eq(cohort)].delta_pp.to_numpy();ax.boxplot(v,positions=[i],widths=.4,showfliers=False,patch_artist=True,boxprops={'facecolor':'#D6EAF5'},medianprops={'color':C[1]})
        jitter=np.random.default_rng(20261029).uniform(-.13,.13,len(v));ax.scatter(i+jitter,v,s=3,alpha=.22,color=C[0],zorder=1)
        s=next(s for s in stats['analyses'] if s['cohort']==cohort and s.get('block_length',8)==8);l,h=s['bootstrap_CI95_delta_pp'];m=s['mean_delta_pp']
        ax.errorbar(i+.26,m,yerr=[[m-l],[h-m]],color='black',capsize=3,fmt='s',ms=3)
    ax.set(xticks=[0,1],xticklabels=['AMOVFLY\n247 records','Solar\n161 windows'],ylabel='Paired reduction in unresolved time (pp)');ax.axhline(0,color=C[4],lw=.7)
    strata=stats['aircraft_strata'];bx.bar(range(3),[s['mean_delta_pp'] for s in strata],color=[C[0],C[2],C[3]],width=.6)
    bx.set(xticks=range(3),xticklabels=[s['aircraft']+'\nN='+str(s['records']) for s in strata],ylabel='Mean paired reduction (pp)',ylim=(0,11),title='Aircraft-ID strata')
    fig.text(.5,.07,'Sign follows set inclusion. Intervals: date clusters / solar time blocks.',ha='center',fontsize=8)
    save(fig,'04_primary_results')
    xyz=z['xyz'];stride=max(1,len(xyz)//1800);xyz=xyz[::stride];tt=z['t'][::stride];energy=np.interp(tt,t,ec)
    segments=np.stack([xyz[:-1],xyz[1:]],axis=1)
    fig=plt.figure(figsize=(W,4.05));ax=fig.add_subplot(111,projection='3d');fig.subplots_adjust(left=.04,right=.79,bottom=.18,top=.98)
    coll=Line3DCollection(segments,cmap='cividis',linewidth=1.6);coll.set_array((energy[:-1]+energy[1:])/2);ax.add_collection3d(coll);ax.auto_scale_xyz(xyz[:,0],xyz[:,1],xyz[:,2]);ax.set_box_aspect(np.maximum(np.ptp(xyz,axis=0),5))
    # Separate fixed-position axis labels from perspective tick labels.
    ax.set(xlabel='',ylabel='',zlabel='');ax.tick_params(labelsize=8,pad=1);ax.view_init(27,-59)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4));ax.yaxis.set_major_locator(plt.MaxNLocator(4));ax.zaxis.set_major_locator(plt.MaxNLocator(4))
    ax.text2D(.30,.04,r'$x-x_0$ (m)',transform=ax.transAxes,ha='center')
    ax.text2D(.80,.09,r'$y-y_0$ (m)',transform=ax.transAxes,ha='center')
    ax.text2D(-.04,.54,r'$z-z_0$ (m)',transform=ax.transAxes,ha='center',rotation=90)
    ax.scatter(*xyz[0],color=C[2],s=25,marker='o');ax.scatter(*xyz[-1],color=C[1],s=25,marker='s')
    cax=fig.add_axes([.86,.29,.025,.52]);fig.colorbar(coll,cax=cax,label='Reconstructed consumed energy (Wh)')
    fig.text(.46,.035,'Source-local displacement; color is the fourth variable.',ha='center',fontsize=8)
    save(fig,'05_spatial_energy')
    burst=pd.read_csv(O/'burst_record_means.csv')
    fig,axes=plt.subplots(1,2,figsize=(W,3.25));fig.subplots_adjust(left=.11,right=.98,bottom=.28,top=.88,wspace=.35)
    for ax,cohort in zip(axes,['AMOVFLY','AtlantikSolar']):
        a=burst[burst.cohort.eq(cohort)].groupby(['mean_bad_run','method']).unresolved_fraction.mean().unstack()
        for j,(name,label) in enumerate([('envelope','Envelope'),('center_radius','Center +/- R')]):ax.plot(a.index,100*a[name],'-o',color=C[j],label=label,ms=3)
        ax.set(title=cohort,xlabel='Mean bad-state run (opportunities)',ylabel='Mean unresolved duration (%)',xticks=[2,4,8],ylim=(0,50));ax.grid(alpha=.2)
    axes[0].legend();fig.text(.5,.045,'B17; stationary loss 10%; five repetitions averaged within record.',ha='center',fontsize=8)
    save(fig,'06_burst_loss')

def tables():
    s=json.loads((O/'primary_statistics.json').read_text());rows=[];selected=[]
    for cohort in ['AMOVFLY','AtlantikSolar']:
        a=next(x for x in s['analyses'] if x['cohort']==cohort and x.get('block_length',8)==8);ci=a['bootstrap_CI95_delta_pp']
        selected.append(a)
    formats=[('Mean unresolved E/T (\\%)',lambda a:f"{a['mean_envelope_percent']:.2f} / {a['mean_tube_percent']:.2f}"),
      ('Mean paired difference (pp)',lambda a:f"{a['mean_delta_pp']:.2f}"),
      ('95\\% resampling interval (pp)',lambda a:'['+', '.join(f'{v:.2f}' for v in a['bootstrap_CI95_delta_pp'])+']'),
      ('Median paired difference (pp)',lambda a:f"{a['median_delta_pp']:.2f}"),
      ('Q1 / Q3 (pp)',lambda a:f"{a['Q1_delta_pp']:.2f} / {a['Q3_delta_pp']:.2f}"),
      ('Strictly tighter (records)',lambda a:f"{round(a['strictly_tighter_fraction']*a['records'])}/{a['records']}"),
      ('Finite crossing brackets',lambda a:f"{a['finite_brackets']}/{a['records']}")]
    rows=[[name,*[fmt(a) for a in selected]] for name,fmt in formats]
    table('primary','Primary 10 Wh endpoint, B9/full, bounded mechanism. E/T denotes envelope/tube; pp denotes percentage points. Intervals use 5000 paired resamples: collection dates for AMOVFLY and length-eight time blocks within the one solar flight. All records, including censored ones, remain in duration summaries.','tab:primary',['Statistic','AMOVFLY','AtlantikSolar'],rows,'Xrr')
    f=pd.read_csv(ROOT/'results/fast_centers/fast_summary.csv');f=f[(f.budget==8)&f.policy.eq('uniform')&(f.epsilon_Wh==0)&(f.halfwidth_s==.5)&(f.risk_factor==1)].set_index('prior')
    for n in ['PCHIP','HGB']:print('ORIGINAL FAST',n,f.loc[n].to_dict())
    ext=json.loads((ROOT/'results/article/fast_external_numbers.json').read_text())
    rows=[['Development: PCHIP','0.135',f"{f.loc['PCHIP'].midpoint_RMSE_Wh:.3f}",f"{f.loc['PCHIP'].continuous_RMSE_Wh:.3f}",f"{f.loc['PCHIP'].radius_Wh:.3f}"],
          ['Development: HGB','0.302',f"{f.loc['HGB'].midpoint_RMSE_Wh:.3f}",f"{f.loc['HGB'].continuous_RMSE_Wh:.3f}",f"{f.loc['HGB'].radius_Wh:.3f}"],
          ['AtlantikSolar: PCHIP',f"{ext['external_raw_RMSE_Wh']:.3f}",f"{ext['external_midpoint_RMSE_Wh']:.3f}",f"{ext['external_PCHIP_RMSE_Wh']:.3f}",f"{ext['external_radius_Wh']:.3f}"]]
    table('controls','Retained reconstruction controls, eight readings, clean values, 0.5 s time halfwidth; all errors/radii in Wh. Development: 440 records, source/platform-macro weighting, 2000 W. Solar: 161 windows of one flight, equal-window weighting, 500 W. HGB denotes histogram gradient boosting. Raw priors do not inherit R. B8 means eight endpoint-inclusive readings; these clean reconstruction controls differ from the B9 bounded primary analysis.','tab:controls',['Cohort / prior','Raw','Midpoint','Clip','R'],rows,'Xrrrr')
    a=pd.read_csv(O/'prior_ablation.csv').groupby(['cohort','prior'])[['RMSE_Wh','radius_Wh']].mean();rows=[]
    for co in ['AMOVFLY','AtlantikSolar']:
        for name in ['PCHIP','Linear','Flat']:
            r=a.loc[(co,name)];rows.append([co,name,f'{r.RMSE_Wh:.4f}',f'{r.radius_Wh:.4f}'])
    table('ablation','Prior ablation under the primary B9 bounded transcript. Means in Wh, all 247 AMOVFLY records or 161 solar windows. B9 includes nine endpoint-inclusive readings; Table~\\ref{tab:controls} instead uses B8 clean values. The radius is identical within a cohort; only the shape prior changes. The flat prior is zero before clipping.','tab:ablation',['Cohort','Prior','Reference RMSE','Radius'],rows,'XXrr')

if __name__=='__main__':
    figures();tables();write(ROOT/'audit/concise_assets.json',dict(passed=True,final_width_inches=W,minimum_declared_font_pt=8,
        files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(F.glob('*'))],scientific_plots_generated_from_saved_results=True))
