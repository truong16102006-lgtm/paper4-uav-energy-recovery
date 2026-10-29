"""Tables and figures for the registered adaptation grid and numerical budget."""
from common4 import *
from make_article import table,G,F,A
from figure_style import journal_layout
import pandas as pd,numpy as np,matplotlib.pyplot as plt

def main():
    s=pd.read_csv(ROOT/'results/shift/summary.csv');primary=s[s.budget.eq(8)&s.epsilon_Wh.eq(0)&s.halfwidth_s.eq(.5)];rows=[];numbers={}
    for dataset in ['development','external_followup']:
        z=primary[primary.dataset.eq(dataset)];control=float(z[z.method.eq('PCHIP_control')].RMSE_Wh.iloc[0]);numbers[dataset]={'PCHIP_control_Wh':control}
        for shift in ['identity','global_2','interval_gain','ramp','smooth']:
            p=z[z['shift'].eq(shift)].set_index('method');v={m:float(p.loc[m,'RMSE_Wh']) for m in ['unchanged','global_endpoint','bounded_local']};numbers[dataset][shift]=v
            rows.append([('Development' if dataset=='development' else 'External'),shift.replace('_',' '),*[f'{v[m]:.3f}' for m in ['unchanged','global_endpoint','bounded_local']],f'{control:.3f}'])
        numbers[dataset]['primary_reduction_percent']=100*(1-numbers[dataset]['interval_gain']['bounded_local']/numbers[dataset]['interval_gain']['global_endpoint'])
    table('shift_primary.tex','Query-only adaptation at B8, clean readings and h=0.5 s. All methods have the same history-specific minimax radius. Development uses all 440 records and equal source/platform weighting. External follow-up uses 161 correlated windows and an exactly integrated native-power reference; its historical prior uses only the preceding scheduled window\'s sparse readings. This follow-up was designed after inspecting both pools. Errors are continuous RMSE in Wh.',['Pool','Prior shift','Unchanged','Global','Local','PCHIP'],rows,label='tab:shiftprimary')
    rows=[]
    for key,z in s[~s.method.eq('PCHIP_control')].groupby(['dataset','budget','epsilon_Wh','halfwidth_s','shift']):
        dataset,B,eps,h,shift=key;p=z.set_index('method');rows.append([('Dev' if dataset=='development' else 'Ext'),B,f'{eps:g}',f'{h:g}',shift.replace('_',' '),*[f'{p.loc[m,"RMSE_Wh"]:.3f}' for m in ['unchanged','global_endpoint','bounded_local']],f'{p.radius_Wh.iloc[0]:.3f}'])
    table('full_shift.tex','Every registered adaptation condition; no outcome-based selection. Dev/Ext denote development and external follow-up. Columns give Wh. PCHIP control rows and all individual errors are retained in the case/summary CSVs. All methods use identical original reading/time contracts.',['Pool','$B$','$\\epsilon$','$h$','Shift','Unchanged','Global','Local','$R$'],rows,long=True)
    budget=pd.read_csv(ROOT/'results/shift/reference_error_budget.csv');numbers['maximum_linear_energy_peak_difference_Wh']=float(budget.peak_Wh.max());numbers['maximum_linear_energy_RMSE_difference_Wh']=float(budget.RMSE_Wh.max());numbers['original_reference_audit']=json.loads((ROOT/'audit/reference_refinement.json').read_text());numbers['gate']=json.loads((ROOT/'results/shift/gate.json').read_text());write(A/'shift_numbers.json',numbers)
    text=f'The registered interval-gain contrast reduces mean RMSE by {numbers["development"]["primary_reduction_percent"]:.1f}\\% in development and {numbers["external_followup"]["primary_reduction_percent"]:.1f}\\% in the external follow-up relative to global endpoint normalization. The local output is unchanged, within numerical tolerance, by the identity, global-gain and interval-gain transformations. Nonconstant gains inside an interval are outside that exact invariance claim; ramp and smooth results remain in Table~\\ref{{tab:shiftprimary}}.\n\n'
    text+='Without imposed shift, local HGB correction gives 0.336 Wh versus 0.353 Wh for current-reading PCHIP in development. This is an exploratory improvement following the earlier negative learned-prior comparison. It does not reverse that frozen comparison or establish learned superiority on a new campaign. The lagged-profile local method gives 0.628 Wh externally, versus 0.605 Wh for current-reading PCHIP; the simpler interpolation control remains stronger there.\n\n'
    text+=f'All {numbers["gate"]["evaluations"]:,} follow-up recoveries satisfy their original minimax checks. The largest checked primal and risk residuals are {numbers["gate"]["max_primal_residual_Wh"]:.2g} and {numbers["gate"]["max_risk_residual_Wh"]:.2g} Wh. Three external windows use the declared linear historical-prior fallback after initialization or a support gap. Direct comparison of the original saved reference with the refined reference gives a maximum difference of {numbers["original_reference_audit"]["max_original_to_refined_peak_Wh"]:.6f} Wh at a prefix and {numbers["original_reference_audit"]["max_original_to_refined_RMSE_Wh"]:.6f} Wh in continuous RMSE. The analytic linearization budget applies separately to the refined native knots, including the shortened boundary intervals. The follow-up eliminates this particular interpolation approximation through exact polynomial integration; it does not remove physical sensor error.\n'
    (G/'shift_interpretation.tex').write_text(text,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,dataset,title in zip(axes,['development','external_followup'],['Development: HGB prior','External: lagged sparse prior']):
        x=np.arange(5)
        for method,label,color in [('unchanged','Unchanged','#777777'),('global_endpoint','Global normalization','#a5522a'),('bounded_local','Bounded local correction','#176f78')]:
            values=[numbers[dataset][shift][method] for shift in ['identity','global_2','interval_gain','ramp','smooth']];ax.plot(x,values,'o-',ms=3,lw=1,color=color,label=label)
        ax.axhline(numbers[dataset]['PCHIP_control_Wh'],color='#8c629d',ls='--',lw=1,label='Current-reading PCHIP');ax.set(xticks=x,xticklabels=['None','Global','Interval','Ramp','Smooth'],ylabel='Continuous RMSE (Wh)',xlabel='Imposed prior distortion',title=title)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.01),fontsize=8);journal_layout(fig);fig.tight_layout(rect=[0,.25,1,1]);fig.savefig(F/'query_adaptation.png');plt.close(fig);print(json.dumps({d:numbers[d]['primary_reduction_percent'] for d in ['development','external_followup']},indent=2))

if __name__=='__main__':main()
