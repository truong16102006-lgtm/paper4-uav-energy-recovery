"""Recover one declared counter history; JSON input/output, no flight-control action."""
from common4 import *
from minimax_centers import CenterProblem
from review6_envelopes_v2 import envelopes
from review6_envelopes_v2 import prepare
from fast_center import recover_clipped
from local_mass import adapt_local
from query_validated_prior import select_prior
from calibration_enclosure import enclosure
from calibration_spec import power_bounds
from scipy.interpolate import PchipInterpolator
import numpy as np,argparse

def recover(data):
    """Preserve the original interface; opt-in calibration precedes compatibility."""
    if data.get('algorithm')=='coupled':return recover_coupled_api(data)
    if data.get('algorithm')=='shared_affine':return recover_shared_affine_api(data)
    if 'calibration' not in data:return recover_boxes(data)
    specification=dict(data['calibration']);cap=float(data['cap_W'])
    if 'component_specification' in specification:
        supplied=power_bounds(**specification['component_specification'])
        gamma=supplied['relative_power_gain_bound'];bias=supplied['additive_power_bias_bound_W']
    else:
        gamma=float(specification['relative_gain_bound']);bias=float(specification['bias_bound_W']);supplied=None
    if not np.isfinite([gamma,bias,cap]).all() or not (0<=gamma<1 and bias>=0 and cap>=0):raise ValueError('Finite gain in [0,1), nonnegative finite power bias and cap required.')
    en=enclosure(data['time_lower_s'],data['time_upper_s'],data['counter_Wh'],cap,data['epsilon_Wh'],gamma,bias)
    if en['status']!='CONDITIONAL_CALIBRATION_RELAXATION':return dict(status=en['status'],contract='Supplied gain/bias and reading bounds are incompatible; no curve returned.')
    converted=dict(data,counter_Wh=en['centers'].tolist(),epsilon_Wh=en['epsilon'].tolist());converted.pop('calibration')
    out=recover_boxes(converted)
    if 'absolute_counter_Wh' in out:out['relaxed_energy_box_lift_Wh']=out.pop('absolute_counter_Wh')
    out.update(calibration_bounds_supplied_not_measured=True,relative_power_gain_bound=gamma,additive_power_bias_bound_W=bias,sharpness_scope=en['sharpness_scope'],clock_witness_scope='Witness for relaxed energy boxes; not a jointly fitted gain/bias trajectory',contract='Conditional recovery for a conservative rectangular relaxation of supplied bounded gain/bias, reading and time errors. Sensor calibration, synchrony and cap validity require external evidence.')
    if supplied is not None:out['component_bound_conversion']=supplied
    return out

def recover_coupled_api(data):
    from coupled_calibration import CoupledClass
    if data.get('prior_adapter','none')!='none' or float(data.get('risk_factor',1))!=1:raise ValueError('Coupled recovery uses its supplied prior and feasible minimax bracket; other adapters/risk multipliers are not combined implicitly.')
    nominal=np.asarray(data['nominal_times_s'],float);a=np.asarray(data['time_lower_s'],float);b=np.asarray(data['time_upper_s'],float)
    if nominal.shape!=a.shape or a.shape!=b.shape or np.any(a>nominal) or np.any(b<nominal) or np.any(np.diff(nominal)<=0):raise ValueError('Ordered nominal times within their supplied boxes required.')
    if a[0]!=b[0] or a[-1]!=b[-1]:raise ValueError('Current observed-horizon endpoints must have exact times.')
    origin=nominal[0];anchors=nominal-origin;half=np.maximum(nominal-a,b-nominal);spec=data.get('calibration',{});component=None
    if 'component_specification' in spec:
        component=power_bounds(**spec['component_specification']);gamma=component['relative_power_gain_bound'];bias=component['additive_power_bias_bound_W']
    else:gamma=float(spec.get('relative_gain_bound',0));bias=float(spec.get('bias_bound_W',0))
    grid=np.asarray(data.get('grid_times_s',nominal),float)-origin;y=np.asarray(data['counter_Wh'],float)
    if grid[0]!=0 or grid[-1]!=anchors[-1] or np.any(np.diff(grid)<=0):raise ValueError('Prior grid must span exactly the current observed horizon.')
    prior=np.asarray(data['prior_Wh'],float) if 'prior_Wh' in data else PchipInterpolator(anchors,y-y[0])(grid)
    if prior.shape!=grid.shape or not np.isfinite(prior).all():raise ValueError('Finite prior matching the grid required.')
    nodes=int(data.get('coupled_grid_nodes',33))
    if nodes<3 or nodes>513:raise ValueError('coupled_grid_nodes must be between 3 and 513.')
    model=CoupledClass(anchors,y,data['cap_W'],data['epsilon_Wh'],gamma,bias,half);got=model.recover(grid,prior,nodes)
    if got['status']!='CHECKED_COUPLED_CENTER':return got
    shape=polish_api(model,got,grid,prior,'coupled_calibration')
    got.update(E=shape['E'],Q=shape['Q'],joint_witness=shape['witness'],shape_QP_optimized=shape['mode']=='continuous_L2_QP',shape_QP_status=';'.join(shape['statuses']),seconds=got['seconds']+shape['seconds'])
    out=dict(status=got['status'],algorithm='coupled',times_s=(got['t']+origin).tolist(),increment_Wh=got['E'].tolist(),measured_counter_lift_relative_to_first_reading_Wh=got['Q'].tolist(),lower_increment_Wh=got['lower'].tolist(),upper_increment_Wh=got['upper'].tolist(),minimax_radius_lower_Wh=got['discrete_radius_lower_Wh'],conditional_error_upper_Wh=got['continuous_radius_upper_Wh'],optimality_gap_Wh=max(0.,got['continuous_radius_upper_Wh']-got['discrete_radius_lower_Wh']),analytic_grid_gap_bound_Wh=got['analytic_grid_gap_bound_Wh'],joint_calibration_witness=got['joint_witness'],shape_QP_optimized=got['shape_QP_optimized'],shape_QP_status=got['shape_QP_status'],LP_calls=got['LP_calls'],seconds=got['seconds'],clock_scope=got['clock_scope'],reading_error_after_clock_relaxation_Wh=model.eps.tolist(),rounding_scope=got['rounding_scope'],contract='Feasible consumed-energy recovery under supplied joint gain/bias and reading bounds. Exact finite-state model at fixed times; uncertain times use a nominal-time outer relaxation. No future mission energy or physical calibration is inferred.')
    if component is not None:out['component_bound_conversion']=component
    out.update(shape_solver_mode=shape['mode'],numerical_risk_excess_Wh=shape['numerical_risk_excess_Wh'],computed_conditional_error_upper_Wh=out['conditional_error_upper_Wh']+shape['numerical_risk_excess_Wh'])
    out['optimality_gap_Wh']=max(0.,out['computed_conditional_error_upper_Wh']-out['minimax_radius_lower_Wh'])
    if 'target_precision_Wh' in data:
        target=float(data['target_precision_Wh'])
        if not np.isfinite(target) or target<0:raise ValueError('Finite nonnegative precision target required.')
        out.update(target_precision_Wh=target,conditional_precision_reached=bool(got['continuous_radius_upper_Wh']+shape['numerical_risk_excess_Wh']<=target))
    return out

def recover_shared_affine_api(data):
    from shared_affine import SharedAffineClass
    if data.get('prior_adapter','none')!='none' or float(data.get('risk_factor',1))!=1:raise ValueError('Shared-affine recovery uses its supplied prior and feasible minimax bracket; other adapters/risk multipliers are not combined implicitly.')
    nominal=np.asarray(data['nominal_times_s'],float);a=np.asarray(data['time_lower_s'],float);b=np.asarray(data['time_upper_s'],float);y=np.asarray(data['counter_Wh'],float)
    if nominal.shape!=a.shape or a.shape!=b.shape or np.any(a>nominal) or np.any(b<nominal) or np.any(np.diff(nominal)<=0):raise ValueError('Ordered nominal times within their supplied boxes required.')
    if a[0]!=b[0] or a[-1]!=b[-1]:raise ValueError('Current observed-horizon endpoints must have exact times.')
    spec=data['calibration']
    if 'component_specification' in spec:raise ValueError('The shared-affine model requires explicit common-gain, common-bias and residual-drift bounds; pointwise component specifications alone do not establish that structure.')
    origin=nominal[0];anchors=nominal-origin;half=np.maximum(nominal-a,b-nominal);grid=np.asarray(data.get('grid_times_s',nominal),float)-origin
    if grid[0]!=0 or grid[-1]!=anchors[-1] or np.any(np.diff(grid)<=0):raise ValueError('Prior grid must span exactly the current observed horizon.')
    prior=np.asarray(data['prior_Wh'],float) if 'prior_Wh' in data else PchipInterpolator(anchors,y-y[0])(grid)
    if prior.shape!=grid.shape or not np.isfinite(prior).all():raise ValueError('Finite prior matching the grid required.')
    count=int(data.get('coupled_grid_nodes',33))
    if not 3<=count<=513:raise ValueError('coupled_grid_nodes must be between 3 and 513.')
    model=SharedAffineClass(anchors,y,data['cap_W'],data['epsilon_Wh'],spec['relative_gain_bound'],spec['bias_bound_W'],spec['residual_drift_bound_W'],half);got=model.recover(grid,prior,count)
    if got['status']!='CHECKED_SHARED_AFFINE_CENTER':return got
    shape=polish_api(model,got,grid,prior,'shared_affine')
    got.update(E=shape['E'],Q=shape['Q'],R=shape['R'],witness=shape['witness'],shape_QP_optimized=shape['mode']=='continuous_L2_QP',shape_QP_status=';'.join(shape['statuses']),seconds=got['seconds']+shape['seconds'])
    out=dict(status=got['status'],algorithm='shared_affine',times_s=(got['t']+origin).tolist(),increment_Wh=got['E'].tolist(),measured_counter_lift_relative_to_first_reading_Wh=got['Q'].tolist(),integrated_inverse_drift_Wh=got['R'].tolist(),lower_increment_Wh=got['lower'].tolist(),upper_increment_Wh=got['upper'].tolist(),minimax_radius_lower_Wh=got['radius_lower_Wh'],conditional_error_upper_Wh=got['radius_upper_Wh'],optimality_gap_Wh=got['optimality_gap_Wh'],joint_calibration_witness=got['witness'],shape_QP_optimized=got['shape_QP_optimized'],shape_QP_status=got['shape_QP_status'],LP_calls=got['LP_calls'],seconds=got['seconds'],clock_scope=got['clock_scope'],reading_error_after_clock_relaxation_Wh=model.eps.tolist(),calibration_identified_by_data=False,rounding_scope='Float64 checks, not formal outward rounding',contract='One shared unknown gain and bias plus arbitrary bounded residual power drift over the current horizon. Witness parameters establish compatibility; they are not an independently calibrated sensor estimate. No future mission energy is inferred.')
    if 'target_precision_Wh' in data:
        target=float(data['target_precision_Wh'])
        if not np.isfinite(target) or target<0:raise ValueError('Finite nonnegative precision target required.')
        out.update(target_precision_Wh=target,conditional_precision_reached=bool(got['radius_upper_Wh']+shape['numerical_risk_excess_Wh']<=target))
    out.update(shape_solver_mode=shape['mode'],numerical_risk_excess_Wh=shape['numerical_risk_excess_Wh'],computed_conditional_error_upper_Wh=out['conditional_error_upper_Wh']+shape['numerical_risk_excess_Wh'])
    out['optimality_gap_Wh']=max(0.,out['computed_conditional_error_upper_Wh']-out['minimax_radius_lower_Wh'])
    return out

def polish_api(model,got,prior_t,prior,kind):
    from joint_shape_solver import refine
    t=got['t'];raw=np.interp(t,prior_t,prior);raw-=raw[0];bounded=np.r_[0,np.cumsum(np.clip(np.diff(raw),0,model.cap*np.diff(t)/3600))]
    if kind=='coupled_calibration':seed=np.r_[got['E'],got['Q']]/model.scale;radius=got['continuous_radius_upper_Wh']
    else:
        w=got['witness'];seed=np.r_[got['E']/model.scale,got['R']/model.scale,w['inverse_gain'],w['inverse_bias_W']/model.scale,w['origin_parameter_Wh']/model.scale];radius=got['radius_upper_Wh']
    return refine(model,t,got['lower'],got['upper'],radius,bounded,seed,kind)

def recover_boxes(data):
    nominal=np.array(data['nominal_times_s'],float);a=np.array(data['time_lower_s'],float);b=np.array(data['time_upper_s'],float);y=np.array(data['counter_Wh'],float);cap=float(data['cap_W']);eps=data['epsilon_Wh'];k=float(data.get('risk_factor',1))
    env=envelopes(a,b,y,cap,eps)
    if env['status']!='CONDITIONAL_CLOSED_CLOCK_ENVELOPE':return dict(status=env['status'],diagnostic=env)
    if len(nominal)!=len(y) or np.any(np.diff(nominal)<=0) or np.any(nominal<a) or np.any(nominal>b):raise ValueError('Nominal observation times must be strictly increasing inside their boxes.')
    algorithm=data.get('algorithm','clip')
    if algorithm=='clip':return recover_fast_api(data,nominal,a,b,y,cap,eps,k)
    if algorithm!='QP':raise ValueError('algorithm must be clip or QP')
    grid=np.array(data.get('grid_times_s',nominal),float);problem=CenterProblem(grid,a,b,y,cap,eps)
    if 'prior_Wh' in data:
        raw=np.array(data['prior_Wh'],float)
        if raw.shape!=grid.shape:raise ValueError('prior_Wh must have the shape of grid_times_s.')
        prior=np.interp(problem.t+problem.t0,grid,raw)
    else:prior=PchipInterpolator(nominal,y-y[0])(problem.t+problem.t0)
    curve,diag=problem.solve(prior,k);absolute=curve+diag['origin_centered_Wh'];clock=[]
    for ai,bi,li,ui in zip(problem.a,problem.b,problem.l,problem.u):
        if np.interp(ai,problem.t,absolute)>=li-1e-10:x=ai
        else:
            j=int(np.searchsorted(np.maximum.accumulate(absolute),li,side='left'));j=min(max(j,1),len(absolute)-1);den=absolute[j]-absolute[j-1];x=problem.t[j] if den<=0 else problem.t[j-1]+(li-absolute[j-1])*(problem.t[j]-problem.t[j-1])/den;x=max(ai,x)
        clock.append(min(bi,max(x,clock[-1] if clock else 0)))
    clock=np.array(clock);values=np.interp(clock,problem.t,absolute);residual=max(0,float(np.max(problem.a-clock)),float(np.max(clock-problem.b)),float(np.max(-np.diff(clock))),float(np.max(problem.l-values)),float(np.max(values-problem.u)));assert residual<1e-8
    return dict(status='MINIMAX_CENTER_OPTIMIZED' if diag['optimized'] else 'CHECKED_MIDPOINT_FALLBACK',risk_factor=k,times_s=(problem.t+problem.t0).tolist(),increment_Wh=curve.tolist(),absolute_counter_Wh=(absolute+y[0]).tolist(),lower_increment_Wh=problem.lower.tolist(),upper_increment_Wh=problem.upper.tolist(),sharp_radius_Wh=problem.R,worst_case_ceiling_Wh=k*problem.R,observation_clock_witness_s=(clock+problem.t0).tolist(),clock_witness_residual_Wh_or_s=residual,solver=diag,contract='Conditional numerical recovery with supplied monotone bounded-slope and closed ordered time/value boxes; not sensor calibration or flight-control authorization.')

def recover_fast_api(data,nominal,a,b,y,cap,eps,k):
    grid=np.asarray(data.get('grid_times_s',nominal),float)
    prior=np.asarray(data['prior_Wh'],float) if 'prior_Wh' in data else PchipInterpolator(nominal,y-y[0])(grid)
    adapter=data.get('prior_adapter','none');adaptation=None
    if adapter=='bounded_local_mass':grid,prior,adaptation=adapt_local(grid,prior,nominal,y,cap)
    elif adapter=='query_validated':
        if 'prior_Wh' not in data:raise ValueError('query_validated requires an explicit prior independent of the current returned reading values.')
        grid,prior,adaptation=select_prior(grid,prior,nominal,y,a,b,cap,eps)
        adaptation['prior_information_contract']='Caller-supplied prior must be independent of current returned reading values; the API cannot verify its provenance.'
    elif adapter!='none':raise ValueError('prior_adapter must be none, bounded_local_mass or query_validated')
    got=recover_clipped(grid,prior,a,b,y,cap,eps,k)
    if got['status']!='CHECKED_CLIPPED_MINIMAX_CENTER':return got
    aa,bb,l,u,m,t0,y0,*_=prepare(a,b,y,cap,eps);t=got['grid'];curve=got['increment'];absolute=curve+got['origin_centered_Wh'];clock=[]
    for ai,bi,li in zip(aa,bb,l):
        if np.interp(ai,t,absolute)>=li-1e-10:x=ai
        else:
            j=int(np.searchsorted(np.maximum.accumulate(absolute),li,side='left'));j=min(max(j,1),len(t)-1);den=absolute[j]-absolute[j-1];x=t[j] if den<=0 else t[j-1]+(li-absolute[j-1])*(t[j]-t[j-1])/den;x=max(ai,x)
        clock.append(min(bi,max(x,clock[-1] if clock else 0.)))
    clock=np.asarray(clock);values=np.interp(clock,t,absolute);res=max(0.,float(np.max(aa-clock)),float(np.max(clock-bb)),float(np.max(-np.diff(clock))),float(np.max(l-values)),float(np.max(values-u)));assert res<1e-8
    diagnostic={key:got[key] for key in ['primal_residual_Wh','risk_residual_Wh','conditional_worst_Wh','prior_slope_repair_RMSE_Wh','origin_centered_Wh','origin_interval_Wh','knots','seconds']};diagnostic['optimizer_used']=False
    out=dict(status='MINIMAX_CENTER_CLIPPED',algorithm='clip',risk_factor=k,times_s=(t+t0).tolist(),increment_Wh=curve.tolist(),absolute_counter_Wh=(absolute+y0).tolist(),lower_increment_Wh=got['lower'].tolist(),upper_increment_Wh=got['upper'].tolist(),sharp_radius_Wh=got['radius_Wh'],worst_case_ceiling_Wh=k*got['radius_Wh'],observation_clock_witness_s=(clock+t0).tolist(),clock_witness_residual_Wh_or_s=res,solver=diagnostic,contract='Conditional numerical recovery for supplied bounded-slope and closed ordered time/value boxes; sensor calibration and physical cap validity are separate.')
    out.update(prior_adapter=adapter,computed_conditional_ceiling_Wh=max(out['worst_case_ceiling_Wh'],got['conditional_worst_Wh']),numerical_risk_excess_Wh=max(0.,got['conditional_worst_Wh']-out['worst_case_ceiling_Wh']),numerical_bound_scope='Float64 diagnostic, not a formal outward-rounded enclosure')
    if adaptation is not None:out['adaptation']=adaptation
    if 'target_precision_Wh' in data:
        target=float(data['target_precision_Wh']);assert np.isfinite(target) and target>=0;out.update(target_precision_Wh=target,conditional_precision_reached=bool(out['computed_conditional_ceiling_Wh']<=target))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();data=json.loads(Path(a.input).read_text(encoding='utf-8'));got=recover(data);write(Path(a.output),got);print(got['status'])

if __name__=='__main__':main()
