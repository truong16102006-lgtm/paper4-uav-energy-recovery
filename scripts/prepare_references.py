"""Recreate extension references after acquisition; no model training."""
from pathlib import Path
import sys,zipfile,tempfile,shutil,json
import numpy as np
import pandas as pd
import h5py
R=Path(__file__).resolve().parents[1]

def prepare_amov():
    ledger=pd.read_csv(R/'results/robust_transfer/evaluation_ledger.csv',keep_default_na=False)
    ledger=ledger[ledger.role.eq('TEST')&ledger.evaluation_status.eq('ELIGIBLE')];out=[]
    with zipfile.ZipFile(R/'data/external_amovfly/AMOVFLY_pinned.zip') as z:
        prefix=z.namelist()[0]
        for row in ledger.itertuples():
            d=pd.read_csv(z.open(prefix+row.path),usecols=['time','battery_voltage','battery_current']).apply(pd.to_numeric,errors='coerce')
            good=np.isfinite(d).all(axis=1)&d.battery_voltage.gt(0)&d.battery_current.ge(0)
            d=d[good].sort_values('time',kind='stable').drop_duplicates('time',keep='first')
            out.append(pd.DataFrame(dict(record_id=row.record_id,time_s=d.time.to_numpy()-d.time.iloc[0],power_W=(d.battery_voltage*d.battery_current).to_numpy())))
    pd.concat(out,ignore_index=True).to_parquet(R/'data/external_amovfly/native_power.parquet',index=False)

def prepare_solar():
    directory=R/'data/external_solar'
    with tempfile.TemporaryDirectory(prefix='uav_mat_',dir=directory) as tmp:
        mat=Path(tmp)/'sysvector.mat'
        with zipfile.ZipFile(directory/'AtlantikSolar_81hFlight_Dataset.zip') as z,z.open('sysvector.mat') as src,mat.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
        with h5py.File(mat,'r') as f:
            refs=f['#subsystem#/MCOS'][()].ravel();names=[''.join(chr(int(c)) for c in f[r][()].ravel()) for r in f[refs[4]][()].ravel()];values=f[refs[5]][()].ravel()
            a={n:np.asarray(f[values[names.index(n)]][()]).ravel() for n in ['POWS_t','POWS_Main_Volt','POWS_Main_Cur']}
    d=pd.DataFrame(dict(source_row=np.arange(len(a['POWS_t'])),time_s=a['POWS_t']/1e6,voltage_V=a['POWS_Main_Volt'],current_A=a['POWS_Main_Cur']));d['power_W']=d.voltage_V*d.current_A
    d['valid_electrical']=np.isfinite(d[['time_s','voltage_V','current_A']]).all(axis=1)&d.voltage_V.gt(0)&d.current_A.ge(0)
    d=d[d.valid_electrical].sort_values('time_s',kind='stable').drop_duplicates('time_s',keep='first');d.to_parquet(directory/'native_power.parquet',index=False)
    times=d.time_s.to_numpy();p=d.power_W.to_numpy();gap=np.diff(times);e=np.r_[0.,np.cumsum(np.where((gap>0)&(gap<=2),gap/7200,0.)*(p[:-1]+p[1:]))];out=[]
    ledger=pd.read_csv(R/'results/external_solar/window_ledger.csv')
    for row in ledger[ledger.status.eq('ELIGIBLE')].itertuples():
        t=np.unique(np.r_[row.left_s,times[(times>row.left_s)&(times<row.right_s)],row.right_s]);energy=np.interp(t,times,e);energy-=energy[0]
        out.append(pd.DataFrame(dict(window_id=row.window_id,time_s=t-row.left_s,energy_Wh=energy)))
    pd.concat(out,ignore_index=True).to_parquet(R/'results/external_solar/native_reference_windows.parquet',index=False)

if __name__=='__main__':prepare_amov();prepare_solar();print('Prepared 247 AMOVFLY references and 161 solar windows. Run replay next.')
