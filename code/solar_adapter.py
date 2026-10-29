"""Documented main-system power adapter; source field names precede values."""
from common4 import *
import h5py,numpy as np,pandas as pd

def main():
    reg=json.loads((ROOT/'audit/external_registration.json').read_text())
    for row in reg['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    D=ROOT/'data/external_solar'
    with h5py.File(D/'sysvector.mat','r') as f:
        refs=f['#subsystem#/MCOS'][()].ravel();names=[''.join(chr(int(c)) for c in f[r][()].ravel()) for r in f[refs[4]][()].ravel()];values=f[refs[5]][()].ravel();assert len(names)==len(values)==565
        chosen={name:names.index(name) for name in ['POWS_t','POWS_Main_Volt','POWS_Main_Cur']};assert chosen=={'POWS_t':512,'POWS_Main_Volt':513,'POWS_Main_Cur':514};a={name:np.asarray(f[values[index]][()]).ravel() for name,index in chosen.items()}
    frame=pd.DataFrame(dict(source_row=np.arange(len(a['POWS_t'])),time_s=a['POWS_t']/1e6,voltage_V=a['POWS_Main_Volt'],current_A=a['POWS_Main_Cur']));frame['power_W']=frame.voltage_V*frame.current_A;frame['valid_electrical']=np.isfinite(frame[['time_s','voltage_V','current_A']]).all(axis=1)&frame.voltage_V.gt(0)&frame.current_A.ge(0);frame.to_parquet(D/'native_power_with_flags.parquet',index=False)
    valid=frame[frame.valid_electrical].sort_values('time_s',kind='stable');dup=int(valid.duplicated('time_s').sum());conflicts=int((valid.groupby('time_s').power_W.nunique()>1).sum());valid=valid.drop_duplicates('time_s',keep='first');valid.to_parquet(D/'native_power.parquet',index=False);air=valid[valid.time_s.between(2449,295626)]
    report=dict(passed=bool(len(air)>2 and len(air)/max(1,len(frame[frame.time_s.between(2449,295626)]))>.5),source_rows=len(frame),valid_rows=int(frame.valid_electrical.sum()),invalid_rows=int((~frame.valid_electrical).sum()),negative_current_rows=int(frame.current_A.lt(0).sum()),duplicate_timestamps_removed=dup,conflicting_duplicate_timestamps=conflicts,unique_valid_samples=len(valid),airborne_samples=len(air),minimum_time_s=float(valid.time_s.min()),maximum_time_s=float(valid.time_s.max()),airborne_start_s=2449,airborne_end_s=295626,airborne_max_power_W=float(air.power_W.max()),airborne_min_power_W=float(air.power_W.min()),airborne_gaps_over_2s=int((np.diff(air.time_s)>2).sum()),fields=chosen,new_independent_campaigns=1,new_flight_collection_by_author=0,physical_calibration_provided=False)
    write(D/'adapter_gate.json',report);print(json.dumps(report,indent=2));assert report['passed'],'Eligibility gate failed; do not claim external performance.'

if __name__=='__main__':main()
