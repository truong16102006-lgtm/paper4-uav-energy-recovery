"""Staged schema validation and complete native-power ledger for pinned AMOVFLY."""
from common4 import *
import numpy as np,pandas as pd,zipfile,io,argparse,re

D=ROOT/'data/external_amovfly'

def frame(archive,prefix,path):
    return pd.read_csv(archive.open(prefix+path),usecols=['time','battery_voltage','battery_current','power'])

def schema():
    acquired=json.loads((D/'acquisition.json').read_text());rows=[]
    with zipfile.ZipFile(D/'AMOVFLY_pinned.zip') as z:
        prefix=z.namelist()[0]
        for aircraft,path in acquired['schema_validation_records'].items():
            f=frame(z,prefix,path);I=pd.to_numeric(f.battery_current,errors='coerce').to_numpy();V=pd.to_numeric(f.battery_voltage,errors='coerce').to_numpy();P=pd.to_numeric(f.power,errors='coerce').to_numpy();valid=np.isfinite(I)&np.isfinite(V)&(V>0)&(abs(I)>1e-9);positive=float((I[valid]>0).mean());sign=1 if positive>=.95 else (-1 if positive<=.05 else 0)
            rows.append(dict(aircraft=aircraft,path=path,rows=len(f),nonzero_current_rows=int(valid.sum()),positive_current_fraction=positive,orientation=sign,max_provided_power_vs_VI_W=float(np.nanmax(abs(P-V*I))),max_abs_power_W=float(np.nanmax(abs(V*I)))))
    signs={x['orientation'] for x in rows};assert len(signs)==1 and 0 not in signs,'Unresolved discharge convention: retain ambiguity; do not choose by reconstruction outcome.'
    report=dict(passed=True,phase='Only the three preselected schema records decoded; no reconstruction scores computed',orientation=signs.pop(),schema_records=rows,remaining_ready_records_not_decoded=274,physical_calibration_established=False)
    write(D/'schema_gate.json',report);print(json.dumps(report,indent=2))

def all_records():
    reg=json.loads((ROOT/'audit/robust_external_registration.json').read_text())
    for item in reg['files']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    acquired=json.loads((D/'acquisition.json').read_text());choice=json.loads((D/'schema_gate.json').read_text());orientation=choice['orientation'];reserved=set(acquired['schema_validation_records'].values());tree=json.loads((D/'source_tree.json').read_text());files=[x['path'] for x in tree['tree'] if x['type']=='blob' and x['path'].endswith('.csv') and len(x['path'].split('/'))==3 and x['path'].split('/')[0]==x['path'].split('/')[1]];ledger=[];native=[];flags=[]
    with zipfile.ZipFile(D/'AMOVFLY_pinned.zip') as z:
        prefix=z.namelist()[0];meta=pd.read_csv(z.open(prefix+'Flight_info.csv'),usecols=['Data Dir','Date time','FlightName','Uav','BatteryName'],dtype=str);meta['basename']=meta.FlightName.str.replace('\\','/',regex=False).str.split('/').str[-1].str.replace('.csv','',regex=False);meta['parsed_date']=pd.to_datetime(meta['Date time'],errors='coerce',format='mixed');meta.to_csv(D/'non_electrical_flight_metadata.csv',index=False)
        for path in sorted(files):
            f=frame(z,prefix,path);f=f.apply(pd.to_numeric,errors='coerce');t=f.time.to_numpy();V=f.battery_voltage.to_numpy();I=f.battery_current.to_numpy();P=orientation*V*I;valid=np.isfinite(t)&np.isfinite(V)&np.isfinite(I)&(V>0)&(orientation*I>=0);name=path.split('/')[-1];fid=path.replace('/','__');aircraft=name.split('_')[0];matches=meta[meta.basename.eq(name[:-4])];parsed=matches.parsed_date.iloc[0] if len(matches)==1 else pd.NaT;date=str(parsed.date()) if not pd.isna(parsed) else 'unknown';date_time=parsed.isoformat() if not pd.isna(parsed) else '';role='SCHEMA' if path in reserved else 'TEST'
            raw=pd.DataFrame(dict(record_id=fid,row=np.arange(len(f)),time_s=t,voltage_V=V,current_A=I,power_W=P,valid=valid));flags.append(raw);v=raw[valid].sort_values('time_s',kind='stable');conflicts=int((v.groupby('time_s').power_W.nunique()>1).sum());duplicates=int(v.time_s.duplicated().sum());v=v.drop_duplicates('time_s',keep='first');status='ELIGIBLE'
            if len(v)<3:status='INSUFFICIENT_VALID_SAMPLES'
            elif v.time_s.iloc[-1]<=v.time_s.iloc[0]:status='NONPOSITIVE_DURATION'
            elif np.any(np.diff(v.time_s)>2):status='REFERENCE_SUPPORT_GAP'
            row=dict(record_id=fid,path=path,aircraft=aircraft,scenario=path.split('/')[0],role=role,date=date,date_time=date_time,metadata_matches=len(matches),source_rows=len(f),invalid_rows=int((~valid).sum()),duplicates=duplicates,duplicate_conflicts=conflicts,native_samples=len(v),status=status,duration_s=float(v.time_s.iloc[-1]-v.time_s.iloc[0]) if len(v) else 0.,max_power_W=float(v.power_W.max()) if len(v) else None)
            ledger.append(row)
            if status=='ELIGIBLE':
                native.append(pd.DataFrame(dict(record_id=fid,time_s=v.time_s.to_numpy()-v.time_s.iloc[0],power_W=v.power_W.to_numpy())))
    pd.DataFrame(ledger).to_csv(D/'flight_ledger.csv',index=False);pd.concat(flags,ignore_index=True).to_parquet(D/'native_with_flags.parquet',index=False);pd.concat(native,ignore_index=True).to_parquet(D/'native_power.parquet',index=False);l=pd.DataFrame(ledger);report=dict(passed=True,candidates=len(l),schema_records=3,test_candidates=int(l.role.eq('TEST').sum()),eligible_test_records=int((l.role.eq('TEST')&l.status.eq('ELIGIBLE')).sum()),status_by_role=l.groupby('role').status.value_counts().to_dict().__str__(),aircraft_ids=sorted(l.aircraft.unique().tolist()),matched_metadata=int(l.metadata_matches.eq(1).sum()),collection_dates=sorted(set(l.date)-{'unknown'}),unknown_date_records=int(l.date.eq('unknown').sum()),source_rows=int(l.source_rows.sum()),invalid_rows=int(l.invalid_rows.sum()),duplicate_rows=int(l.duplicates.sum()),duplicate_conflicts=int(l.duplicate_conflicts.sum()),orientation=orientation,author_collected_flights=0);write(D/'adapter_gate.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['schema','all']);args=p.parse_args();schema() if args.phase=='schema' else all_records()
