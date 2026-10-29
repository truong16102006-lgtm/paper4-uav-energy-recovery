"""Canonical CLI: retain numeric audit and normalize malformed-CSV errors."""
import csv,json,sys
from audit_counter import main

if __name__=='__main__':
    try:raise SystemExit(main())
    except (csv.Error,TypeError,OverflowError) as exc:
        print(json.dumps({'status':'INVALID_INPUT','error':str(exc)}),file=sys.stderr)
        raise SystemExit(2)
