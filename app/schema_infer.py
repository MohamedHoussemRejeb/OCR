from typing import List, Dict, Any
from datetime import datetime
import re
from .models import ColumnSchema

NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
INT_RE = re.compile(r"^-?\d+$")
BOOL = {"true","false","0","1","yes","no","y","n","t","f"}
DATE_FORMATS = ["%Y-%m-%d","%d/%m/%Y","%m/%d/%Y","%Y/%m/%d","%d-%m-%Y","%m-%d-%Y",
                "%Y-%m-%d %H:%M:%S","%d/%m/%Y %H:%M","%m/%d/%Y %H:%M"]

def _is_date(s:str)->bool:
  s=s.strip()
  for f in DATE_FORMATS:
    try:
      datetime.strptime(s, f)
      return True
    except:
      pass
  try:
    datetime.fromisoformat(s.replace("Z","").replace("T"," "))
    return True
  except:
    return False

def infer_schema(rows: List[Dict[str, Any]], sample_size:int=200)->List[ColumnSchema]:
  if not rows: return []
  sample = rows[:sample_size]
  keys = list({k for r in sample for k in r.keys()})
  out: List[ColumnSchema] = []
  for k in keys:
    vals = ["" if v is None else str(v).strip() for v in (r.get(k) for r in sample)]
    non_empty = [v for v in vals if v!=""]
    n = len(vals) or 1; ne = len(non_empty)
    if ne==0:
      out.append(ColumnSchema(name=k, nullable=True)); continue
    num = sum(1 for v in non_empty if NUM_RE.match(v))
    integ = sum(1 for v in non_empty if INT_RE.match(v))
    boo = sum(1 for v in non_empty if v.lower() in BOOL)
    dat = sum(1 for v in non_empty if _is_date(v))
    card = len(set(non_empty))
    dtype="string"; conf=0.1
    if boo >= 0.8*ne: dtype,conf="boolean", boo/ne
    elif integ >= 0.5*ne: dtype,conf="integer", integ/ne
    elif num >= 0.5*ne: dtype,conf="number", num/ne
    elif dat >= 0.5*ne: dtype,conf="date", dat/ne
    elif card <= max(10, 0.3*ne): dtype,conf="categorical", min(0.9, max(0.3, 1-(card/ne)))
    out.append(ColumnSchema(name=k, type=dtype, nullable=(ne<n), confidence=round(conf,3)))
  return out
