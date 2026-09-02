import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("gate",Path(__file__).parents[1]/"scripts/validate_generalist_g0_gate.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def _inputs():
    edges=[{"from":a,"to":b,"reset_count":0,"success":True,"metrics":{"finite":True}} for a,b in mod.LEGAL_EDGES]
    behaviors=[{"behavior":x,"success":True,"metrics":{"finite":True}} for x in ("stand","locomotion","sit_stand")]
    ev={"finite":True,"legal_edges":edges,"unsupported_edges":[{"from":a,"to":b,"exercised":False} for a,b in mod.UNSUPPORTED_EDGES],"behaviors":behaviors}
    return ev,{"passed":True},{"passed":True},{"preserved":True}

def test_acceptance_requires_all_supporting_gates():
    result=mod.validate(*_inputs()); assert result["accepted"] and result["status"]=="ACCEPT"

def test_behavior_failure_is_diagnostic():
    values=list(_inputs()); values[0]["behaviors"][0]["metrics"]["finite"]=False
    result=mod.validate(*values); assert result["status"]=="DIAGNOSTIC_FAIL"; assert not result["accepted"]
