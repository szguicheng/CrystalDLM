"""Frozen rule dictionary + actual positional token stream, on the same 10 CIFs."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np

from cif_codec import encode_cif,decode_bundle
from symmetry_graph_codec import edge_key
from test_roundtrip import parse,compare

ROOT=Path(__file__).resolve().parent
CLI=ROOT/'fixed_dictionary.py'


def count_cif(path):
    lexer=shlex.shlex(path.read_text(),posix=True)
    lexer.whitespace_split=True; lexer.commenters='#'
    return len(list(lexer))


def invoke(*args):
    result=subprocess.run([sys.executable,str(CLI),*map(str,args)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def fields(value):
    if isinstance(value,dict): return sum(map(fields,value.values()))
    if isinstance(value,list): return sum(map(fields,value))
    return 1


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use allocated compute node'
    assert CLI.exists(),'Shared dictionary and token stream codec not implemented'
    from fixed_dictionary import build_dictionary,pack,unpack,dumps,loads
    started=time.perf_counter()
    selected=json.loads((ROOT/'results/selection.json').read_text())['materials']
    bundles=[json.loads((ROOT/'results'/m['id']/'encoded.json').read_text()) for m in selected]
    out=ROOT/'fixed_dictionary_results'
    out.mkdir(exist_ok=True)
    dictionary=build_dictionary(bundles)
    dictionary_path=out/'rules.json'
    dictionary_path.write_text(json.dumps(dictionary,separators=(',',':'),sort_keys=True))
    frozen=dictionary_path.read_bytes()
    # Dictionary depends only on crystallographic rules; elements, edges and
    # continuous geometry must neither enter the table nor change its identity.
    altered=json.loads(json.dumps(bundles))
    for b in altered:
        b['geometry']['lattice']=(np.array(b['geometry']['lattice'])*1.02).tolist()
        b['geometry']['free_vectors']=(np.array(b['geometry']['free_vectors'])+.003).tolist()
        b['prototype']['edge_tokens']=[]
        for n in b['prototype']['nodes']: n['Z']=1
    assert build_dictionary(altered)==dictionary
    assert build_dictionary(list(reversed(bundles)))==dictionary
    unknown=json.loads(json.dumps(bundles[0]))
    unknown['prototype']['nodes'][0]['wyckoff_letter']='unseen-rule'
    try: pack(unknown,dictionary)
    except ValueError as e: assert 'rule' in str(e).lower()
    else: raise AssertionError('Unknown rule silently added to frozen dictionary')
    records=[]
    for material,old in zip(selected,bundles):
        mid=material['id']; case=out/mid; case.mkdir(exist_ok=True)
        source=ROOT/'results'/mid/'input.cif'
        target=case/'decoded.cif'; stream=case/'encoded.tokens'; repeated=case/'reencoded.tokens'
        # Start from original CIF, not just from the cached JSON.
        invoke('encode',source,stream,'--dictionary',dictionary_path)
        tokens=loads(stream.read_text()); recovered=unpack(tokens,dictionary)
        original=parse(source)
        expanded,new_edges=decode_bundle(recovered)
        old_expanded,old_edges=decode_bundle(old)
        compare(original,expanded)
        assert old_edges==new_edges
        assert old['prototype']==recovered['prototype']
        assert np.allclose(old['geometry']['lattice'],recovered['geometry']['lattice'],atol=0,rtol=0)
        wrong=json.loads(json.dumps(dictionary)); wrong['id']='wrong-version'
        try: unpack(tokens,wrong)
        except ValueError as e: assert 'dictionary' in str(e).lower()
        else: raise AssertionError('Decoder accepted wrong dictionary version')
        # Independent decode sees only stream and dictionary, not source CIF.
        invoke('decode',stream,target,'--dictionary',dictionary_path)
        rebuilt=parse(target)
        _,_,error,lattice_error=compare(original,rebuilt)
        invoke('encode',target,repeated,'--dictionary',dictionary_path)
        again=unpack(loads(repeated.read_text()),dictionary)
        again_structure,again_edges=decode_bundle(again)
        compare(original,again_structure)
        assert again_edges==new_edges
        assert dictionary_path.read_bytes()==frozen
        assert dumps(pack(recovered,dictionary))==stream.read_text()
        assert repeated.read_bytes()==stream.read_bytes()
        proto=old['prototype']; n=len(proto['nodes']); e=len(proto['edge_tokens'])
        d=sum(node['dof'] for node in proto['nodes'])
        assert len(tokens)==6+9+2*n+d+6*e
        old_fields=fields(proto)+fields(old['geometry'])
        records.append({'id':mid,'formula':material['formula'],'space_group':proto['space_group'],'input_cif_tokens':count_cif(source),'old_structural_fields':old_fields,
          'new_structural_fields':1+9+2*n+d+6*e,'actual_stream_tokens':len(tokens),'decoded_cif_tokens':count_cif(target),
          'site_records':n,'edge_records':e,'free_coordinate_values':d,'global_rule_tokens':1,'lattice_values':9,'site_reference_tokens':2*n,'edge_tokens':6*e,'framing_tokens':5,
          'maximum_coordinate_error_A':error,'maximum_lattice_error_A':lattice_error,'edge_loss':0,'reencode_identical':True,'stream_bytes':stream.stat().st_size})
        (case/'recovered.json').write_text(json.dumps(recovered,separators=(',',':'),sort_keys=True))
        print(json.dumps(records[-1]),flush=True)
    summary={'passed':len(records),'selected':len(selected),'dictionary_id':dictionary['id'],'dictionary_bytes':len(frozen),
      'global_rules':len(dictionary['groups']),'site_rules':len(dictionary['sites']),'dictionary_sha256':hashlib.sha256(frozen).hexdigest(),
      'frozen_during_tests':True,'dictionary_has_no_elements_edges_or_continuous_geometry':True,'unknown_rule_rejected':True,'wrong_dictionary_rejected':True,
      'scope':'Dictionary built from the symbolic rules of the same 10 cases; reconstruction test, not unseen-rule/generalization evaluation. Float values are continuous scalar slots, not finite-vocabulary quantization.',
      'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.perf_counter()-started,
      'total_original_fields':sum(r['old_structural_fields'] for r in records),'total_new_structural_fields':sum(r['new_structural_fields'] for r in records),
      'total_actual_stream_tokens':sum(r['actual_stream_tokens'] for r in records),'total_cif_tokens':sum(r['input_cif_tokens'] for r in records),
      'stream_bytes':sum(r['stream_bytes'] for r in records),
      'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
    (out/'records.json').write_text(json.dumps(records,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
