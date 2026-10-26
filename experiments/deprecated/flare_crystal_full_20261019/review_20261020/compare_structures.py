"""Compare preserved generated structures with the benchmark's fixed matcher."""
from collections import Counter
import json
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
SETTINGS=dict(ltol=.3,stol=.5,angle_tol=10,primitive_cell=True,scale=True,attempt_supercell=False)
MATCHER=StructureMatcher(**SETTINGS)


def item(arm,receipt,structure):
    return dict(key=f"{arm}/{receipt['request_id']}/{receipt['attempt_id']:03d}",
                formula=structure.composition.reduced_formula,structure=structure,
                sg=receipt.get('actual_space_group_number_post_relax',receipt.get('actual_space_group_number')),
                ehull=receipt.get('ehull_proxy_corrected_eV_per_atom'))


def same(a,b):
    return a['formula']==b['formula'] and MATCHER.fit(a['structure'],b['structure'])


def dedup(rows):
    representatives=[]
    duplicates=[]
    for row in rows:
        prior=next((r for r in representatives if same(row,r)),None)
        if prior is None:representatives.append(row)
        else:duplicates.append([row['key'],prior['key']])
    return representatives,duplicates


def compare(a,b):
    ua,da=dedup(a);ub,db=dedup(b)
    cross=[]
    for x in ua:
        for y in ub:
            if same(x,y):cross.append(dict(ar=x['key'],flare=y['key'],formula=x['formula'],
                                           ar_sg=x['sg'],flare_sg=y['sg'],ar_ehull=x['ehull'],flare_ehull=y['ehull']))
    union,_=dedup(ua+ub)
    reverse,_=dedup(ub+ua)
    return dict(ar_n=len(a),flare_n=len(b),ar_unique=len(ua),flare_unique=len(ub),
        ar_duplicate_pairs=da,flare_duplicate_pairs=db,cross_pairs=cross,
        ar_unique_with_cross_match=len({r['ar'] for r in cross}),
        flare_unique_with_cross_match=len({r['flare'] for r in cross}),
        union_greedy_ar_first=len(union),union_greedy_flare_first=len(reverse))


def main():
    datasets={name:{} for name in ['original_valid','relaxed_converged_valid','final_novel_80meV']}
    raw_texts={}
    formula_counts={}
    for arm in ['ar','flare']:
        root=BASE/f'{arm}_evaluation'
        generated=[json.loads(p.read_text()) for p in sorted(root.glob('shard-*/attempts/*/*.json'))]
        scored=[json.loads(p.read_text()) for p in sorted((root/'grace_score/attempts').glob('*/*.json'))]
        assert len(generated)==len(scored)==700
        raw_texts[arm]=Counter(r.get('crystal_raw_text','') for r in generated if r.get('crystal_raw_text'))
        original=[]
        for r in generated:
            if not r['structural_valid']:continue
            path=Path(r['cif']['path'])
            if not path.is_absolute():path=root/path
            original.append(item(arm,r,Structure.from_file(path)))
        relaxed=[];final=[]
        for r in scored:
            if r.get('converged') and r.get('post_relax_valid') and r.get('relaxed_structure'):
                entry=item(arm,r,Structure.from_dict(r['relaxed_structure']))
                relaxed.append(entry)
                if .08 in r.get('global_unique_novel_thresholds_eV_per_atom',[]):final.append(entry)
        datasets['original_valid'][arm]=original
        datasets['relaxed_converged_valid'][arm]=relaxed
        datasets['final_novel_80meV'][arm]=final
        formula_counts[arm]=dict(Counter(x['formula'] for x in final))
    assert len(datasets['final_novel_80meV']['ar'])==25
    assert len(datasets['final_novel_80meV']['flare'])==24
    assert same(datasets['original_valid']['ar'][0],datasets['original_valid']['ar'][0])
    results={'matcher':SETTINGS,'interpretation':'matches under fixed benchmark tolerances; not byte identity',
             'raw_text':{'ar_unique':len(raw_texts['ar']),'flare_unique':len(raw_texts['flare']),
                         'distinct_common_strings':len(raw_texts['ar'].keys()&raw_texts['flare'].keys())},
             'final_formula_counts':formula_counts,'stages':{}}
    for name,arms in datasets.items():
        results['stages'][name]=compare(arms['ar'],arms['flare'])
        r=results['stages'][name]
        print(name,{k:r[k] for k in ['ar_n','flare_n','ar_unique','flare_unique',
            'ar_unique_with_cross_match','flare_unique_with_cross_match','union_greedy_ar_first','union_greedy_flare_first']},flush=True)
    (HERE/'structure-overlap.json').write_text(json.dumps(results,indent=2))
    print('final matched formulas',[p['formula'] for p in results['stages']['final_novel_80meV']['cross_pairs']])


if __name__=='__main__':main()
