"""Isolate edge coalescing from previously known ASU node compression."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
from orbit_graph_network import OrbitGraphNetwork
from compared_graph_network import ComparedGraphNetwork
from benchmark_graph_backbone import convert


def add_asu(d):
    d = dict(d)
    keep = torch.isin(d['src'], d['reps'])
    target = d['dst'][keep]
    d['ausrc'] = d['orbit'][d['src'][keep]]
    d['audst'] = d['orbit'][target]
    d['auA'] = d['A'][target]
    d['aub'] = d['b'][target] + d['image'][keep]
    degree = torch.zeros(len(d['q']), dtype=d['q'].dtype).index_add(0,d['ausrc'],torch.ones(len(target),dtype=d['q'].dtype))
    quotient_degree = torch.zeros_like(degree).index_add(0,d['cs'],d['count'])
    assert torch.equal(degree,quotient_degree)
    return d


def observe(model,d,u,mode):
    model.zero_grad(set_to_none=True)
    u = u.clone().requires_grad_()
    pooled,force,norm = model(d,u,mode)
    (pooled.square().mean()+norm.mean()).backward()
    return [pooled.detach(),force.detach()[d['reps']] if mode=='full' else force.detach(),
            u.grad.clone(),torch.cat([p.grad.flatten() for p in model.parameters()])]


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2)
    raw = torch.load('graph_backbone/data.pt',map_location='cpu')
    assert ComparedGraphNetwork.__init__ is OrbitGraphNetwork.__init__
    parity = {}
    for split,dtype in [('small',torch.float64),('all',torch.float32)]:
        d = convert(add_asu(raw[split]),dtype)
        torch.manual_seed(43)
        model = ComparedGraphNetwork().cuda().to(dtype)
        checks = {}
        for scale in [0.,.015]:
            u = torch.randn_like(d['q'])*scale
            reference = observe(model,d,u,'full')
            for mode in ['asu','orbit']:
                actual = observe(model,d,u,mode)
                errors = {}
                for name,a,b in zip(['pooled','vector','coordinate_gradient','parameter_gradient'],reference,actual):
                    atol,rtol = (1e-8,1e-7) if dtype==torch.float64 else (3e-5,3e-4)
                    assert torch.allclose(a,b,atol=atol,rtol=rtol),(split,scale,mode,name,float((a-b).abs().max()))
                    errors[name] = float((a-b).abs().max())
                checks[f'{mode}_{scale}'] = errors
        parity[split] = checks
        del model,d,reference,actual,u,a,b
        torch.cuda.empty_cache()
    timings,peaks,counts = {},{},{}
    for split in ['small','all']:
        d = convert(add_asu(raw[split]),torch.float32)
        torch.manual_seed(43)
        model = ComparedGraphNetwork().cuda()
        u = (torch.randn_like(d['q'])*.015).requires_grad_()
        def step(mode):
            model.zero_grad(set_to_none=True); u.grad=None
            pooled,_,norm = model(d,u,mode)
            (pooled.square().mean()+norm.mean()).backward()
        times={m:[] for m in ['full','asu','orbit']}
        memory={m:[] for m in times}
        for repeat in range(4):
            order=['full','asu','orbit'] if repeat%2==0 else ['orbit','asu','full']
            for mode in order:
                for _ in range(3):step(mode)
                torch.cuda.synchronize()
                model.zero_grad(set_to_none=True);u.grad=None
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                for _ in range(10):step(mode)
                torch.cuda.synchronize()
                times[mode].append((time.perf_counter()-start)/10)
                memory[mode].append(torch.cuda.max_memory_allocated())
        timings[split]=times;peaks[split]=memory
        counts[split]={'graphs':len(d['lattice']),'full_nodes':len(d['orbit']),'asu_nodes':len(d['q']),
                       'full_edges':len(d['src']),'asu_occurrence_edges':len(d['ausrc']),
                       'edge_orbit_classes':len(d['cs'])}
        del model,d,u
        torch.cuda.empty_cache()
    derived={s:{'orbit_speedup_over_asu':statistics.median(t['asu'])/statistics.median(t['orbit']),
                'orbit_memory_reduction_from_asu':1-peaks[s]['orbit'][0]/peaks[s]['asu'][0],
                'median_seconds':{m:statistics.median(v) for m,v in t.items()}} for s,t in timings.items()}
    report={'status':'COMPLETED','job_id':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name(),
            'parity':parity,'seconds':timings,'peak_allocated_bytes':peaks,'sizes':counts,'derived':derived,
            'shared_constructor':True,'control_scope':'ASU neighbor-occurrence granularity, not a full coGN architecture reproduction',
            'fixed_topology_only':True,'trained_generative_model':False,
            'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                             ['orbit_graph_network.py','compared_graph_network.py','benchmark_asu_control.py']}}
    Path('graph_backbone/asu_control.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
