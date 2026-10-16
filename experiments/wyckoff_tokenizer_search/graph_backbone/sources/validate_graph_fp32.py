"""All-graph FP32 parity and non-vacuous geometric response checks."""
import json
import os
from pathlib import Path
import torch
from orbit_graph_network import OrbitGraphNetwork
from benchmark_graph_backbone import convert


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(29)
    d = convert(torch.load('graph_backbone/data.pt', map_location='cpu')['all'], torch.float32)
    model = OrbitGraphNetwork().cuda()
    ranks = d['projection'].diagonal(dim1=-2, dim2=-1).sum(-1).round().long()
    noise = torch.randn_like(d['q']) * .015
    assert int((ranks > 0).sum()) > 0
    projected = torch.einsum('nij,nj->ni', d['projection'], noise)
    assert projected.abs().max() > .001
    checks, clean = {}, None
    for label, perturbation in [('clean', torch.zeros_like(noise)), ('perturbed', noise)]:
        outputs = []
        full_force = None
        for mode in ['full','orbit']:
            u = perturbation.clone().requires_grad_()
            model.zero_grad(set_to_none=True)
            pooled, force, norm = model(d, u, mode)
            (pooled.square().mean() + norm.mean()).backward()
            if mode == 'full': full_force = force.detach()
            outputs.append((pooled.detach(), force.detach()[d['reps']] if mode == 'full' else force.detach(),
                            u.grad.clone(), torch.cat([p.grad.flatten() for p in model.parameters()])))
        errors = {}
        for key, a, b in zip(['pooled','representative_vector','coordinate_gradient','parameter_gradient'], *outputs):
            errors[key] = {'max_abs_difference':float((a-b).abs().max()), 'reference_max_abs':float(a.abs().max())}
            assert torch.allclose(a,b,atol=3e-5,rtol=3e-4), (label,key,errors[key])
        l = d['lattice'][d['full_graph']].transpose(-1,-2)
        cartesian_rotations = l @ d['A'] @ torch.linalg.inv(l)
        expanded_force = torch.einsum('nij,nj->ni',cartesian_rotations,outputs[1][1][d['orbit']])
        assert torch.allclose(full_force,expanded_force,atol=3e-5,rtol=3e-4)
        errors['all_node_vector_equivariance_max_abs'] = float((full_force-expanded_force).abs().max())
        checks[label] = errors
        if clean is None:
            clean = [x.clone() for x in outputs[0][:2]]
        else:
            response = {'pooled_max_abs_change':float((clean[0]-outputs[0][0]).abs().max()),
                        'vector_max_abs_change':float((clean[1]-outputs[0][1]).abs().max())}
            assert response['pooled_max_abs_change'] > 1e-7 and response['vector_max_abs_change'] > 1e-7
    report = {'status':'COMPLETED','job_id':os.environ['SLURM_JOB_ID'], 'graphs':len(d['lattice']),
              'site_dof_histogram':torch.bincount(ranks,minlength=4).tolist(),
              'projected_noise_max_abs_fractional':float(projected.abs().max()),
              'nonzero_geometric_response':response,'FP32_parity':checks,
              'fixed_topology_only':True}
    Path('graph_backbone/fp32_parity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
