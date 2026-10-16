"""Primitive/orbit parity under WP-tangent perturbation and GPU resource probe."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from orbit_graph_network import OrbitGraphNetwork


def convert(data, dtype):
    return {k: v.cuda().to(dtype if v.is_floating_point() else torch.long) for k, v in data.items()}


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    data = torch.load('graph_backbone/data.pt', map_location='cpu')
    d = convert(data['small'], torch.float64)
    model = OrbitGraphNetwork().cuda().double()
    results = {}
    for noise_scale in [0., .015]:
        noise = torch.randn_like(d['q']) * noise_scale
        observations = []
        for mode in ['full', 'orbit']:
            u = noise.clone().requires_grad_()
            model.zero_grad(set_to_none=True)
            pooled, force, norm = model(d, u, mode)
            loss = pooled.square().mean() + norm.mean()
            loss.backward()
            observations.append((pooled.detach(), force.detach()[d['reps']] if mode == 'full' else force.detach(),
                                 u.grad.clone(), torch.cat([p.grad.flatten() for p in model.parameters()])))
        names = ['pooled_output', 'ASU_vector_output', 'tangent_coordinate_gradient', 'parameter_gradient']
        error = {}
        for name, a, b in zip(names, observations[0], observations[1]):
            error[name] = float((a - b).abs().max())
            assert torch.allclose(a, b, rtol=1e-7, atol=1e-8), (noise_scale, name, error[name])
        results[str(noise_scale)] = error
    del d, model, observations, pooled, force, norm, loss, u, noise
    torch.cuda.empty_cache()
    d = convert(data['all'], torch.float32)
    model = OrbitGraphNetwork().cuda()
    u = (torch.randn_like(d['q']) * .015).requires_grad_()
    def step(mode):
        model.zero_grad(set_to_none=True)
        u.grad = None
        pooled, _, norm = model(d, u, mode)
        (pooled.square().mean() + norm.mean()).backward()
    timings = {mode: [] for mode in ['full','orbit']}
    peaks = {mode: [] for mode in ['full','orbit']}
    for repetition in range(4):
        order = ['full','orbit'] if repetition % 2 == 0 else ['orbit','full']
        for mode in order:
            for _ in range(3): step(mode)
            torch.cuda.synchronize()
            model.zero_grad(set_to_none=True); u.grad = None
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(10): step(mode)
            torch.cuda.synchronize()
            timings[mode].append((time.perf_counter() - start) / 10)
            peaks[mode].append(torch.cuda.max_memory_allocated())
    report = {'status':'COMPLETED', 'job_id': os.environ['SLURM_JOB_ID'], 'gpu':torch.cuda.get_device_name(),
              'parity_graphs': len(data['small']['lattice']), 'benchmark_graphs':len(data['all']['lattice']),
              'fp64_parity_errors':results, 'forward_backward_seconds':timings, 'peak_cuda_allocated_bytes':peaks,
              'primitive_nodes':len(d['orbit']), 'orbit_nodes':len(d['q']),
              'primitive_directed_edges':len(d['src']), 'orbit_directed_edge_classes':len(d['cs']),
              'parameter_count':sum(p.numel() for p in model.parameters()),
              'same_input_buffers_retained_in_memory_for_both_paths':True,
              'fixed_clean_neighbor_topology_during_tangent_perturbation':True,
              'not_trained_diffusion_or_generated_crystal_quality':True,
              'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                               ['prepare_graph_backbone.py','orbit_graph_network.py','benchmark_graph_backbone.py']}}
    Path('graph_backbone/benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
