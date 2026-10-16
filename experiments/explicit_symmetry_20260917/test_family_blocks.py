"""Shared explicit coordinate template, without rule IDs or hidden dictionary."""
import os
import numpy as np
import program_codec as codec


assert os.environ.get('SLURM_JOB_ID')
program='SP1 CELL 4 4 4 90 90 90 ROUND 6 GENERATORS 0 SITES 2 FAMILY 2 y PARAM 1/4 C .2 N .4 RESIDUALS 0 END'
result=codec.decode(program)
assert [str(s.specie) for s in result]==['C','N']
assert np.allclose(result.frac_coords,[[.2,.2,.25],[.4,.4,.25]])
print('Explicit family template shares its rule, not parameter values; passed')
