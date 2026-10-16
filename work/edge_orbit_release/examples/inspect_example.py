import json
from pathlib import Path
import sys
import numpy as np

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'codec'))
from discrete_graph_prototype import expand,integer_tree

prototype=json.loads((root/'examples/mp-30009.prototype.json').read_text())
geometry=json.loads((root/'examples/mp-30009.geometry.json').read_text())
assert integer_tree(prototype)
positions,species,edges,cosets=expand(prototype,geometry['free_vectors'])
generic=np.random.default_rng(17).uniform(.05,.95,(len(prototype['nodes']),3))
other,other_species,other_edges,other_cosets=expand(prototype,generic)
assert edges==other_edges and cosets==other_cosets
assert np.array_equal(species,other_species)
print(json.dumps({'space_group':prototype['space_group'],'orbit_nodes':len(prototype['nodes']),
                  'expanded_atoms':len(positions),'edge_orbit_records':len(prototype['edge_tokens']),
                  'expanded_periodic_edges':len(edges),'integer_only_prototype':True,
                  'same_abstract_graph_with_independent_geometry':True,
                  'random_geometry_is_not_a_validity_or_stability_claim':True},indent=2))
