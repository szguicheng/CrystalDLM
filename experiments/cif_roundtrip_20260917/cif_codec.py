"""CIF <-> symmetry edge-orbit records plus continuous geometry.

No standardization or idealization. The decoded structure uses only the JSON.
Scope: ordered, fully occupied sites without magnetic or CIF metadata fields.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import spglib
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser, CifWriter

from symmetry_graph_codec import encode, edge_key, wrap
from discrete_graph_prototype import split, expand


def neighbors(structure, radius):
    i,j,images,distances = structure.get_neighbor_list(radius, numerical_tol=1e-8)
    return {edge_key(a,b,n) for a,b,n,d in zip(i,j,images,distances) if d>1e-8}


def encode_cif(path, radius=4.5, symprec=1e-7):
    # Disable pymatgen's optional rounding of fractional values to rational sites.
    crystal = CifParser(str(path), frac_tolerance=0).parse_structures(primitive=False)[0]
    if not crystal.is_ordered:
        raise ValueError('This minimal codec requires ordered fully occupied sites')
    cell = (crystal.lattice.matrix, crystal.frac_coords, crystal.atomic_numbers)
    symmetry = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if symmetry is None:
        raise ValueError('spglib could not determine symmetry in the input cell')
    encoded,_ = encode(crystal.frac_coords, np.array(crystal.atomic_numbers),
                       symmetry.rotations, symmetry.translations,
                       symmetry.equivalent_atoms, symmetry.wyckoffs,
                       neighbors(crystal,radius), crystal.lattice.matrix,
                       symmetry.number, symmetry.hall_number)
    # The existing split codec consumes JSON lists; encode emits Python tuples.
    prototype,geometry = split(json.loads(json.dumps(encoded)))
    return {'format':'edge-orbit-cif-v1', 'radius_A':radius,
            'prototype':prototype, 'geometry':geometry}


def decode_bundle(bundle):
    if 'geometry' not in bundle:
        raise ValueError('Discrete prototype alone does not determine a unique crystal: lattice and free coordinates are required')
    geometry = bundle['geometry']
    positions,species,edges,_ = expand(bundle['prototype'],geometry['free_vectors'])
    wrapped = wrap(positions)
    offsets = np.rint(positions-wrapped).astype(int)
    wrapped_edges = {edge_key(i,j,np.array(n)+offsets[j]-offsets[i]) for i,j,*n in edges}
    crystal = Structure(geometry['lattice'],species,wrapped)
    return crystal,wrapped_edges


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    enc=sub.add_parser('encode',help='CIF -> self-contained JSON with prototype and geometry')
    enc.add_argument('input'); enc.add_argument('output')
    enc.add_argument('--radius',type=float,default=4.5)
    enc.add_argument('--symprec',type=float,default=1e-7)
    dec=sub.add_parser('decode',help='JSON -> structural CIF (all atoms explicitly listed)')
    dec.add_argument('input'); dec.add_argument('output')
    args=parser.parse_args()
    if args.command=='encode':
        bundle=encode_cif(args.input,args.radius,args.symprec)
        Path(args.output).write_text(json.dumps(bundle,separators=(',',':'),sort_keys=True))
    else:
        bundle=json.loads(Path(args.input).read_text())
        crystal,_=decode_bundle(bundle)
        # P1 is the output listing convention; the expanded structure retains symmetry.
        CifWriter(crystal,symprec=None,significant_figures=12).write_file(args.output)


if __name__=='__main__':
    main()
