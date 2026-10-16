# Deterministic inline binding fixture

This is generated from frozen source facts without an LLM call. It demonstrates binding and is not a generated description or an LLM result.

## Placeholder prose

The input cell has edge lengths [[param:cell.a]], [[param:cell.b]], [[param:cell.c]] angstrom and angles [[param:cell.alpha]], [[param:cell.beta]], [[param:cell.gamma]] degrees.
Its symmetry is space group [[param:space_group.number]] in Hall setting [[param:space_group.hall]].
The cell contains [[param:input.atom_count]] atoms in total.
The declared element count is [[param:composition.Al]].
The declared element count is [[param:composition.Cu]].
The declared element count is [[param:composition.In]].
The declared element count is [[param:composition.S]].
Independent orbit [[param:orbit.o000.id]] contains element [[param:orbit.o000.element]] at Wyckoff position [[param:orbit.o000.wyckoff]].
Independent orbit [[param:orbit.o001.id]] contains element [[param:orbit.o001.element]] at Wyckoff position [[param:orbit.o001.wyckoff]].
Independent orbit [[param:orbit.o002.id]] contains element [[param:orbit.o002.element]] at Wyckoff position [[param:orbit.o002.wyckoff]].
Independent orbit [[param:orbit.o003.id]] contains element [[param:orbit.o003.element]] at Wyckoff position [[param:orbit.o003.wyckoff]].
Independent orbit [[param:orbit.o004.id]] contains element [[param:orbit.o004.element]] at Wyckoff position [[param:orbit.o004.wyckoff]] with standard-frame free coordinates [[param:orbit.o004.parameters]].
The standard fractional frame is obtained from the input frame by applying matrix [[param:frame.matrix]] and adding origin shift [[param:frame.origin_shift]].
Input fractional coordinates are rounded to [[param:round_digits]] decimal places before corrections.
Precision correction for retained site [[param:correction.c000.site_id]] of element [[param:correction.c000.element]] is the input-frame fractional displacement [[param:correction.c000.delta]].
Precision correction for retained site [[param:correction.c001.site_id]] of element [[param:correction.c001.element]] is the input-frame fractional displacement [[param:correction.c001.delta]].

## Bound inline document

The input cell has edge lengths <param name="cell.a" type="float64" unit="angstrom" frame="input">5.474408</param>, <param name="cell.b" type="float64" unit="angstrom" frame="input">5.474408</param>, <param name="cell.c" type="float64" unit="angstrom" frame="input">6.66096313</param> angstrom and angles <param name="cell.alpha" type="float64" unit="degree" frame="input">65.7365946</param>, <param name="cell.beta" type="float64" unit="degree" frame="input">65.7365946</param>, <param name="cell.gamma" type="float64" unit="degree" frame="input">90.0</param> degrees.
Its symmetry is space group <param name="space_group.number" type="int" unit="identifier" frame="public">82</param> in Hall setting <param name="space_group.hall" type="string" unit="Hall_symbol" frame="public">"I -4"</param>.
The cell contains <param name="input.atom_count" type="int" unit="atom" frame="input">8</param> atoms in total.
The declared element count is <param name="composition.Al" type="composition_count" unit="atom" frame="input">{"element":"Al","count":1}</param>.
The declared element count is <param name="composition.Cu" type="composition_count" unit="atom" frame="input">{"element":"Cu","count":2}</param>.
The declared element count is <param name="composition.In" type="composition_count" unit="atom" frame="input">{"element":"In","count":1}</param>.
The declared element count is <param name="composition.S" type="composition_count" unit="atom" frame="input">{"element":"S","count":4}</param>.
Independent orbit <param name="orbit.o000.id" type="string" unit="orbit_identifier" frame="public_expansion">"o000"</param> contains element <param name="orbit.o000.element" type="string" unit="element" frame="orbit">"Al"</param> at Wyckoff position <param name="orbit.o000.wyckoff" type="string" unit="Wyckoff_label" frame="public_table">"2c"</param>.
Independent orbit <param name="orbit.o001.id" type="string" unit="orbit_identifier" frame="public_expansion">"o001"</param> contains element <param name="orbit.o001.element" type="string" unit="element" frame="orbit">"In"</param> at Wyckoff position <param name="orbit.o001.wyckoff" type="string" unit="Wyckoff_label" frame="public_table">"2b"</param>.
Independent orbit <param name="orbit.o002.id" type="string" unit="orbit_identifier" frame="public_expansion">"o002"</param> contains element <param name="orbit.o002.element" type="string" unit="element" frame="orbit">"Cu"</param> at Wyckoff position <param name="orbit.o002.wyckoff" type="string" unit="Wyckoff_label" frame="public_table">"2d"</param>.
Independent orbit <param name="orbit.o003.id" type="string" unit="orbit_identifier" frame="public_expansion">"o003"</param> contains element <param name="orbit.o003.element" type="string" unit="element" frame="orbit">"Cu"</param> at Wyckoff position <param name="orbit.o003.wyckoff" type="string" unit="Wyckoff_label" frame="public_table">"2a"</param>.
Independent orbit <param name="orbit.o004.id" type="string" unit="orbit_identifier" frame="public_expansion">"o004"</param> contains element <param name="orbit.o004.element" type="string" unit="element" frame="orbit">"S"</param> at Wyckoff position <param name="orbit.o004.wyckoff" type="string" unit="Wyckoff_label" frame="public_table">"8g"</param> with standard-frame free coordinates <param name="orbit.o004.parameters" type="float64[3]" unit="fractional" frame="standard">[0.23178300000000007,0.757471,0.13320900000000008]</param>.
The standard fractional frame is obtained from the input frame by applying matrix <param name="frame.matrix" type="float64[3,3]" unit="1" frame="standard_from_input">[[-7.141553772848957e-18,-1.0,-0.5],[1.0,-4.836959745840887e-17,0.5],[7.141553772848957e-18,-2.0614021842779957e-17,0.5]]</param> and adding origin shift <param name="frame.origin_shift" type="float64[3]" unit="fractional" frame="standard">[-5.551115123125783e-17,5.551115123125783e-17,5.551115123125783e-17]</param>.
Input fractional coordinates are rounded to <param name="round_digits" type="null|int" unit="decimal_places" frame="input_fractional">6</param> decimal places before corrections.
Precision correction for retained site <param name="correction.c000.site_id" type="string" unit="expanded_site_identifier" frame="input_cell">"o004_1"</param> of element <param name="correction.c000.element" type="string" unit="element" frame="input_site">"S"</param> is the input-frame fractional displacement <param name="correction.c000.delta" type="float64[3]" unit="fractional" frame="input_delta">[-1.0000000000287557e-06,0.0,0.0]</param>.
Precision correction for retained site <param name="correction.c001.site_id" type="string" unit="expanded_site_identifier" frame="input_cell">"o004_3"</param> of element <param name="correction.c001.element" type="string" unit="element" frame="input_site">"S"</param> is the input-frame fractional displacement <param name="correction.c001.delta" type="float64[3]" unit="fractional" frame="input_delta">[0.0,9.999999999732445e-07,0.0]</param>.
