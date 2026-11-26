#!/usr/bin/env python3
"""Generate motif-based initial amorphous Li-M-X structures in POSCAR format.

Requirements: Python >= 3.9, numpy, scipy, and pymatgen.
Input: cation_anion_averages.json, mapping each cation to anion coordination
averages. Coordination numbers are obtained using Python's round().

Examples::

    python Generating_motif_based_initial_amorphous_structure.py --seed 42
    python Generating_motif_based_initial_amorphous_structure.py \
        --cation Si --anion O --num-cations 100 --output-dir generated --seed 42

Estimate a cubic cell, place cations with periodic distance checks, add
independently rotated anion motifs for coordination numbers 4 through 9,
and insert Li+ atoms to compensate the nominal negative M-X charge.
Only final structures are written, to limx_str by default. Intermediate
M and M-X structures are kept in memory. Positively charged M-X compositions
are skipped because adding Li+ cannot neutralize them; neutral M-X
compositions require no Li atoms.

The M-X construction follows 5_one_q_make_str.py. Its commented-out Li
stage is enabled here, with a retry correction: failed Li placement batches
reduce the distance threshold by 0.9 and retry the same atom, rather than
silently omitting it. The initial Li separation is 6 angstroms; the reduced
threshold persists for subsequent Li atoms and is reported in the log.

Method details retained from the original implementation:
* Tabulated atomic radii (not ionic radii) set the motif length scale.
* Motif coordinates are preserved exactly. For CN=4, 8, and 9, not all
  cation-anion distances equal that length scale; vectors are not normalized.
* Cation separation starts at five times the estimated minimum distance,
  and decreases by a factor of 0.9 after each failed batch of proposals.
* Proposal polar angles are uniform in [0, pi], as in the original code.
* No anion-overlap rejection, energy minimization, or density optimization
  is performed. These are initial configurations for subsequent relaxation.

A fixed seed reproduces a run with the same inputs, options, and numerical
library versions. Filtering the list of compositions changes random-number
consumption and therefore changes structures later in a batch.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from pymatgen.core import Element, Lattice, Structure
from scipy.spatial.transform import Rotation


LOGGER = logging.getLogger(__name__)

TOXIC_ELEMENTS = {"Cd", "Hg", "As", "Tl", "Pb"}
RADIOACTIVE_ELEMENTS = {"U", "Th", "Ra", "Po", "Np", "Pu", "Am"}
PRECIOUS_METALS = {"Au", "Pt", "Pd", "Ir", "Rh", "Re", "Os", "Ag", "Ru"}
EXCLUDED_ELEMENTS = TOXIC_ELEMENTS | RADIOACTIVE_ELEMENTS | PRECIOUS_METALS

# Fixed oxidation-state assignments used in the original volume estimate.
# These are modeling assumptions, not inferred oxidation states.
NOMINAL_CHARGES = {
    symbol: charge
    for charge, symbols in (
        (4, "Ir Hf Os Ru Ti Zr Sn Si Mn Ge C"),
        (7, "Tc"),
        (6, "W"),
        (5, "Ta Nb V Sb Mo P Pa"),
        (3, "Y Tm Sm Sc Pr Lu La In Ho Gd Ga Er Dy Cr Ce Bi B Al Nd Tb Ac Pm"),
        (2, "Fe Cu Zn Sr Ni Mg Eu Co Be Ca Ba"),
        (1, "H"),
        (-2, "O Se Te S"),
        (-1, "Cl F Br I"),
        (-3, "N"),
    )
    for symbol in symbols.split()
}


def nominal_charge(num_cations, cation, anion, coordination):
    """Return the nominal total charge of M_n X_(n * coordination)."""
    for symbol in (cation, anion):
        if symbol not in NOMINAL_CHARGES:
            raise ValueError(f"No nominal charge is defined for {symbol}.")
    return num_cations * (
        NOMINAL_CHARGES[cation] + coordination * NOMINAL_CHARGES[anion]
    )


def motif_length_scale(cation, anion):
    """Return the sum of tabulated atomic radii, in angstroms."""
    radii = [Element(symbol).atomic_radius for symbol in (cation, anion)]
    if any(radius is None for radius in radii):
        raise ValueError(f"Missing atomic radius for {cation}-{anion}.")
    return float(radii[0] + radii[1])


def estimate_cell_parameters(cation, anion, charge):
    """Return the original cell-length estimate and separation, in angstroms.

    The empirical M-X volume uses a fixed factor of 100, as in the source,
    even when --num-cations differs from 100. The main workflow doubles the
    returned cell length. The 1.52 angstrom Li radius is retained solely as
    an empirical volume/separation parameter.
    """
    length = motif_length_scale(cation, anion)
    li_radius = 1.52
    mx_volume = (4 / 3) * np.pi * length**3 * 100
    li_volume = (4 / 3) * np.pi * li_radius**3 * abs(charge)
    return (mx_volume + li_volume) ** (1 / 3), (length + li_radius) * 3


def create_cation_structure(
    num_cations, species, min_distance, lattice_param, rng,
    max_attempts_per_atom=5000, max_distance_reductions=200,
):
    """Place cations using the original adaptive periodic rejection scheme.

    The reduced separation persists for subsequent cations. A finite limit
    on failed batches prevents an unbounded placement loop.
    """
    lattice = Lattice.cubic(lattice_param)
    # Preserve the original first-site fractional-coordinate expression.
    # Subsequent positions are explicitly Cartesian and wrapped into the cell.
    structure = Structure(lattice, [species], [rng.rand(3) * lattice_param])
    reductions = 0

    while len(structure) < num_cations:
        current_min_distance = min_distance * 5.0
        for _ in range(max_attempts_per_atom):
            center = structure[rng.randint(len(structure))]
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2 * np.pi)
            # RandomState deliberately retains the legacy uniform behavior
            # when current_min_distance exceeds half the cell length.
            radius = rng.uniform(current_min_distance, lattice_param / 2)
            offset = radius * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            candidate = (center.coords + offset) % lattice_param
            too_close = any(
                lattice.get_distance_and_image(
                    site.frac_coords, candidate / lattice_param
                )[0] < current_min_distance
                for site in structure
            )
            if not too_close:
                structure.append(species, candidate, coords_are_cartesian=True)
                LOGGER.debug("Placed cation %d/%d", len(structure), num_cations)
                break
        else:
            if reductions >= max_distance_reductions:
                raise RuntimeError(
                    f"Could not place {species} cation {len(structure) + 1} "
                    f"after {reductions} distance reductions."
                )
            min_distance *= 0.9
            reductions += 1
            LOGGER.debug("Reduced cation separation to %.6f A", min_distance * 5)

    return structure


def motif_offsets(coordination, length):
    """Return Cartesian motif offsets, preserving the original geometry.

    CN=4: original four-site motif; CN=5: trigonal bipyramid;
    CN=6: octahedron; CN=7: pentagonal bipyramid; CN=8: cube;
    CN=9: three aligned triangular layers at z=-length, 0, +length.
    """
    axial = np.array([[0, 0, length], [0, 0, -length]])
    if coordination == 4:
        return np.array([
            [0, 0, length],
            [length * np.sqrt(3) / 2, 0, -length / 3],
            [-length * np.sqrt(3) / 4, length * 3 / 4, -length / 3],
            [-length * np.sqrt(3) / 4, -length * 3 / 4, -length / 3],
        ])
    if coordination == 5:
        equatorial = np.array([
            [length, 0, 0],
            [-0.5 * length, np.sqrt(3) / 2 * length, 0],
            [-0.5 * length, -np.sqrt(3) / 2 * length, 0],
        ])
        return np.vstack((axial, equatorial))
    if coordination == 6:
        equatorial = np.array([
            [length, 0, 0], [-length, 0, 0],
            [0, length, 0], [0, -length, 0],
        ])
        return np.vstack((axial, equatorial))
    if coordination == 7:
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        equatorial = np.array([
            [length * np.cos(angle), length * np.sin(angle), 0]
            for angle in angles
        ])
        return np.vstack((axial, equatorial))
    if coordination == 8:
        return np.array([
            [x, y, z]
            for x in (length, -length)
            for y in (length, -length)
            for z in (length, -length)
        ])
    if coordination == 9:
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        return np.array([
            [length * np.cos(angle), length * np.sin(angle), z]
            for z in (length, 0, -length)
            for angle in angles
        ])
    raise ValueError(f"Unsupported coordination number: {coordination} (use 4-9).")


def add_anion_motifs(cation_structure, cation, anion, coordination, rng):
    """Add one independently rotated anion motif around each cation."""
    offsets = motif_offsets(coordination, motif_length_scale(cation, anion))
    structure = cation_structure.copy()
    for site in cation_structure:
        rotation = Rotation.random(random_state=rng).as_matrix()
        for offset in offsets @ rotation.T:
            structure.append(anion, site.coords + offset, coords_are_cartesian=True)
    return structure


def add_lithium_atoms(
    mx_structure, charge, min_distance, rng,
    max_attempts_per_atom=3000, max_distance_reductions=200,
):
    """Return a copy with exactly -charge Li atoms, or raise on failure.

    Random candidates are accepted only when all periodic distances to
    existing atoms (including previously inserted Li) meet the current
    threshold. The initial threshold may be relaxed after failed batches;
    it is not a guaranteed minimum distance in the final structure.
    """
    if charge > 0:
        raise ValueError("Adding Li+ cannot neutralize a positive M-X charge.")
    structure = mx_structure.copy()
    num_li = -charge
    reductions = 0
    LOGGER.info("Adding %d Li atoms; initial minimum distance=%.6f A", num_li, min_distance)
    for index in range(num_li):
        while True:
            for _ in range(max_attempts_per_atom):
                candidate = rng.rand(3)
                distances = structure.lattice.get_all_distances(
                    candidate.reshape(1, 3), structure.frac_coords
                )
                if np.all(distances >= min_distance):
                    structure.append("Li", candidate, coords_are_cartesian=False)
                    LOGGER.debug("Placed Li %d/%d", index + 1, num_li)
                    break
            else:
                if reductions >= max_distance_reductions:
                    raise RuntimeError(
                        f"Could not place Li {index + 1}/{num_li} after "
                        f"{reductions} distance reductions; no partial structure saved."
                    )
                min_distance *= 0.9
                reductions += 1
                LOGGER.warning("Reduced Li minimum distance to %.6f A", min_distance)
                continue
            break
    LOGGER.info("Li placement complete; final distance threshold=%.6f A", min_distance)
    return structure


def positive_integer(value):
    """Argparse validator for atom counts and placement limits."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("The value must be a positive integer.")
    return number


def positive_float(value):
    """Argparse validator for finite, positive distances."""
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("The value must be finite and positive.")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).resolve().with_name("cation_anion_averages.json"),
                        help="Coordination JSON (default: next to this script).")
    parser.add_argument("--output-dir", type=Path, default=Path("lmx_str"),
                        help="Final POSCAR directory (default: li-m-x_str in the current directory).")
    parser.add_argument("--num-cations", type=positive_integer, default=100)
    parser.add_argument("--cation", help="Generate only this cation (exclusions still apply).")
    parser.add_argument("--anion", help="Generate only this anion.")
    parser.add_argument("--seed", type=int, help="Random seed in [0, 2**32 - 1].")
    parser.add_argument("--max-attempts-per-atom", type=positive_integer, default=5000)
    parser.add_argument("--max-distance-reductions", type=positive_integer, default=200)
    parser.add_argument("--li-min-distance", type=positive_float, default=6.0,
                        help="Initial Li separation in angstroms; relaxed on failure (default: 6).")
    parser.add_argument("--li-max-attempts-per-atom", type=positive_integer, default=3000)
    parser.add_argument("--verbose", action="store_true", help="Log placement progress.")
    args = parser.parse_args(argv)
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32 - 1].")
    return args


def main(argv=None):
    """Generate the selected compositions in the original iteration order."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)

    # Validate selections before starting a potentially expensive batch.
    compositions = []
    for cation in reversed(list(data)):
        if cation in EXCLUDED_ELEMENTS or (args.cation and cation != args.cation):
            continue
        for anion, average in data[cation].items():
            if args.anion and anion != args.anion:
                continue
            coordination = round(average)
            if 4 <= coordination <= 9:
                charge = nominal_charge(args.num_cations, cation, anion, coordination)
                if charge > 0:
                    LOGGER.info("Skipping %s-%s: positive M-X charge (%d).", cation, anion, charge)
                    continue
                compositions.append((cation, anion, coordination))
            else:
                LOGGER.info("Skipping %s-%s: coordination %d is outside 4-9.",
                            cation, anion, coordination)
    if not compositions:
        raise ValueError("No supported compositions match the input and element filters.")
    for cation, anion, coordination in compositions:
        nominal_charge(args.num_cations, cation, anion, coordination)
        motif_length_scale(cation, anion)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    LOGGER.info("Seed: %s; selected compositions: %d", args.seed, len(compositions))
    for cation, anion, coordination in compositions:
        charge = nominal_charge(args.num_cations, cation, anion, coordination)
        lattice_param, min_distance = estimate_cell_parameters(cation, anion, charge)
        LOGGER.info("Generating %s-%s (CN=%d): %d cations, cell=%.6f A",
                    cation, anion, coordination, args.num_cations, lattice_param * 2)
        cations = create_cation_structure(
            args.num_cations, cation, min_distance, lattice_param * 2, rng,
            args.max_attempts_per_atom, args.max_distance_reductions,
        )
        structure = add_anion_motifs(cations, cation, anion, coordination, rng)
        structure = add_lithium_atoms(
            structure, charge, args.li_min_distance, rng,
            args.li_max_attempts_per_atom, args.max_distance_reductions,
        )
        structure.sort()
        output = args.output_dir / f"{cation}_{anion}_{coordination}_Li.vasp"
        structure.to(filename=str(output), fmt="poscar")
        LOGGER.info("Saved %s (%d atoms)", output, len(structure))


if __name__ == "__main__":
    main()
