"""
Unified POD-basis loading for DAFoam ROMs.

Reads the labelled POD bases written by TrainingDataInterface (the /pod/{label}/ layout
with a manifest in /pod.attrs) and assembles a single block-diagonal basis in DAFoam's
state-vector ordering. The same code path serves both cases:

  * "monolithic" — one basis spanning all state variables (one block);
  * "separate"   — several per-variable-group bases (block-diagonal).

Loading one label is the monolithic case; loading several labels is the separate case.
Only the requested /pod/{label}/ subgroups are read (not the whole file).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import h5py


# region PODBasis
@dataclass
class PODBasis:
    """Assembled POD basis ready to hand to a DAFoam ROM model."""
    pod_modes:           np.ndarray                 # (n_local_dofs, n_modes_total), DAFoam ordering
    reference_fom_state: np.ndarray                 # (n_local_dofs,)
    scaling:             np.ndarray                 # (n_local_dofs,)
    weights:             np.ndarray                 # (n_local_dofs,) cell-volume inner product
    labels:              list                       # bases included, in column-block order
    var_groups:          dict                       # label -> [vars]
    singular_values:     dict                       # label -> full singular-value array
    n_modes:             dict                       # label -> retained mode count
    n_modes_total:       int
    var_indices:         dict                       # var -> DAFoam state-vector indices
    provenance:          dict = field(default_factory=dict)  # /pod manifest attrs
    full_modes:          dict = field(default_factory=dict)  # label -> (n_local_dofs, n_full) untruncated modes, DAFoam ordering

    # region check_provenance
    def check_provenance(self, expected: dict, print_fn=print) -> bool:
        """
        Warn if the stored reconstruction conventions differ from what the ROM assumes.
        Mismatches here (centering / scaling / inner_product / phi_mode) silently corrupt
        the reconstruction, so it's worth catching loudly. Returns True if all match.
        """
        ok = True
        for key, want in expected.items():
            have = self.provenance.get(key, None)
            if have is not None and str(have) != str(want):
                ok = False
                print_fn(f"WARNING: POD provenance mismatch for '{key}': "
                         f"file has '{have}', ROM expects '{want}'.")
        return ok


# region _decode
def _decode(x):
    """h5py returns str attrs as bytes or numpy str — normalize to python str / list[str]."""
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, np.ndarray):
        return [_decode(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_decode(v) for v in x]
    return str(x)


# region read_pod_manifest
def read_pod_manifest(h5_path) -> dict:
    """Read the /pod manifest attrs (basis labels + provenance) without loading any modes."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        if "pod" not in f:
            raise KeyError(f"No '/pod' group in {h5_path}. Was it written with the unified "
                           f"POD save format (TrainingDataInterface._compute_pod_modes)?")
        attrs = dict(f["pod"].attrs)
        if "basis_labels" not in attrs:
            # Older or partial file: discover labels by listing subgroups.
            attrs["basis_labels"] = list(f["pod"].keys())
    manifest = {k: _decode(v) for k, v in attrs.items()}
    # numeric attrs come back as strings via _decode of non-str scalars; keep ints usable
    for k in ("n_bases", "n_samples"):
        if k in attrs:
            manifest[k] = int(attrs[k])
    return manifest


# region _select_n_modes
def _select_n_modes(singular_values: np.ndarray, target_variance: float, min_modes: int) -> int:
    """Smallest mode count reaching target cumulative energy, floored at min_modes."""
    if singular_values.size == 0:
        return 0
    energy = np.cumsum(singular_values ** 2) / np.sum(singular_values ** 2)
    n_var  = int(np.argmax(energy >= target_variance)) + 1
    return int(min(max(n_var, min_modes), singular_values.size))


# region load_pod_basis
def load_pod_basis(data_generator,
                   h5_path,
                   labels=None,
                   target_variance: float = 0.999,
                   min_modes=0) -> PODBasis:
    """
    Load and assemble a (possibly block-diagonal) POD basis in DAFoam state-vector ordering.

    data_generator : TrainingDataInterface — provides state_info, load_h5, comm, dafoam_instance.
    labels         : None  -> all bases in the file's manifest (in manifest order);
                     str   -> a single basis;
                     list  -> these bases, combined block-diagonally (column-block order).
    target_variance: cumulative-energy threshold for per-basis mode truncation.
    min_modes      : int (applied to every basis) or dict {label: int}.
    """
    state_info   = data_generator.state_info
    n_local_dofs = data_generator.dafoam_instance.getNLocalAdjointStates()

    manifest = read_pod_manifest(h5_path)
    if labels is None:
        labels = list(manifest["basis_labels"])
    elif isinstance(labels, str):
        labels = [labels]
    else:
        labels = list(labels)

    def _min_modes_for(label):
        return int(min_modes.get(label, 0)) if isinstance(min_modes, dict) else int(min_modes)

    # --- First pass: read each basis subgroup and pick its mode count ---
    groups, n_modes, var_groups, sv_dict = {}, {}, {}, {}
    for label in labels:
        grp = data_generator.load_h5(h5_path, group_to_read=f"pod/{label}")
        if "modes" not in grp:
            raise KeyError(f"/pod/{label} has no 'modes' group in {h5_path}.")
        var_group = _decode(grp.get("_attrs", {}).get("var_group", list(grp["modes"].keys())))
        if isinstance(var_group, str):
            var_group = [var_group]
        sv          = np.asarray(grp["singular_values"])
        groups[label]     = grp
        var_groups[label] = var_group
        sv_dict[label]    = sv
        n_modes[label]    = _select_n_modes(sv, target_variance, _min_modes_for(label))

    # --- Coverage check: every state var must come from exactly one selected basis ---
    covered = [v for vg in var_groups.values() for v in vg]
    missing = [v for v in state_info if v not in covered]
    if missing:
        raise ValueError(f"Selected bases {labels} do not cover state variables {missing}. "
                         f"reference/scaling/weights would be undefined for them.")
    dupes = {v for v in covered if covered.count(v) > 1}
    if dupes:
        raise ValueError(f"State variables {sorted(dupes)} appear in more than one selected basis.")

    # --- Second pass: assemble block-diagonal basis + per-DOF reference/scaling/weights ---
    n_modes_total       = sum(n_modes[l] for l in labels)
    pod_modes           = np.zeros((n_local_dofs, n_modes_total))
    reference_fom_state = np.zeros(n_local_dofs)
    scaling             = np.ones(n_local_dofs)
    weights             = np.ones(n_local_dofs)
    var_indices         = {}
    full_modes          = {}   # label -> (n_local_dofs, n_full) untruncated block, for snapshot weighting

    col0 = 0
    for label in labels:
        grp     = groups[label]
        n_b     = n_modes[label]
        n_full  = sv_dict[label].shape[0]
        has_w   = "weights" in grp
        full_b  = np.zeros((n_local_dofs, n_full))
        for var in var_groups[label]:
            idx = state_info[var]["indices"]
            var_indices[var] = idx
            modes_var = np.asarray(grp["modes"][var])       # (n_local_var, n_full), unsliced
            full_b[idx, :]                  = modes_var
            pod_modes[idx, col0:col0 + n_b] = modes_var[:, :n_b]
            reference_fom_state[idx]        = np.asarray(grp["reference_state"][var])
            s_var = np.asarray(grp["scaling"][var])
            scaling[idx] = s_var if s_var.size == idx.size else float(s_var)
            if has_w:
                weights[idx] = np.asarray(grp["weights"][var])
        full_modes[label] = full_b
        col0 += n_b

    return PODBasis(
        pod_modes           = pod_modes,
        reference_fom_state = reference_fom_state,
        scaling             = scaling,
        weights             = weights,
        labels              = labels,
        var_groups          = var_groups,
        singular_values     = sv_dict,
        n_modes             = n_modes,
        n_modes_total       = n_modes_total,
        var_indices         = var_indices,
        provenance          = manifest,
        full_modes          = full_modes,
    )


# region project_onto_basis
def project_onto_basis(basis: PODBasis, full_state: np.ndarray, comm) -> np.ndarray:
    """
    W-orthogonal projection coefficients of a full DAFoam state onto the basis:
        q = Phi^T W (state - ref) / s        (summed across ranks)
    Useful for the interpolated initial ROM state and for snapshot weighting.
    """
    from mpi4py import MPI
    scaled = basis.weights * (full_state - basis.reference_fom_state) / basis.scaling
    return comm.allreduce(basis.pod_modes.T @ scaled, op=MPI.SUM)


# region block_orthonormality_error
def block_orthonormality_error(basis: PODBasis, comm) -> float:
    """Frobenius norm of (Phi^T W Phi - I) — should be ~0 for a valid W-orthonormal basis."""
    from mpi4py import MPI
    G_local = basis.pod_modes.T @ (basis.weights[:, None] * basis.pod_modes)
    G       = comm.allreduce(G_local, op=MPI.SUM)
    return float(np.linalg.norm(G - np.eye(G.shape[0]), "fro"))
