"""Shared translation between SPACE and the integer-index encoding the adapters use."""
from src.search.space import SPACE, to_design


def design_from_indices(x: dict):
    """x: {param name: choice index (int) or range value (float)} -> Design."""
    params = {}
    for p in SPACE:
        params[p.name] = p.values[int(x[p.name])] if p.kind == "choice" else float(x[p.name])
    return to_design(params)


def indices_of(design) -> dict:
    """Design -> {param name: index or value}, the inverse of design_from_indices."""
    from src.evaluator import PI_LEVELS
    raw = dict(num_layers=design.num_layers, width_mult=design.width_mult, quant_bits=design.quant_bits,
               time_slice=design.time_slice_ms, priority=PI_LEVELS.index(design.priority), freq_idx=design.freq_idx)
    out = {}
    for p in SPACE:
        out[p.name] = p.values.index(raw[p.name]) if p.kind == "choice" else raw[p.name]
    return out
