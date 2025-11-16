"""Python re-implementations of the legacy material-property FORTRAN utilities."""

from .debye import (
    DebyeControls,
    DebyeFitResult,
    compute_debye_ceps,
    fit_debye,
    generate_dbe_report,
)
from .double_lorentz import (
    DoubleLorentzControls,
    DoubleLorentzResult,
    QuantityKind,
    compute_debye_lorentz_ceps,
    fit_double_lorentz,
    generate_dlm_report,
)
from .rop import RopData, read_material_file

__all__ = [
    "DebyeControls",
    "DebyeFitResult",
    "RopData",
    "DoubleLorentzControls",
    "DoubleLorentzResult",
    "QuantityKind",
    "compute_debye_ceps",
    "compute_debye_lorentz_ceps",
    "fit_debye",
    "fit_double_lorentz",
    "generate_dbe_report",
    "generate_dlm_report",
    "read_material_file",
]
