from __future__ import annotations

"""
2D boundary-integral / MoM RCS solver.

High-level workflow:
1) Parse geometry/material flags into discretized panels.
2) Build boundary-integral operators (single-layer + normal-derivative terms).
3) Assemble and solve either:
   - legacy single-equation EFIE/MFIE-like system, or
   - coupled dielectric trace system (u, q-) with interface/junction constraints.
4) Post-process solved boundary unknowns into monostatic far-field RCS.

Notes:
- Uses e^{-j omega t} convention.
- Supports lossy media via complex wavenumber in coupled mode.
- Pulse basis + point matching (collocation) on each panel.
"""

import cmath
import concurrent.futures
import ctypes
import ctypes.util
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np
try:
    from scipy import special as _SCIPY_SPECIAL
except Exception:
    _SCIPY_SPECIAL = None
try:
    from scipy import linalg as _SCIPY_LINALG
except Exception:
    _SCIPY_LINALG = None

try:
    import mpmath as _MPMATH
except Exception:
    _MPMATH = None

C0 = 299_792_458.0
ETA0 = 376.730313668
EPS = 1e-12
EULER_GAMMA = 0.5772156649015329
CFIE_EPS = 1e-3
MAX_PANELS_DEFAULT = 20_000
# Monostatic 2D RCS normalization controls.
#
# For the asymptotic convention used here,
#   G(r) = (j/4) H_0^(2)(k r),
# and for a far-field amplitude A defined such that
#   u_s(r,phi) ~ sqrt(1 / (8*pi*k*r)) * exp(-j(kr-pi/4)) * A(phi),
# the 2D scattering width per unit length is
#   sigma_2d(phi) = |A(phi)|^2 / (4 k).
#
# Use physical 2D scattering-width normalization by default.
#
# Historical workflows may still request the legacy "no_k" behavior for
# comparison against older internal solver outputs, but exported GRIM data now
# defaults to the physical width sigma_2d = |A|^2 / (4 k).
RCS_NORM_NUMERATOR = 0.25
RCS_NORM_MODE_DEFAULT = "physical"
RCS_NORM_MODE_PHYSICAL = "physical"


@dataclass
class Panel:
    """Single discretized boundary element (panel) used by collocation."""

    name: str
    seg_type: int
    ibc_flag: int
    ipn1: int
    ipn2: int
    p0: np.ndarray
    p1: np.ndarray
    center: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: float


@dataclass
class LinearNode:
    """Unique mesh node for a continuous piecewise-linear boundary discretization."""

    xy: np.ndarray
    key: Tuple[int, int]


@dataclass
class LinearElement:
    """Two-node straight boundary element used by the linear Galerkin groundwork path."""

    name: str
    seg_type: int
    ibc_flag: int
    ipn1: int
    ipn2: int
    node_ids: Tuple[int, int]
    p0: np.ndarray
    p1: np.ndarray
    center: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: float
    panel_index: int


@dataclass
class LinearMesh:
    """Continuous linear boundary mesh assembled from the collocation panels."""

    nodes: List[LinearNode]
    elements: List[LinearElement]


@dataclass
class PanelCoupledInfo:
    """
    Per-panel material/interface bookkeeping for the coupled dielectric formulation.

    Unknown vector in coupled mode is [u_trace, q_minus], and this record tells us
    how to map each panel's plus/minus side constitutive/interface behavior.
    """

    seg_type: int
    plus_region: int
    minus_region: int
    plus_has_incident: bool
    minus_has_incident: bool
    eps_plus: complex
    mu_plus: complex
    eps_minus: complex
    mu_minus: complex
    k_plus: complex
    k_minus: complex
    q_plus_beta: complex
    q_plus_gamma: complex
    bc_kind: str
    robin_impedance: complex


@dataclass
class ComplexTable:
    """Frequency-dependent complex scalar table with linear interpolation."""

    freqs_ghz: np.ndarray
    values: np.ndarray

    def sample(self, freq_ghz: float) -> complex:
        if len(self.freqs_ghz) == 1:
            return complex(self.values[0])
        real = np.interp(freq_ghz, self.freqs_ghz, self.values.real)
        imag = np.interp(freq_ghz, self.freqs_ghz, self.values.imag)
        return complex(real, imag)


@dataclass
class MediumTable:
    """Frequency-dependent (eps, mu) table with linear interpolation."""

    freqs_ghz: np.ndarray
    eps_values: np.ndarray
    mu_values: np.ndarray

    def sample(self, freq_ghz: float) -> Tuple[complex, complex]:
        if len(self.freqs_ghz) == 1:
            return complex(self.eps_values[0]), complex(self.mu_values[0])
        eps_r = np.interp(freq_ghz, self.freqs_ghz, self.eps_values.real)
        eps_i = np.interp(freq_ghz, self.freqs_ghz, self.eps_values.imag)
        mu_r = np.interp(freq_ghz, self.freqs_ghz, self.mu_values.real)
        mu_i = np.interp(freq_ghz, self.freqs_ghz, self.mu_values.imag)
        return complex(eps_r, eps_i), complex(mu_r, mu_i)


@dataclass
class PreparedLinearSolver:
    """Reusable linear-solve handle for repeated Ax=b with fixed A."""

    a_mat: np.ndarray
    method: str
    lu: np.ndarray | None = None
    piv: np.ndarray | None = None
    null_basis: np.ndarray | None = None
    reduced_mat: np.ndarray | None = None
    constraint_mat: np.ndarray | None = None


class MaterialLibrary:
    """Material lookup facade for constant values and fort.* frequency tables."""

    def __init__(
        self,
        impedance_models: Dict[int, complex | ComplexTable],
        dielectric_models: Dict[int, Tuple[complex, complex] | MediumTable],
    ):
        self.impedance_models = impedance_models
        self.dielectric_models = dielectric_models
        self.warnings: List[str] = []
        self._warning_seen: Set[str] = set()

    @classmethod
    def from_entries(
        cls,
        ibcs_entries: List[List[str]],
        dielectric_entries: List[List[str]],
        base_dir: str,
    ) -> "MaterialLibrary":
        impedance_models: Dict[int, complex | ComplexTable] = {}
        dielectric_models: Dict[int, Tuple[complex, complex] | MediumTable] = {}

        for row in ibcs_entries:
            if not row:
                continue
            flag = _parse_flag(row[0])
            if flag <= 0:
                continue
            if flag > 50:
                path = _resolve_fort_file(base_dir, flag)
                impedance_models[flag] = _load_impedance_table(path)
                continue
            z_real = _parse_float(row[1] if len(row) > 1 else 0.0, 0.0)
            z_imag = _parse_float(row[2] if len(row) > 2 else 0.0, 0.0)
            impedance_models[flag] = _ensure_finite_complex(
                complex(z_real, z_imag),
                f"IBC flag {flag} impedance",
            )

        for row in dielectric_entries:
            if not row:
                continue
            flag = _parse_flag(row[0])
            if flag <= 0:
                continue
            if flag > 50:
                path = _resolve_fort_file(base_dir, flag)
                dielectric_models[flag] = _load_dielectric_table(path)
                continue
            eps_real = _parse_float(row[1] if len(row) > 1 else 1.0, 1.0)
            eps_imag = _parse_float(row[2] if len(row) > 2 else 0.0, 0.0)
            mu_real = _parse_float(row[3] if len(row) > 3 else 1.0, 1.0)
            mu_imag = _parse_float(row[4] if len(row) > 4 else 0.0, 0.0)
            eps_raw = _ensure_finite_complex(
                complex(eps_real, -eps_imag),
                f"Dielectric flag {flag} epsilon",
            )
            mu_raw = _ensure_finite_complex(
                complex(mu_real, -mu_imag),
                f"Dielectric flag {flag} mu",
            )
            eps = _normalize_material_value(eps_raw, 1.0 + 0j)
            mu = _normalize_material_value(mu_raw, 1.0 + 0j)
            dielectric_models[flag] = (eps, mu)

        return cls(impedance_models=impedance_models, dielectric_models=dielectric_models)

    def get_impedance(self, flag: int, freq_ghz: float) -> complex:
        if flag <= 0:
            return 0.0 + 0.0j
        model = self.impedance_models.get(flag)
        if model is None:
            return 0.0 + 0.0j
        if isinstance(model, ComplexTable):
            fmin = float(np.min(model.freqs_ghz))
            fmax = float(np.max(model.freqs_ghz))
            if freq_ghz < fmin or freq_ghz > fmax:
                self._warn_once(
                    f"Impedance flag {flag} sampled at {freq_ghz:g} GHz outside table range [{fmin:g}, {fmax:g}] GHz."
                )
            return _ensure_finite_complex(
                model.sample(freq_ghz),
                f"IBC flag {flag} impedance sampled at {freq_ghz:g} GHz",
            )
        return _ensure_finite_complex(model, f"IBC flag {flag} impedance")

    def get_medium(self, flag: int, freq_ghz: float) -> Tuple[complex, complex]:
        if flag <= 0:
            return 1.0 + 0.0j, 1.0 + 0.0j
        model = self.dielectric_models.get(flag)
        if model is None:
            return 1.0 + 0.0j, 1.0 + 0.0j
        if isinstance(model, MediumTable):
            fmin = float(np.min(model.freqs_ghz))
            fmax = float(np.max(model.freqs_ghz))
            if freq_ghz < fmin or freq_ghz > fmax:
                self._warn_once(
                    f"Dielectric flag {flag} sampled at {freq_ghz:g} GHz outside table range [{fmin:g}, {fmax:g}] GHz."
                )
            eps, mu = model.sample(freq_ghz)
            return (
                _normalize_material_value(eps, 1.0 + 0.0j),
                _normalize_material_value(mu, 1.0 + 0.0j),
            )
        eps, mu = model
        return (
            _normalize_material_value(eps, 1.0 + 0.0j),
            _normalize_material_value(mu, 1.0 + 0.0j),
        )

    def _warn_once(self, message: str) -> None:
        if message in self._warning_seen:
            return
        self._warning_seen.add(message)
        self.warnings.append(message)

    def warn_once(self, message: str) -> None:
        self._warn_once(message)


class _BesselBackend:
    """
    Real-argument Bessel backend.

    Backend preference:
    1) libc/libm j0/y0/j1/y1
    2) scipy.special j0/y0/j1/y1
    3) local series/asymptotic approximations
    """

    def __init__(self):
        self._lib = None
        self._j0 = None
        self._y0 = None
        self._j1 = None
        self._y1 = None
        self._backend_name = "series-fallback"

        libname = ctypes.util.find_library("m")
        if libname:
            try:
                lib = ctypes.CDLL(libname)
                self._j0 = lib.j0
                self._j0.argtypes = [ctypes.c_double]
                self._j0.restype = ctypes.c_double
                self._y0 = lib.y0
                self._y0.argtypes = [ctypes.c_double]
                self._y0.restype = ctypes.c_double
                self._j1 = lib.j1
                self._j1.argtypes = [ctypes.c_double]
                self._j1.restype = ctypes.c_double
                self._y1 = lib.y1
                self._y1.argtypes = [ctypes.c_double]
                self._y1.restype = ctypes.c_double
                self._lib = lib
                self._backend_name = "libm"
                return
            except Exception:
                self._lib = None
                self._j0 = None
                self._y0 = None
                self._j1 = None
                self._y1 = None

        if _SCIPY_SPECIAL is not None:
            try:
                # Ensure required real-order functions are present/callable.
                float(_SCIPY_SPECIAL.j0(0.0))
                float(_SCIPY_SPECIAL.y0(1.0))
                float(_SCIPY_SPECIAL.j1(0.0))
                float(_SCIPY_SPECIAL.y1(1.0))
                self._backend_name = "scipy-special"
            except Exception:
                self._backend_name = "series-fallback"

    @property
    def available(self) -> bool:
        return self._backend_name != "series-fallback"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def j0(self, x: float) -> float:
        if self._j0 is not None:
            return float(self._j0(float(x)))
        if self._backend_name == "scipy-special" and _SCIPY_SPECIAL is not None:
            return float(_SCIPY_SPECIAL.j0(float(x)))
        return _j0_fallback(x)

    def y0(self, x: float) -> float:
        if self._y0 is not None:
            return float(self._y0(float(x)))
        if self._backend_name == "scipy-special" and _SCIPY_SPECIAL is not None:
            return float(_SCIPY_SPECIAL.y0(float(x)))
        return _y0_fallback(x)

    def j1(self, x: float) -> float:
        if self._j1 is not None:
            return float(self._j1(float(x)))
        if self._backend_name == "scipy-special" and _SCIPY_SPECIAL is not None:
            return float(_SCIPY_SPECIAL.j1(float(x)))
        return _j1_fallback(x)

    def y1(self, x: float) -> float:
        if self._y1 is not None:
            return float(self._y1(float(x)))
        if self._backend_name == "scipy-special" and _SCIPY_SPECIAL is not None:
            return float(_SCIPY_SPECIAL.y1(float(x)))
        return _y1_fallback(x)


_BESSEL = _BesselBackend()


# --- Special-function helpers -------------------------------------------------
# Real-argument helpers are used heavily for lossless/real-k paths.
# Complex-argument Hankel is needed for lossy media (complex-k kernels).
def _j0_fallback(x: float) -> float:
    ax = abs(float(x))
    if ax < 12.0:
        xsq = 0.25 * ax * ax
        term = 1.0
        acc = 1.0
        for m in range(1, 80):
            term *= -xsq / (m * m)
            acc += term
            if abs(term) < 1e-16:
                break
        return acc

    phase = ax - math.pi / 4.0
    amp = math.sqrt(2.0 / (math.pi * ax))
    return amp * math.cos(phase)


def _y0_fallback(x: float) -> float:
    ax = max(abs(float(x)), 1e-12)
    if ax < 12.0:
        j0 = _j0_fallback(ax)
        xsq = 0.25 * ax * ax
        term = 1.0
        harmonic = 0.0
        acc = 0.0
        for m in range(1, 80):
            harmonic += 1.0 / m
            term *= -xsq / (m * m)
            acc -= harmonic * term
            if abs(term * harmonic) < 1e-16:
                break
        return (2.0 / math.pi) * ((math.log(ax / 2.0) + EULER_GAMMA) * j0 + acc)

    phase = ax - math.pi / 4.0
    amp = math.sqrt(2.0 / (math.pi * ax))
    return amp * math.sin(phase)


def _j1_fallback(x: float) -> float:
    ax = abs(float(x))
    sign = -1.0 if x < 0.0 else 1.0
    if ax < 12.0:
        xhalf = 0.5 * ax
        term = xhalf
        acc = term
        for m in range(1, 80):
            term *= -(xhalf * xhalf) / (m * (m + 1.0))
            acc += term
            if abs(term) < 1e-16:
                break
        return sign * acc

    phase = ax - 3.0 * math.pi / 4.0
    amp = math.sqrt(2.0 / (math.pi * ax))
    return sign * (amp * math.cos(phase))


def _y1_fallback(x: float) -> float:
    ax = max(abs(float(x)), 1e-12)
    sign = -1.0 if x < 0.0 else 1.0
    if ax < 12.0:
        return sign * (
            -2.0 / (math.pi * ax)
            + (ax / math.pi) * (math.log(ax / 2.0) + EULER_GAMMA - 0.5)
        )

    phase = ax - 3.0 * math.pi / 4.0
    amp = math.sqrt(2.0 / (math.pi * ax))
    return sign * (amp * math.sin(phase))


def _complex_hankel_backend_name() -> str:
    """Report which complex Hankel implementation is active."""

    if _SCIPY_SPECIAL is not None:
        return "scipy-special"
    if _MPMATH is not None:
        return "mpmath"
    return "native-series-asymptotic"


def _raise_if_untrusted_math_backends() -> None:
    """Abort production solves when only approximation fallback math backends are available."""

    if _BESSEL.backend_name == "series-fallback":
        raise RuntimeError(
            "Aborting solve: real-argument Bessel evaluation is using the native series/asymptotic "
            "fallback backend. Install SciPy or provide libm j0/y0/j1/y1 before running production solves."
        )


def _j0_complex_series(z: complex) -> complex:
    zz = 0.25 * z * z
    term = 1.0 + 0.0j
    acc = term
    for m in range(1, 160):
        term *= -zz / (m * m)
        acc += term
        if abs(term) <= 1e-16 * max(1.0, abs(acc)):
            break
    return acc


def _j1_complex_series(z: complex) -> complex:
    z_half = 0.5 * z
    term = z_half
    acc = term
    for m in range(1, 160):
        term *= -(z_half * z_half) / (m * (m + 1.0))
        acc += term
        if abs(term) <= 1e-16 * max(1.0, abs(acc)):
            break
    return acc


def _y0_complex_series(z: complex) -> complex:
    z_safe = z if abs(z) > 1e-14 else (1e-14 + 0.0j)
    j0 = _j0_complex_series(z_safe)
    zz = 0.25 * z_safe * z_safe
    term = 1.0 + 0.0j
    harmonic = 0.0
    acc = 0.0 + 0.0j
    for m in range(1, 160):
        harmonic += 1.0 / m
        term *= -zz / (m * m)
        acc -= harmonic * term
        if abs(harmonic * term) <= 1e-16 * max(1.0, abs(acc), abs(j0)):
            break
    return (2.0 / math.pi) * ((cmath.log(z_safe / 2.0) + EULER_GAMMA) * j0 + acc)


def _y1_complex_series(z: complex) -> complex:
    z_safe = z if abs(z) > 1e-14 else (1e-14 + 0.0j)
    j1 = _j1_complex_series(z_safe)
    z_half = 0.5 * z_safe
    term = z_half
    harmonic_k = 0.0
    harmonic_k1 = 1.0
    acc = (harmonic_k + harmonic_k1) * term
    for k in range(1, 160):
        term *= -(z_half * z_half) / (k * (k + 1.0))
        harmonic_k += 1.0 / k
        harmonic_k1 = harmonic_k + 1.0 / (k + 1.0)
        contrib = (harmonic_k + harmonic_k1) * term
        acc += contrib
        if abs(contrib) <= 1e-16 * max(1.0, abs(acc), abs(j1)):
            break
    return (
        (2.0 / math.pi) * (cmath.log(z_safe / 2.0) + EULER_GAMMA) * j1
        - (1.0 / math.pi) * acc
        - (2.0 / (math.pi * z_safe))
    )


def _hankel2_asymptotic(order: int, z: complex) -> complex:
    z_safe = z if abs(z) > 1e-14 else (1e-14 + 0.0j)
    phase = z_safe - ((0.5 * order) + 0.25) * math.pi
    amp = cmath.sqrt(2.0 / (math.pi * z_safe))
    return amp * cmath.exp(-1j * phase)


def _hankel2_complex_fallback(order: int, z: complex) -> complex:
    if abs(z) < 16.0:
        if order == 0:
            return _j0_complex_series(z) - 1j * _y0_complex_series(z)
        return _j1_complex_series(z) - 1j * _y1_complex_series(z)
    return _hankel2_asymptotic(order, z)


def _hankel2_0(x: complex | float) -> complex:
    """Hankel H_0^(2), with real fast path and no approximation fallback in production."""

    z = complex(x)
    if abs(z.imag) <= 1e-14 and z.real >= 0.0:
        xx = max(float(z.real), 1e-12)
        return complex(_BESSEL.j0(xx), -_BESSEL.y0(xx))
    if _SCIPY_SPECIAL is not None:
        try:
            return complex(_SCIPY_SPECIAL.hankel2(0, z))
        except Exception:
            pass
    if _MPMATH is not None:
        try:
            return complex(_MPMATH.hankel2(0, z))
        except Exception:
            pass
    raise RuntimeError(
        "Aborting solve: complex Hankel H_0^(2) evaluation requires SciPy or mpmath. "
        "Native complex series/asymptotic fallback is disabled for production runs."
    )


def _hankel2_1(x: complex | float) -> complex:
    """Hankel H_1^(2), with real fast path and no approximation fallback in production."""

    z = complex(x)
    if abs(z.imag) <= 1e-14 and z.real >= 0.0:
        xx = max(float(z.real), 1e-12)
        return complex(_BESSEL.j1(xx), -_BESSEL.y1(xx))
    if _SCIPY_SPECIAL is not None:
        try:
            return complex(_SCIPY_SPECIAL.hankel2(1, z))
        except Exception:
            pass
    if _MPMATH is not None:
        try:
            return complex(_MPMATH.hankel2(1, z))
        except Exception:
            pass
    raise RuntimeError(
        "Aborting solve: complex Hankel H_1^(2) evaluation requires SciPy or mpmath. "
        "Native complex series/asymptotic fallback is disabled for production runs."
    )


def _parse_flag(token: Any) -> int:
    text = str(token).strip().lower()
    if not text:
        return 0
    if text.startswith("fort."):
        text = text.split("fort.", 1)[1]
    try:
        return int(float(text))
    except ValueError:
        return 0


def _parse_float(token: Any, default: float = 0.0) -> float:
    try:
        return float(token)
    except (TypeError, ValueError):
        return default


def _parse_int(token: Any, default: int = 0) -> int:
    try:
        return int(round(float(token)))
    except (TypeError, ValueError):
        return default


def _ensure_finite_complex(value: complex, context: str) -> complex:
    z = complex(value)
    if not np.isfinite(z.real) or not np.isfinite(z.imag):
        raise ValueError(f"{context} contains non-finite value {z!r}.")
    return z


def _normalize_material_value(value: complex, fallback: complex) -> complex:
    if not np.isfinite(value.real) or not np.isfinite(value.imag) or abs(value) < EPS:
        return fallback
    return value


def _resolve_fort_file(base_dir: str, flag: int) -> str:
    """Resolve a fort.<flag> material file relative to geometry dir/current cwd."""

    name = f"fort.{flag}"
    candidates = [os.path.join(base_dir, name), os.path.join(os.getcwd(), name)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not locate material file {name} in {base_dir} or current directory.")


def _read_numeric_rows(path: str, min_columns: int) -> List[List[float]]:
    """Read numeric rows, drop comments/bad rows, sort by frequency, de-duplicate."""

    rows: List[List[float]] = []
    with open(path, "r") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < min_columns:
                continue
            try:
                parsed = [float(tokens[i]) for i in range(min_columns)]
            except ValueError:
                continue
            if not all(math.isfinite(v) for v in parsed):
                raise ValueError(
                    f"Material file '{path}' line {lineno} contains non-finite numeric value(s): {tokens[:min_columns]}."
                )
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No valid numeric rows found in {path}")
    rows.sort(key=lambda row: row[0])
    dedup: Dict[float, List[float]] = {}
    for row in rows:
        dedup[row[0]] = row
    return [dedup[freq] for freq in sorted(dedup.keys())]


def _load_impedance_table(path: str) -> ComplexTable:
    """Load frequency -> complex impedance table: f(GHz) z_real z_imag."""

    rows = _read_numeric_rows(path, 3)
    freqs = np.asarray([r[0] for r in rows], dtype=float)
    vals = np.asarray([complex(r[1], r[2]) for r in rows], dtype=np.complex128)
    return ComplexTable(freqs_ghz=freqs, values=vals)


def _load_dielectric_table(path: str) -> MediumTable:
    """Load frequency -> (eps, mu) table: f eps_r eps_i mu_r mu_i."""

    rows = _read_numeric_rows(path, 5)
    freqs = np.asarray([r[0] for r in rows], dtype=float)
    eps_vals = np.asarray([complex(r[1], -r[2]) for r in rows], dtype=np.complex128)
    mu_vals = np.asarray([complex(r[3], -r[4]) for r in rows], dtype=np.complex128)
    return MediumTable(freqs_ghz=freqs, eps_values=eps_vals, mu_values=mu_vals)




def _canonical_user_polarization_label(label: str | None) -> str:
    text = str(label or '').strip().upper()
    if text in {'TE', 'VV', 'V', 'VERTICAL'}:
        return 'TE'
    if text in {'TM', 'HH', 'H', 'HORIZONTAL'}:
        return 'TM'
    raise ValueError(f"Unsupported polarization '{label}'. Use TE/TM or VV/HH.")


def _normalize_polarization(polarization: str) -> str:
    """
    Normalize user-facing polarization labels without swapping TE and TM.

    Production convention in this file is now direct:
    - user/internal "TE" are the same branch
    - user/internal "TM" are the same branch

    Accepted aliases are retained for convenience:
    - TE, VV, V, VERTICAL -> TE
    - TM, HH, H, HORIZONTAL -> TM
    """

    pol = (polarization or "").strip().upper()
    if pol in {"TE", "VV", "V", "VERTICAL"}:
        return "TE"
    if pol in {"TM", "HH", "H", "HORIZONTAL"}:
        return "TM"
    raise ValueError(f"Unsupported polarization '{polarization}'. Use TE/TM or VV/HH.")


def _unit_scale_to_meters(units: str) -> float:
    value = (units or "").strip().lower()
    if value in {"inch", "inches", "in"}:
        return 0.0254
    if value in {"meter", "meters", "m"}:
        return 1.0
    raise ValueError(f"Unsupported geometry units '{units}'. Use inches or meters.")


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _arc_center_from_endpoints(p0: np.ndarray, p1: np.ndarray, ang_rad: float) -> Tuple[np.ndarray, float, float]:
    """Recover arc center/radius/start-angle from endpoints + subtended angle."""

    chord_vec = p1 - p0
    chord = float(np.linalg.norm(chord_vec))
    if chord <= EPS:
        raise ValueError("Arc endpoints are coincident.")

    abs_phi = abs(ang_rad)
    if abs_phi <= 1e-9:
        raise ValueError("Arc angle too small.")

    radius = chord / (2.0 * math.sin(abs_phi * 0.5))
    h = chord / (2.0 * math.tan(abs_phi * 0.5))
    mid = 0.5 * (p0 + p1)
    perp = np.asarray([-chord_vec[1], chord_vec[0]], dtype=float) / chord

    centers = [mid + perp * h, mid - perp * h]
    best_center = centers[0]
    best_err = float("inf")
    best_a0 = 0.0

    for center in centers:
        a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
        p1_pred = center + radius * np.asarray([math.cos(a0 + ang_rad), math.sin(a0 + ang_rad)], dtype=float)
        err = float(np.linalg.norm(p1_pred - p1))
        if err < best_err:
            best_err = err
            best_center = center
            best_a0 = a0

    return best_center, radius, best_a0


def _discretize_primitive(p0: np.ndarray, p1: np.ndarray, ang_deg: float, count: int) -> List[np.ndarray]:
    """Generate panel endpoints for a line or circular-arc primitive."""

    count = max(1, int(count))
    if abs(ang_deg) < 1e-9:
        return [p0 + (p1 - p0) * (i / count) for i in range(count + 1)]

    ang_rad = math.radians(ang_deg)
    center, radius, a0 = _arc_center_from_endpoints(p0, p1, ang_rad)
    points: List[np.ndarray] = []
    for i in range(count + 1):
        t = i / count
        a = a0 + ang_rad * t
        points.append(center + radius * np.asarray([math.cos(a), math.sin(a)], dtype=float))
    return points


def _primitive_length(p0: np.ndarray, p1: np.ndarray, ang_deg: float) -> float:
    chord = float(np.linalg.norm(p1 - p0))
    if chord <= EPS:
        return 0.0
    if abs(ang_deg) < 1e-9:
        return chord
    phi = abs(math.radians(ang_deg))
    radius = chord / (2.0 * math.sin(phi * 0.5))
    return radius * phi


def _panel_count_from_n(n_prop: int, primitive_len: float, min_wavelength: float) -> int:
    """
    Convert geometry n property to panel count.

    n > 0: explicit panel count.
    n < 0: panels-per-wavelength style control.
    """

    if primitive_len <= EPS:
        return 1
    if n_prop > 0:
        return max(1, n_prop)
    if n_prop < 0:
        n_wave = max(1, abs(n_prop))
        target = max(min_wavelength / n_wave, primitive_len / 2000.0)
        return max(1, int(math.ceil(primitive_len / target)))
    return 1


def _segment_closed_area2(point_pairs: List[Dict[str, Any]], meters_scale: float) -> tuple[bool, float]:
    """Return (is_closed, signed_area2) for a multi-primitive segment chain."""

    if not point_pairs:
        return False, 0.0

    pts: List[tuple[float, float]] = []
    for idx, pair in enumerate(point_pairs):
        x1 = _parse_float(pair.get("x1", 0.0), 0.0) * meters_scale
        y1 = _parse_float(pair.get("y1", 0.0), 0.0) * meters_scale
        x2 = _parse_float(pair.get("x2", 0.0), 0.0) * meters_scale
        y2 = _parse_float(pair.get("y2", 0.0), 0.0) * meters_scale
        if idx == 0:
            pts.append((x1, y1))
        pts.append((x2, y2))

    if len(pts) < 3:
        return False, 0.0

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    diag = max(float(math.hypot(max(xs) - min(xs), max(ys) - min(ys))), 1.0)
    tol = max(1e-12, 1e-9 * diag)
    closed = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= tol
    if not closed:
        return False, 0.0

    area2 = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        area2 += x0 * y1 - x1 * y0
    return True, float(area2)


def _normalize_segment_orientation(
    seg_type: int,
    ang_deg: float,
    point_pairs: List[Dict[str, Any]],
    meters_scale: float,
) -> tuple[List[Dict[str, Any]], float]:
    """
    Canonicalize closed contour orientation for boundary types that depend on outward normals.

    The project convention uses the left-hand normal of each drawn primitive. For a closed
    contour, clockwise traversal gives outward normals. Counter-clockwise closed contours are
    auto-flipped here so the solve is less sensitive to drawing order for PEC/IBC/dielectric
    boundaries.
    """

    if seg_type not in {2, 3, 4, 5} or len(point_pairs) < 2:
        return point_pairs, ang_deg

    closed, area2 = _segment_closed_area2(point_pairs, meters_scale)
    if not closed or area2 <= 0.0:
        return point_pairs, ang_deg

    flipped: List[Dict[str, Any]] = []
    for pair in reversed(point_pairs):
        flipped.append(
            {
                "x1": pair.get("x2", 0.0),
                "y1": pair.get("y2", 0.0),
                "x2": pair.get("x1", 0.0),
                "y2": pair.get("y1", 0.0),
            }
        )
    return flipped, -float(ang_deg)




def _snapshot_segments(geometry_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(geometry_snapshot.get('segments', []) or [])


def _solver_point_key(x: float, y: float, tol: float) -> Tuple[int, int]:
    inv = 1.0 / max(tol, 1e-12)
    return int(round(float(x) * inv)), int(round(float(y) * inv))


def _points_close(a: Tuple[float, float], b: Tuple[float, float], tol: float) -> bool:
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) <= (tol * tol)


def _segment_intersects_strict(
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    b1: Tuple[float, float],
    b2: Tuple[float, float],
    tol: float,
) -> bool:
    if _points_close(a1, b1, tol) or _points_close(a1, b2, tol) or _points_close(a2, b1, tol) or _points_close(a2, b2, tol):
        return False

    def orient(p, q, r):
        return (float(q[0]) - float(p[0])) * (float(r[1]) - float(p[1])) - (float(q[1]) - float(p[1])) * (float(r[0]) - float(p[0]))

    def on_seg(p, q, r):
        return (
            min(float(p[0]), float(r[0])) - tol <= float(q[0]) <= max(float(p[0]), float(r[0])) + tol
            and min(float(p[1]), float(r[1])) - tol <= float(q[1]) <= max(float(p[1]), float(r[1])) + tol
        )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)):
        return True
    if abs(o1) <= tol and on_seg(a1, b1, a2):
        return True
    if abs(o2) <= tol and on_seg(a1, b2, a2):
        return True
    if abs(o3) <= tol and on_seg(b1, a1, b2):
        return True
    if abs(o4) <= tol and on_seg(b1, a2, b2):
        return True
    return False


def validate_geometry_snapshot_for_solver(
    geometry_snapshot: Dict[str, Any],
    base_dir: str,
) -> Dict[str, Any]:
    """
    Strict solver-side preflight for geometry/material consistency.

    This complements the GUI validator and protects headless solves / exports.
    Fatal problems raise before assembly begins.
    """

    segments = _snapshot_segments(geometry_snapshot)
    if not segments:
        raise ValueError('Geometry snapshot contains no segments.')

    ibc_rows = [list(row) for row in (geometry_snapshot.get('ibcs', []) or []) if list(row)]
    diel_rows = [list(row) for row in (geometry_snapshot.get('dielectrics', []) or []) if list(row)]
    ibc_flags = {_parse_flag(row[0]) for row in ibc_rows if row}
    diel_flags = {_parse_flag(row[0]) for row in diel_rows if row}

    warnings: List[str] = []
    primitives: List[Tuple[int, int, str, Tuple[float, float], Tuple[float, float]]] = []
    all_points: List[Tuple[float, float]] = []

    for seg_idx, seg in enumerate(segments):
        props = list(seg.get('properties', []) or [])
        if len(props) < 6:
            props.extend([''] * (6 - len(props)))
        seg_name = str(seg.get('name', f'segment_{seg_idx + 1}'))
        seg_type = _parse_flag(props[0] if props and str(props[0]).strip() else seg.get('seg_type', 0))
        ibc_flag = _parse_flag(props[3])
        ipn1 = _parse_flag(props[4])
        ipn2 = _parse_flag(props[5])
        point_pairs = list(seg.get('point_pairs', []) or [])

        if seg_type < 1 or seg_type > 5:
            raise ValueError(f"Segment '{seg_name}' has invalid TYPE '{props[0]}'; expected 1..5.")
        if not point_pairs:
            raise ValueError(f"Segment '{seg_name}' has no primitives/point_pairs.")

        prev_end = None
        for prim_idx, pair in enumerate(point_pairs):
            x1 = _parse_float(pair.get('x1', 0.0), 0.0)
            y1 = _parse_float(pair.get('y1', 0.0), 0.0)
            x2 = _parse_float(pair.get('x2', 0.0), 0.0)
            y2 = _parse_float(pair.get('y2', 0.0), 0.0)
            vals = [x1, y1, x2, y2]
            if not all(math.isfinite(v) for v in vals):
                raise ValueError(f"Segment '{seg_name}' primitive {prim_idx + 1} contains non-finite coordinates.")
            if ((x2 - x1) ** 2 + (y2 - y1) ** 2) <= EPS * EPS:
                raise ValueError(f"Segment '{seg_name}' primitive {prim_idx + 1} has near-zero length.")
            p1 = (x1, y1)
            p2 = (x2, y2)
            primitives.append((seg_idx, prim_idx, seg_name, p1, p2))
            all_points.extend([p1, p2])
            if prev_end is not None and not _points_close(prev_end, p1, 1e-9):
                warnings.append(
                    f"Segment '{seg_name}' has a disconnected primitive chain between elements {prim_idx} and {prim_idx + 1}."
                )
            prev_end = p2

        if ibc_flag > 0:
            if ibc_flag > 50:
                _resolve_fort_file(base_dir, ibc_flag)
            elif ibc_flag not in ibc_flags:
                raise ValueError(f"Segment '{seg_name}' references undefined IBC flag {ibc_flag}.")

        if seg_type == 3:
            if ipn1 <= 0:
                raise ValueError(f"TYPE 3 segment '{seg_name}' requires IPN1 > 0.")
            if ipn1 > 50:
                _resolve_fort_file(base_dir, ipn1)
            elif ipn1 not in diel_flags:
                raise ValueError(f"TYPE 3 segment '{seg_name}' references undefined dielectric flag {ipn1}.")
        elif seg_type == 4:
            if ipn1 <= 0:
                raise ValueError(f"TYPE 4 segment '{seg_name}' requires IPN1 > 0.")
            if ipn1 > 50:
                _resolve_fort_file(base_dir, ipn1)
            elif ipn1 not in diel_flags:
                raise ValueError(f"TYPE 4 segment '{seg_name}' references undefined dielectric flag {ipn1}.")
        elif seg_type == 5:
            if ipn1 <= 0 or ipn2 <= 0:
                raise ValueError(f"TYPE 5 segment '{seg_name}' requires IPN1 > 0 and IPN2 > 0.")
            for flag in (ipn1, ipn2):
                if flag > 50:
                    _resolve_fort_file(base_dir, flag)
                elif flag not in diel_flags:
                    raise ValueError(f"TYPE 5 segment '{seg_name}' references undefined dielectric flag {flag}.")

    xs = [p[0] for p in all_points] if all_points else [0.0]
    ys = [p[1] for p in all_points] if all_points else [0.0]
    diag = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)
    tol = max(1e-8, 1e-6 * diag)

    node_degree: Dict[Tuple[int, int], int] = {}
    for _, _, _, p1, p2 in primitives:
        key1 = _solver_point_key(p1[0], p1[1], tol)
        key2 = _solver_point_key(p2[0], p2[1], tol)
        node_degree[key1] = node_degree.get(key1, 0) + 1
        node_degree[key2] = node_degree.get(key2, 0) + 1

    dangling_nodes = sum(1 for v in node_degree.values() if v == 1)
    high_degree_nodes = sum(1 for v in node_degree.values() if v > 2)
    if dangling_nodes > 0:
        warnings.append(f'Geometry contains {dangling_nodes} dangling endpoint node(s).')
    if high_degree_nodes > 0:
        warnings.append(f'Geometry contains {high_degree_nodes} high-degree node(s) (>2 connected primitives).')

    for i in range(len(primitives)):
        seg_i, prim_i, name_i, a1, a2 = primitives[i]
        for j in range(i + 1, len(primitives)):
            seg_j, prim_j, name_j, b1, b2 = primitives[j]
            if seg_i == seg_j and abs(prim_i - prim_j) <= 1:
                continue
            if _segment_intersects_strict(a1, a2, b1, b2, tol):
                raise ValueError(
                    f"Geometry contains an unsupported segment intersection between '{name_i}' primitive {prim_i + 1} and '{name_j}' primitive {prim_j + 1}."
                )

    return {
        'segment_count': int(len(segments)),
        'primitive_count': int(len(primitives)),
        'dangling_nodes': int(dangling_nodes),
        'high_degree_nodes': int(high_degree_nodes),
        'warning_count': int(len(warnings)),
        'warnings': warnings,
    }


def _build_panels(
    geometry_snapshot: Dict[str, Any],
    meters_scale: float,
    min_wavelength: float,
    max_panels: int = MAX_PANELS_DEFAULT,
) -> List[Panel]:
    """
    Discretize all geometry primitives into collocation panels with oriented normals.

    Normal direction follows endpoint ordering of each primitive.
    """

    panels: List[Panel] = []
    segments = geometry_snapshot.get("segments", []) or []

    for seg in segments:
        props = list(seg.get("properties", []) or [])
        seg_type = _parse_flag(props[0] if len(props) > 0 else 2)
        n_prop = _parse_int(props[1] if len(props) > 1 else 1, 1)
        ang_deg = _parse_float(props[2] if len(props) > 2 else 0.0, 0.0)
        ibc_flag = _parse_flag(props[3] if len(props) > 3 else 0)
        ipn1 = _parse_flag(props[4] if len(props) > 4 else 0)
        ipn2 = _parse_flag(props[5] if len(props) > 5 else 0)
        name = str(seg.get("name", "segment"))

        point_pairs = list(seg.get("point_pairs", []) or [])
        point_pairs, ang_deg = _normalize_segment_orientation(seg_type, ang_deg, point_pairs, meters_scale)
        for pair in point_pairs:
            p0 = np.asarray([
                _parse_float(pair.get("x1", 0.0), 0.0) * meters_scale,
                _parse_float(pair.get("y1", 0.0), 0.0) * meters_scale,
            ], dtype=float)
            p1 = np.asarray([
                _parse_float(pair.get("x2", 0.0), 0.0) * meters_scale,
                _parse_float(pair.get("y2", 0.0), 0.0) * meters_scale,
            ], dtype=float)

            prim_len = _primitive_length(p0, p1, ang_deg)
            count = _panel_count_from_n(n_prop, prim_len, min_wavelength)
            pts = _discretize_primitive(p0, p1, ang_deg, count)

            for i in range(count):
                q0 = pts[i]
                q1 = pts[i + 1]
                vec = q1 - q0
                length = float(np.linalg.norm(vec))
                if length <= EPS:
                    continue
                tangent = vec / length
                # Project convention: a segment drawn left->right has an upward normal.
                # This makes IPN1 the medium on the GUI-indicated normal side.
                normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
                center = 0.5 * (q0 + q1)
                panels.append(
                    Panel(
                        name=name,
                        seg_type=seg_type,
                        ibc_flag=ibc_flag,
                        ipn1=ipn1,
                        ipn2=ipn2,
                        p0=q0,
                        p1=q1,
                        center=center,
                        tangent=tangent,
                        normal=normal,
                        length=length,
                    )
                )

    if not panels:
        raise ValueError("Geometry does not contain any valid discretized panels.")
    max_allowed = max(1, int(max_panels))
    if len(panels) > max_allowed:
        raise ValueError(
            f"Discretization produced {len(panels)} panels; limit is {max_allowed}. "
            "Reduce n/frequency range or increase max_panels."
        )
    return panels


def _linear_node_snap_key(xy: np.ndarray, tol: float = 1.0e-9) -> Tuple[int, int]:
    scale = 1.0 / max(float(tol), EPS)
    return (int(round(float(xy[0]) * scale)), int(round(float(xy[1]) * scale)))


def _linear_shape_values(xi: float) -> np.ndarray:
    x = float(xi)
    return np.asarray([1.0 - x, x], dtype=float)


def _build_linear_mesh(
    panels: List[Panel],
    node_snap_tol: float = 1.0e-9,
) -> LinearMesh:
    """
    Convert collocation panels into a continuous two-node linear boundary mesh.

    This is the stage-1 data-structure upgrade for the future linear Galerkin path.
    Each panel becomes one linear element, while shared endpoints are merged into
    unique global nodes by snapped coordinates.
    """

    node_index: Dict[Tuple[int, int], int] = {}
    nodes: List[LinearNode] = []
    elements: List[LinearElement] = []

    def get_node_id(xy: np.ndarray) -> int:
        key = _linear_node_snap_key(xy, tol=node_snap_tol)
        idx = node_index.get(key)
        if idx is not None:
            return idx
        idx = len(nodes)
        node_index[key] = idx
        nodes.append(LinearNode(xy=np.asarray(xy, dtype=float).copy(), key=key))
        return idx

    for pidx, panel in enumerate(panels):
        n0 = get_node_id(panel.p0)
        n1 = get_node_id(panel.p1)
        elements.append(
            LinearElement(
                name=panel.name,
                seg_type=panel.seg_type,
                ibc_flag=panel.ibc_flag,
                ipn1=panel.ipn1,
                ipn2=panel.ipn2,
                node_ids=(n0, n1),
                p0=np.asarray(panel.p0, dtype=float).copy(),
                p1=np.asarray(panel.p1, dtype=float).copy(),
                center=np.asarray(panel.center, dtype=float).copy(),
                tangent=np.asarray(panel.tangent, dtype=float).copy(),
                normal=np.asarray(panel.normal, dtype=float).copy(),
                length=float(panel.length),
                panel_index=int(pidx),
            )
        )

    if not elements:
        raise ValueError("Linear mesh construction requires at least one element.")
    return LinearMesh(nodes=nodes, elements=elements)



def _linear_panel_signature_from_info(
    panel: Panel,
    info: PanelCoupledInfo,
) -> Tuple[Any, ...]:
    """Topology signature used to decide when linear nodes may be shared safely."""

    return (
        int(panel.seg_type),
        int(panel.ibc_flag),
        int(panel.ipn1),
        int(panel.ipn2),
        int(info.minus_region),
        int(info.plus_region),
        str(info.bc_kind),
    )



def _build_linear_mesh_interface_aware(
    panels: List[Panel],
    infos: List[PanelCoupledInfo],
    node_snap_tol: float = 1.0e-9,
) -> Tuple[LinearMesh, Dict[str, int]]:
    """
    Build a linear boundary mesh that only shares nodes across the *same* interface signature.

    This hardens the linear/Galerkin path for ordinary corners where distinct interface types
    touch at the same geometric coordinate. Those cases should not be forced to share a single
    nodal DOF, because that incorrectly imposes trace continuity across different interfaces.

    True branching nodes where more than two elements of the same interface signature meet are
    still reported separately by `_linear_coupled_node_report` and may fall back to the pulse
    solver in production runs.
    """

    if len(panels) != len(infos):
        raise ValueError("Interface-aware linear mesh requires matching panels and panel infos.")

    node_index: Dict[Tuple[Tuple[int, int], Tuple[Any, ...]], int] = {}
    nodes: List[LinearNode] = []
    elements: List[LinearElement] = []
    geometric_keys: Set[Tuple[int, int]] = set()

    def get_node_id(xy: np.ndarray, signature: Tuple[Any, ...]) -> int:
        geom_key = _linear_node_snap_key(xy, tol=node_snap_tol)
        geometric_keys.add(geom_key)
        full_key = (geom_key, signature)
        idx = node_index.get(full_key)
        if idx is not None:
            return idx
        idx = len(nodes)
        node_index[full_key] = idx
        nodes.append(LinearNode(xy=np.asarray(xy, dtype=float).copy(), key=geom_key))
        return idx

    for pidx, (panel, info) in enumerate(zip(panels, infos)):
        sig = _linear_panel_signature_from_info(panel, info)
        n0 = get_node_id(panel.p0, sig)
        n1 = get_node_id(panel.p1, sig)
        elements.append(
            LinearElement(
                name=panel.name,
                seg_type=panel.seg_type,
                ibc_flag=panel.ibc_flag,
                ipn1=panel.ipn1,
                ipn2=panel.ipn2,
                node_ids=(n0, n1),
                p0=np.asarray(panel.p0, dtype=float).copy(),
                p1=np.asarray(panel.p1, dtype=float).copy(),
                center=np.asarray(panel.center, dtype=float).copy(),
                tangent=np.asarray(panel.tangent, dtype=float).copy(),
                normal=np.asarray(panel.normal, dtype=float).copy(),
                length=float(panel.length),
                panel_index=int(pidx),
            )
        )

    if not elements:
        raise ValueError("Interface-aware linear mesh construction requires at least one element.")

    mesh = LinearMesh(nodes=nodes, elements=elements)
    stats = {
        "linear_geometric_node_count": int(len(geometric_keys)),
        "linear_interface_split_nodes": int(max(0, len(nodes) - len(geometric_keys))),
    }
    return mesh, stats


def _linear_param_to_point(elem: LinearElement, xi: float) -> np.ndarray:
    return elem.p0 + float(xi) * (elem.p1 - elem.p0)


def _linear_interval_point(elem: LinearElement, interval: Tuple[float, float], use_start: bool) -> np.ndarray:
    a, b = float(interval[0]), float(interval[1])
    return _linear_param_to_point(elem, a if use_start else b)


def _linear_interval_length(elem: LinearElement, interval: Tuple[float, float]) -> float:
    a, b = float(interval[0]), float(interval[1])
    return max(abs(b - a) * float(elem.length), 0.0)


def _linear_interval_midpoint(elem: LinearElement, interval: Tuple[float, float]) -> np.ndarray:
    a, b = float(interval[0]), float(interval[1])
    return _linear_param_to_point(elem, 0.5 * (a + b))


def _linear_map_local_to_parent(interval: Tuple[float, float], local_xi: float, start_is_shared: bool) -> float:
    a, b = float(interval[0]), float(interval[1])
    h = b - a
    x = float(local_xi)
    return (a + h * x) if start_is_shared else (b - h * x)


def _linear_shared_interval_endpoint_info(
    obs_elem: LinearElement,
    obs_interval: Tuple[float, float],
    src_elem: LinearElement,
    src_interval: Tuple[float, float],
    tol: float = 1.0e-12,
) -> Tuple[bool, bool] | None:
    obs_pts = [
        _linear_interval_point(obs_elem, obs_interval, True),
        _linear_interval_point(obs_elem, obs_interval, False),
    ]
    src_pts = [
        _linear_interval_point(src_elem, src_interval, True),
        _linear_interval_point(src_elem, src_interval, False),
    ]
    for obs_is_start, op in enumerate(obs_pts):
        for src_is_start, sp in enumerate(src_pts):
            if float(np.linalg.norm(op - sp)) <= float(tol):
                return bool(obs_is_start == 0), bool(src_is_start == 0)
    return None


def _integrate_linear_pair_box(
    obs_elem: LinearElement,
    src_elem: LinearElement,
    kernel_eval: Callable[[np.ndarray, np.ndarray], complex],
    obs_interval: Tuple[float, float],
    src_interval: Tuple[float, float],
    obs_order: int,
    src_order: int,
) -> np.ndarray:
    qt_obs, qw_obs = _get_quadrature(max(2, int(obs_order)))
    qt_src, qw_src = _get_quadrature(max(2, int(src_order)))
    obs_scale = max(float(obs_interval[1]) - float(obs_interval[0]), 0.0)
    src_scale = max(float(src_interval[1]) - float(src_interval[0]), 0.0)
    obs_len = float(obs_elem.length) * obs_scale
    src_len = float(src_elem.length) * src_scale
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block

    for tobs, wobs in zip(qt_obs, qw_obs):
        xi_obs = float(obs_interval[0]) + obs_scale * float(tobs)
        phi_obs = _linear_shape_values(xi_obs)
        robs = _linear_param_to_point(obs_elem, xi_obs)
        for tsrc, wsrc in zip(qt_src, qw_src):
            xi_src = float(src_interval[0]) + src_scale * float(tsrc)
            phi_src = _linear_shape_values(xi_src)
            rsrc = _linear_param_to_point(src_elem, xi_src)
            kval = complex(kernel_eval(robs, rsrc))
            block += (float(wobs) * float(wsrc) * kval) * np.outer(phi_obs, phi_src)

    return block * obs_len * src_len


def _integrate_linear_self_duffy(
    elem: LinearElement,
    kernel_eval: Callable[[np.ndarray, np.ndarray], complex],
    interval: Tuple[float, float],
    order: int = 20,
) -> np.ndarray:
    qt, qw = _get_quadrature(max(4, int(order)))
    a, b = float(interval[0]), float(interval[1])
    h = max(b - a, 0.0)
    elem_len = float(elem.length) * h
    block = np.zeros((2, 2), dtype=np.complex128)
    if elem_len <= 0.0:
        return block

    for u, wu in zip(qt, qw):
        uu = float(u)
        jac_outer = float(wu) * uu
        t_major = a + h * uu
        s_major = t_major
        robs_major = _linear_param_to_point(elem, t_major)
        rsrc_major = _linear_param_to_point(elem, s_major)
        phi_t_major = _linear_shape_values(t_major)
        phi_s_major = _linear_shape_values(s_major)
        for v, wv in zip(qt, qw):
            vv = float(v)
            weight = jac_outer * float(wv)
            # Triangle: s <= t
            xi_t = a + h * uu
            xi_s = a + h * (uu * vv)
            phi_t = _linear_shape_values(xi_t)
            phi_s = _linear_shape_values(xi_s)
            robs = _linear_param_to_point(elem, xi_t)
            rsrc = _linear_param_to_point(elem, xi_s)
            block += weight * complex(kernel_eval(robs, rsrc)) * np.outer(phi_t, phi_s)
            # Triangle: t <= s
            xi_t2 = a + h * (uu * vv)
            xi_s2 = a + h * uu
            phi_t2 = _linear_shape_values(xi_t2)
            phi_s2 = _linear_shape_values(xi_s2)
            robs2 = _linear_param_to_point(elem, xi_t2)
            rsrc2 = _linear_param_to_point(elem, xi_s2)
            block += weight * complex(kernel_eval(robs2, rsrc2)) * np.outer(phi_t2, phi_s2)

    return block * (elem_len * elem_len)


def _integrate_linear_touching_duffy(
    obs_elem: LinearElement,
    src_elem: LinearElement,
    kernel_eval: Callable[[np.ndarray, np.ndarray], complex],
    obs_interval: Tuple[float, float],
    src_interval: Tuple[float, float],
    obs_start_is_shared: bool,
    src_start_is_shared: bool,
    order: int = 20,
) -> np.ndarray:
    qt, qw = _get_quadrature(max(4, int(order)))
    obs_len = _linear_interval_length(obs_elem, obs_interval)
    src_len = _linear_interval_length(src_elem, src_interval)
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block

    for u, wu in zip(qt, qw):
        uu = float(u)
        jac_outer = float(wu) * uu
        for v, wv in zip(qt, qw):
            vv = float(v)
            weight = jac_outer * float(wv)
            # Triangle 1: source local distance <= observation local distance
            xi_obs = _linear_map_local_to_parent(obs_interval, uu, obs_start_is_shared)
            xi_src = _linear_map_local_to_parent(src_interval, uu * vv, src_start_is_shared)
            phi_obs = _linear_shape_values(xi_obs)
            phi_src = _linear_shape_values(xi_src)
            robs = _linear_param_to_point(obs_elem, xi_obs)
            rsrc = _linear_param_to_point(src_elem, xi_src)
            block += weight * complex(kernel_eval(robs, rsrc)) * np.outer(phi_obs, phi_src)
            # Triangle 2: observation local distance <= source local distance
            xi_obs2 = _linear_map_local_to_parent(obs_interval, uu * vv, obs_start_is_shared)
            xi_src2 = _linear_map_local_to_parent(src_interval, uu, src_start_is_shared)
            phi_obs2 = _linear_shape_values(xi_obs2)
            phi_src2 = _linear_shape_values(xi_src2)
            robs2 = _linear_param_to_point(obs_elem, xi_obs2)
            rsrc2 = _linear_param_to_point(src_elem, xi_src2)
            block += weight * complex(kernel_eval(robs2, rsrc2)) * np.outer(phi_obs2, phi_src2)

    return block * (obs_len * src_len)


def _integrate_linear_pair_recursive(
    obs_elem: LinearElement,
    src_elem: LinearElement,
    kernel_eval: Callable[[np.ndarray, np.ndarray], complex],
    obs_interval: Tuple[float, float],
    src_interval: Tuple[float, float],
    obs_order: int,
    src_order: int,
    depth: int = 0,
    max_depth: int = 3,
) -> np.ndarray:
    obs_len = _linear_interval_length(obs_elem, obs_interval)
    src_len = _linear_interval_length(src_elem, src_interval)
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block

    same_elem_same_interval = (
        obs_elem.panel_index == src_elem.panel_index
        and abs(float(obs_interval[0]) - float(src_interval[0])) <= 1.0e-15
        and abs(float(obs_interval[1]) - float(src_interval[1])) <= 1.0e-15
    )
    if same_elem_same_interval:
        order = max(6, int(max(obs_order, src_order)) + 1)
        return _integrate_linear_self_duffy(
            obs_elem,
            kernel_eval,
            interval=obs_interval,
            order=order,
        )

    shared = _linear_shared_interval_endpoint_info(obs_elem, obs_interval, src_elem, src_interval)
    if shared is not None:
        order = max(6, int(max(obs_order, src_order)) + 1)
        return _integrate_linear_touching_duffy(
            obs_elem,
            src_elem,
            kernel_eval,
            obs_interval=obs_interval,
            src_interval=src_interval,
            obs_start_is_shared=bool(shared[0]),
            src_start_is_shared=bool(shared[1]),
            order=order,
        )

    obs_mid = _linear_interval_midpoint(obs_elem, obs_interval)
    src_mid = _linear_interval_midpoint(src_elem, src_interval)
    distance = float(np.linalg.norm(obs_mid - src_mid))
    scale = max(obs_len, src_len, EPS)
    ratio = distance / scale

    # Refine near-singular element pairs adaptively before falling back to tensor Gauss.
    if depth < max_depth and ratio < 0.95:
        oa, ob = float(obs_interval[0]), float(obs_interval[1])
        sa, sb = float(src_interval[0]), float(src_interval[1])
        if ratio < 0.16:
            om = 0.5 * (oa + ob)
            sm = 0.5 * (sa + sb)
            sub_obs = [(oa, om), (om, ob)]
            sub_src = [(sa, sm), (sm, sb)]
            for oi in sub_obs:
                for si in sub_src:
                    block += _integrate_linear_pair_recursive(
                        obs_elem,
                        src_elem,
                        kernel_eval,
                        oi,
                        si,
                        obs_order=obs_order,
                        src_order=src_order,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
            return block
        if obs_len >= src_len:
            om = 0.5 * (oa + ob)
            return (
                _integrate_linear_pair_recursive(
                    obs_elem, src_elem, kernel_eval, (oa, om), src_interval,
                    obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
                )
                + _integrate_linear_pair_recursive(
                    obs_elem, src_elem, kernel_eval, (om, ob), src_interval,
                    obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
                )
            )
        sm = 0.5 * (sa + sb)
        return (
            _integrate_linear_pair_recursive(
                obs_elem, src_elem, kernel_eval, obs_interval, (sa, sm),
                obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
            )
            + _integrate_linear_pair_recursive(
                obs_elem, src_elem, kernel_eval, obs_interval, (sm, sb),
                obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
            )
        )

    adapt_order, _ = _near_singular_scheme(distance, scale)
    tensor_order = max(int(max(obs_order, src_order)), min(16, int(max(5, adapt_order))))
    return _integrate_linear_pair_box(
        obs_elem,
        src_elem,
        kernel_eval,
        obs_interval=obs_interval,
        src_interval=src_interval,
        obs_order=tensor_order,
        src_order=tensor_order,
    )


def _integrate_linear_pair_generic(
    obs_elem: LinearElement,
    src_elem: LinearElement,
    kernel_eval: Callable[[np.ndarray, np.ndarray], complex],
    obs_order: int = 6,
    src_order: int = 6,
) -> np.ndarray:
    """
    Assemble a 2x2 Galerkin block for one observation/source element pair.

    This upgraded implementation keeps the straight-element tensor-Gauss backbone but
    adds two accuracy-critical improvements for the experimental linear/Galerkin path:
    - Duffy-type quadrature for same-element and endpoint-touching singular pairs
    - adaptive recursive interval subdivision for near-singular pairs
    """

    return _integrate_linear_pair_recursive(
        obs_elem,
        src_elem,
        kernel_eval,
        obs_interval=(0.0, 1.0),
        src_interval=(0.0, 1.0),
        obs_order=obs_order,
        src_order=src_order,
        depth=0,
        max_depth=6,
    )


def _stable_hankel2_array(order: int, x: np.ndarray) -> np.ndarray:
    """Robust array Hankel evaluator for real and complex arguments.

    Uses scaled SciPy Hankel for complex arguments when available, then repairs
    any remaining non-finite entries with the existing scalar helpers.
    """

    z = np.asarray(x, dtype=np.complex128)
    out: np.ndarray | None = None
    if _SCIPY_SPECIAL is not None:
        try:
            # Real fast path when possible.
            if np.all(np.abs(z.imag) <= 1e-14) and np.all(z.real >= 0.0):
                xr = np.maximum(z.real.astype(float, copy=False), 1e-12)
                if order == 0:
                    out = np.asarray(_SCIPY_SPECIAL.j0(xr) - 1j * _SCIPY_SPECIAL.y0(xr), dtype=np.complex128)
                else:
                    out = np.asarray(_SCIPY_SPECIAL.j1(xr) - 1j * _SCIPY_SPECIAL.y1(xr), dtype=np.complex128)
            elif hasattr(_SCIPY_SPECIAL, 'hankel2e'):
                scaled = np.asarray(_SCIPY_SPECIAL.hankel2e(order, z), dtype=np.complex128)
                out = scaled * np.exp(-1j * z)
            else:
                out = np.asarray(_SCIPY_SPECIAL.hankel2(order, z), dtype=np.complex128)
        except Exception:
            out = None
    if out is None:
        vec = np.vectorize(_hankel2_0 if order == 0 else _hankel2_1, otypes=[np.complex128])
        return np.asarray(vec(z), dtype=np.complex128)

    finite = np.isfinite(out.real) & np.isfinite(out.imag)
    if not np.all(finite):
        vec = np.vectorize(_hankel2_0 if order == 0 else _hankel2_1, otypes=[np.complex128])
        repaired = np.asarray(vec(z[~finite]), dtype=np.complex128)
        out = np.asarray(out, dtype=np.complex128)
        out[~finite] = repaired
    return np.asarray(out, dtype=np.complex128)


def _hankel2_0_array(x: np.ndarray) -> np.ndarray:
    return _stable_hankel2_array(0, x)


def _hankel2_1_array(x: np.ndarray) -> np.ndarray:
    return _stable_hankel2_array(1, x)


def _green_2d_array(k0: complex | float, r: np.ndarray) -> np.ndarray:
    rr = np.maximum(np.asarray(r, dtype=float), EPS)
    x = np.asarray(complex(k0) * rr, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    return 0.25j * _hankel2_0_array(x)


def _dgreen_dn_obs_array(k0: complex | float, r_vec: np.ndarray, n_obs: np.ndarray) -> np.ndarray:
    rr = np.linalg.norm(r_vec, axis=1)
    out = np.zeros(rr.shape[0], dtype=np.complex128)
    mask = rr > EPS
    if not np.any(mask):
        return out
    rrm = rr[mask]
    x = np.asarray(complex(k0) * rrm, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    h1 = _hankel2_1_array(x)
    projection = (r_vec[mask] @ np.asarray(n_obs, dtype=float)) / rrm
    out[mask] = (-0.25j * complex(k0)) * h1 * projection
    return out


def _dgreen_dn_src_array(k0: complex | float, r_vec: np.ndarray, n_src: np.ndarray) -> np.ndarray:
    rr = np.linalg.norm(r_vec, axis=1)
    out = np.zeros(rr.shape[0], dtype=np.complex128)
    mask = rr > EPS
    if not np.any(mask):
        return out
    rrm = rr[mask]
    x = np.asarray(complex(k0) * rrm, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    h1 = _hankel2_1_array(x)
    projection = np.sum(np.asarray(n_src, dtype=float)[mask] * r_vec[mask], axis=1) / rrm
    out[mask] = (0.25j * complex(k0)) * h1 * projection
    return out


def _linear_pair_far_mask(
    elements: List[LinearElement],
    obs_index: int,
    centers: np.ndarray,
    lengths: np.ndarray,
    node_ids: np.ndarray,
    far_ratio: float,
) -> np.ndarray:
    obs_ids = node_ids[obs_index]
    shared = np.any(node_ids == obs_ids[0], axis=1) | np.any(node_ids == obs_ids[1], axis=1)
    dist = np.linalg.norm(centers - centers[obs_index], axis=1)
    scale = np.maximum(np.maximum(lengths, lengths[obs_index]), EPS)
    far = (dist / scale) >= float(far_ratio)
    far[obs_index] = False
    far &= ~shared
    return far


def _assemble_linear_far_blocks_for_obs(
    obs_elem: LinearElement,
    src_elems: List[LinearElement],
    k0: complex | float,
    obs_normal_deriv: bool,
    obs_order: int,
    src_order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised far-pair 2x2 block assembly for one observation element."""
    m = len(src_elems)
    if m == 0:
        return (
            np.zeros((0, 2, 2), dtype=np.complex128),
            np.zeros((0, 2, 2), dtype=np.complex128),
        )

    qt_obs, qw_obs = _get_quadrature(max(2, int(obs_order)))
    qt_src, qw_src = _get_quadrature(max(2, int(src_order)))
    obs_seg = obs_elem.p1 - obs_elem.p0
    src_p0 = np.stack([e.p0 for e in src_elems], axis=0)
    src_seg = np.stack([e.p1 - e.p0 for e in src_elems], axis=0)
    src_normals = np.stack([e.normal for e in src_elems], axis=0)
    src_lengths = np.asarray([e.length for e in src_elems], dtype=float)
    s_blocks = np.zeros((m, 2, 2), dtype=np.complex128)
    k_blocks = np.zeros((m, 2, 2), dtype=np.complex128)

    for tobs, wobs in zip(qt_obs, qw_obs):
        tobs_f = float(tobs)
        phi_obs = _linear_shape_values(tobs_f)
        robs = obs_elem.p0 + tobs_f * obs_seg
        for tsrc, wsrc in zip(qt_src, qw_src):
            tsrc_f = float(tsrc)
            phi_src = _linear_shape_values(tsrc_f)
            rsrc = src_p0 + tsrc_f * src_seg
            diff = robs[None, :] - rsrc
            kval_s = _green_2d_array(k0, np.linalg.norm(diff, axis=1))
            if obs_normal_deriv:
                kval_k = _dgreen_dn_obs_array(k0, diff, obs_elem.normal)
            else:
                kval_k = _dgreen_dn_src_array(k0, diff, src_normals)
            outer = np.outer(phi_obs, phi_src)[None, :, :]
            w = float(wobs) * float(wsrc)
            s_blocks += w * kval_s[:, None, None] * outer
            k_blocks += w * kval_k[:, None, None] * outer

    scale = float(obs_elem.length) * src_lengths[:, None, None]
    s_blocks *= scale
    k_blocks *= scale
    return s_blocks, k_blocks


def _single_layer_block_linear(
    obs_elem: LinearElement,
    src_elem: LinearElement,
    k0: complex | float,
    obs_order: int = 8,
    src_order: int = 8,
) -> np.ndarray:
    return _integrate_linear_pair_generic(
        obs_elem,
        src_elem,
        lambda robs, rsrc: _green_2d(k0, max(float(np.linalg.norm(robs - rsrc)), EPS)),
        obs_order=obs_order,
        src_order=src_order,
    )


def _double_layer_block_linear(
    obs_elem: LinearElement,
    src_elem: LinearElement,
    k0: complex | float,
    obs_normal_deriv: bool,
    obs_order: int = 8,
    src_order: int = 8,
) -> np.ndarray:
    if obs_normal_deriv:
        return _integrate_linear_pair_generic(
            obs_elem,
            src_elem,
            lambda robs, rsrc: _dgreen_dn_obs(k0, robs - rsrc, obs_elem.normal),
            obs_order=obs_order,
            src_order=src_order,
        )
    return _integrate_linear_pair_generic(
        obs_elem,
        src_elem,
        lambda robs, rsrc: _dgreen_dn_src(k0, robs - rsrc, src_elem.normal),
        obs_order=obs_order,
        src_order=src_order,
    )


def _assemble_linear_operator_matrices(
    mesh: LinearMesh,
    k0: complex | float,
    obs_normal_deriv: bool,
    obs_order: int = 8,
    src_order: int = 8,
    far_ratio: float = 3.0,
    source_element_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble dense linear-Galerkin S and K/K' matrices on global nodal DOFs.

    This upgraded implementation preserves the exact scalar recursive quadrature for
    self/touching/near interactions, but batches smooth far interactions through
    vectorised Hankel evaluation. That targets the main runtime sink without changing
    the hard-case math on near-singular element pairs.

    When `source_element_mask` is provided, only source elements with a true mask entry
    contribute columns to the assembled operator. Observation/testing elements still span
    the full mesh so the returned matrices remain global nodal operators.
    """

    nnodes = len(mesh.nodes)
    s_mat = np.zeros((nnodes, nnodes), dtype=np.complex128)
    k_mat = np.zeros((nnodes, nnodes), dtype=np.complex128)
    elements = list(mesh.elements)
    if not elements:
        return s_mat, k_mat

    if source_element_mask is None:
        src_mask = np.ones(len(elements), dtype=bool)
    else:
        src_mask = np.asarray(source_element_mask, dtype=bool).reshape(-1)
        if src_mask.size != len(elements):
            raise ValueError("source_element_mask length must match mesh element count.")
    if not np.any(src_mask):
        return s_mat, k_mat

    centers = np.stack([e.center for e in elements], axis=0)
    lengths = np.asarray([e.length for e in elements], dtype=float)
    node_ids = np.asarray([e.node_ids for e in elements], dtype=int)

    for obs_index, obs_elem in enumerate(elements):
        obs_ids = np.asarray(obs_elem.node_ids, dtype=int)
        far_mask = _linear_pair_far_mask(
            elements=elements,
            obs_index=obs_index,
            centers=centers,
            lengths=lengths,
            node_ids=node_ids,
            far_ratio=far_ratio,
        )
        far_mask &= src_mask
        far_idx = np.flatnonzero(far_mask)
        if far_idx.size > 0:
            far_src_elems = [elements[int(i)] for i in far_idx]
            s_blocks, k_blocks = _assemble_linear_far_blocks_for_obs(
                obs_elem=obs_elem,
                src_elems=far_src_elems,
                k0=k0,
                obs_normal_deriv=obs_normal_deriv,
                obs_order=obs_order,
                src_order=src_order,
            )
            src_ids = node_ids[far_idx]
            rows = np.broadcast_to(obs_ids[[0, 0, 1, 1]], (far_idx.size, 4))
            cols = src_ids[:, [0, 1, 0, 1]]
            np.add.at(s_mat, (rows, cols), s_blocks.reshape(far_idx.size, 4))
            np.add.at(k_mat, (rows, cols), k_blocks.reshape(far_idx.size, 4))

        near_idx = np.flatnonzero((~far_mask) & src_mask)
        for j in near_idx:
            src_elem = elements[int(j)]
            src_ids = src_elem.node_ids
            s_blk = _single_layer_block_linear(
                obs_elem=obs_elem,
                src_elem=src_elem,
                k0=k0,
                obs_order=obs_order,
                src_order=src_order,
            )
            k_blk = _double_layer_block_linear(
                obs_elem=obs_elem,
                src_elem=src_elem,
                k0=k0,
                obs_normal_deriv=obs_normal_deriv,
                obs_order=obs_order,
                src_order=src_order,
            )
            s_mat[np.ix_(obs_ids, src_ids)] += s_blk
            k_mat[np.ix_(obs_ids, src_ids)] += k_blk
    return s_mat, k_mat

def _build_linear_coupled_infos(
    mesh: LinearMesh,
    materials: MaterialLibrary,
    freq_ghz: float,
    pol: str,
    k0: float,
) -> List[PanelCoupledInfo]:
    pseudo_panels = [
        Panel(
            name=e.name,
            seg_type=e.seg_type,
            ibc_flag=e.ibc_flag,
            ipn1=e.ipn1,
            ipn2=e.ipn2,
            p0=e.p0,
            p1=e.p1,
            center=e.center,
            tangent=e.tangent,
            normal=e.normal,
            length=e.length,
        )
        for e in mesh.elements
    ]
    return _build_coupled_panel_info(pseudo_panels, materials, freq_ghz, pol, k0)


def _linear_element_incident_load_many(
    elem: LinearElement,
    k_air: float,
    elevations_deg: np.ndarray,
    order: int = 8,
) -> np.ndarray:
    qt, qw = _get_quadrature(max(2, int(order)))
    seg = elem.p1 - elem.p0
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    phi = np.deg2rad(elev)
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    out = np.zeros((2, elev.size), dtype=np.complex128)
    for t, w in zip(qt, qw):
        shape = _linear_shape_values(float(t))[:, None]
        rp = elem.p0 + float(t) * seg
        phase = np.exp((1j * k_air) * (dirs @ rp))
        out += float(w) * shape * phase[None, :]
    return out * float(elem.length)


def _build_coupled_rhs_many_linear(
    mesh: LinearMesh,
    infos: List[PanelCoupledInfo],
    k_air: float,
    elevations_deg: np.ndarray,
) -> np.ndarray:
    """
    Build tested incident-field load vectors on linear nodal DOFs.

    Returns an array of shape (2 * nnodes, E) corresponding to the future nodal
    unknown ordering [U_trace_nodes, Q_minus_nodes].
    """

    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    rhs = np.zeros((2 * nnodes, elev.size), dtype=np.complex128)
    for elem, info in zip(mesh.elements, infos):
        local = _linear_element_incident_load_many(elem, k_air=k_air, elevations_deg=elev)
        ids = elem.node_ids
        active_is_minus = info.minus_region >= 0
        if info.minus_has_incident if active_is_minus else info.plus_has_incident:
            rhs[np.asarray(ids, dtype=int), :] += local
        if info.bc_kind == "transmission":
            passive_has_inc = info.plus_has_incident if active_is_minus else info.minus_has_incident
            if passive_has_inc:
                rhs[nnodes + np.asarray(ids, dtype=int), :] += local
    return rhs


def _backscatter_rcs_coupled_many_linear(
    mesh: LinearMesh,
    infos: List[PanelCoupledInfo],
    u_trace_nodes_mat: np.ndarray,
    q_minus_nodes_mat: np.ndarray,
    k_air: float,
    elevations_deg: np.ndarray,
    order: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear-element far-field projector for the future coupled Galerkin solve.

    This helper already evaluates the backscatter integral from nodal traces/fluxes;
    it is intended to be used once the linear coupled system assembly is wired in.
    """

    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    u_eval = np.asarray(u_trace_nodes_mat, dtype=np.complex128)
    q_eval = np.asarray(q_minus_nodes_mat, dtype=np.complex128)
    if u_eval.ndim == 1:
        u_eval = u_eval.reshape(-1, 1)
    if q_eval.ndim == 1:
        q_eval = q_eval.reshape(-1, 1)
    nnodes = len(mesh.nodes)
    if u_eval.shape != q_eval.shape or u_eval.shape[0] != nnodes:
        raise ValueError("Linear nodal trace/flux arrays must have shape (nnodes, nelevations).")
    if u_eval.shape[1] != elev.size:
        raise ValueError("Linear nodal solution columns must match elevation count.")

    phi = np.deg2rad(elev)
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    qt, qw = _get_quadrature(max(2, int(order)))
    amp = np.zeros(elev.size, dtype=np.complex128)

    for elem, info in zip(mesh.elements, infos):
        ids = np.asarray(elem.node_ids, dtype=int)
        beta = complex(info.q_plus_beta)
        gamma = complex(info.q_plus_gamma)
        u_local = u_eval[ids, :]
        q_minus_local = q_eval[ids, :]
        q_plus_local = beta * q_minus_local + gamma * u_local
        for t, w in zip(qt, qw):
            shape = _linear_shape_values(float(t))[:, None]
            rp = elem.p0 + float(t) * (elem.p1 - elem.p0)
            phase = np.exp((1j * k_air) * (dirs @ rp))
            dot_scatter = dirs @ elem.normal
            u_t = np.sum(shape * u_local, axis=0)
            q_minus_t = np.sum(shape * q_minus_local, axis=0)
            q_plus_t = np.sum(shape * q_plus_local, axis=0)
            if info.minus_has_incident:
                amp += float(w) * float(elem.length) * phase * (
                    -q_minus_t + 1j * k_air * dot_scatter * u_t
                )
            if info.plus_has_incident:
                amp += float(w) * float(elem.length) * phase * (
                    q_plus_t - 1j * k_air * dot_scatter * u_t
                )

    sigma_lin = _rcs_sigma_from_amp(amp, k_air)
    return np.asarray(sigma_lin, dtype=float), np.asarray(amp, dtype=np.complex128)



def _linear_mass_block(elem: LinearElement) -> np.ndarray:
    """Consistent 2-node boundary mass matrix on one straight element."""

    l = float(elem.length)
    return l * np.asarray([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]], dtype=np.complex128)



def _linear_coupled_interface_signature(elem: LinearElement, info: PanelCoupledInfo) -> Tuple[Any, ...]:
    return (
        int(elem.seg_type),
        int(elem.ibc_flag),
        int(elem.ipn1),
        int(elem.ipn2),
        int(info.minus_region),
        int(info.plus_region),
        str(info.bc_kind),
    )



def _linear_coupled_node_report(
    mesh: LinearMesh,
    infos: List[PanelCoupledInfo],
) -> Dict[str, int]:
    """
    Summarize node configurations for the nodal coupled solve.

    The production-hardened linear/Galerkin path now handles shared geometric junctions by
    augmenting the nodal system with trace-continuity and region-wise flux-balance rows.
    We still report branching and mixed-interface node counts for diagnostics, but they are
    no longer treated as automatic blockers by themselves.
    """

    incident: Dict[int, List[int]] = {}
    for eidx, elem in enumerate(mesh.elements):
        for nid in elem.node_ids:
            incident.setdefault(int(nid), []).append(int(eidx))

    branching_nodes = 0
    mixed_interface_nodes = 0
    for nid, elem_ids in incident.items():
        unique = sorted(set(int(v) for v in elem_ids))
        if len(unique) <= 1:
            continue
        sigs = {
            _linear_coupled_interface_signature(mesh.elements[eidx], infos[eidx])
            for eidx in unique
        }
        if len(unique) > 2:
            branching_nodes += 1
        if len(sigs) > 1:
            mixed_interface_nodes += 1

    return {
        "linear_node_count": int(len(mesh.nodes)),
        "linear_element_count": int(len(mesh.elements)),
        "linear_branching_nodes": int(branching_nodes),
        "linear_mixed_interface_nodes": int(mixed_interface_nodes),
        "linear_unsupported_nodes": 0,
    }



def _build_linear_junction_constraints(
    mesh: LinearMesh,
    infos: List[PanelCoupledInfo],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build nodal junction constraints for the linear/Galerkin coupled solve.

    The linear trace unknown is continuous only across explicitly shared nodes. When the
    interface-aware mesh intentionally splits nodes at the same geometric coordinate, we
    restore pointwise continuity at true shared geometric junctions with explicit trace
    constraints. We also add region-wise flux-balance constraints using the endpoint sign
    convention from the pulse/collocation junction treatment.
    """

    nnodes = len(mesh.nodes)
    grouped: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    for eidx, elem in enumerate(mesh.elements):
        n0, n1 = (int(v) for v in elem.node_ids)
        grouped.setdefault(mesh.nodes[n0].key, []).append((int(eidx), 0, n0))
        grouped.setdefault(mesh.nodes[n1].key, []).append((int(eidx), 1, n1))

    rows: List[np.ndarray] = []
    trace_count = 0
    flux_count = 0
    junction_nodes = 0
    orientation_conflict_nodes = 0
    constrained_nodes: Set[int] = set()
    constrained_elems: Set[int] = set()

    for entries in grouped.values():
        unique_elems = sorted({int(eidx) for eidx, _, _ in entries})
        unique_nodes = sorted({int(nid) for _, _, nid in entries})
        if len(unique_elems) < 2 and len(unique_nodes) < 2:
            continue

        by_elem_sign: Dict[int, int] = {}
        seg_names: Set[str] = set()
        region_set: Set[int] = set()
        for eidx, local_end, nid in entries:
            endpoint_sign = +1 if int(local_end) == 0 else -1
            by_elem_sign[int(eidx)] = by_elem_sign.get(int(eidx), 0) + endpoint_sign
            seg_names.add(mesh.elements[int(eidx)].name)
            info = infos[int(eidx)]
            if info.minus_region >= 0:
                region_set.add(int(info.minus_region))
            if info.plus_region >= 0:
                region_set.add(int(info.plus_region))

        if len(seg_names) >= 2:
            signs = [int(np.sign(by_elem_sign.get(eidx, 0))) for eidx in unique_elems]
            has_pos = any(s > 0 for s in signs)
            has_neg = any(s < 0 for s in signs)
            if not (has_pos and has_neg):
                orientation_conflict_nodes += 1

        if len(unique_nodes) > 1:
            ref_nid = unique_nodes[0]
            for other_nid in unique_nodes[1:]:
                row = np.zeros(2 * nnodes, dtype=np.complex128)
                row[ref_nid] = 1.0 + 0.0j
                row[other_nid] = -1.0 + 0.0j
                rows.append(row)
                trace_count += 1
                constrained_nodes.add(ref_nid)
                constrained_nodes.add(other_nid)

        for region in sorted(region_set):
            row = np.zeros(2 * nnodes, dtype=np.complex128)
            terms = 0
            for eidx, local_end, nid in entries:
                endpoint_sign = +1 if int(local_end) == 0 else -1
                info = infos[int(eidx)]
                coeff_u = 0.0 + 0.0j
                coeff_q = 0.0 + 0.0j
                participates = False
                if info.minus_region == region:
                    coeff_q += 1.0 + 0.0j
                    participates = True
                if info.plus_region == region:
                    coeff_u += complex(info.q_plus_gamma)
                    coeff_q += complex(info.q_plus_beta)
                    participates = True
                if not participates:
                    continue

                w = complex(float(endpoint_sign), 0.0)
                nid_i = int(nid)
                row[nid_i] += w * coeff_u
                row[nnodes + nid_i] += w * coeff_q
                terms += 1
                constrained_nodes.add(nid_i)
                constrained_elems.add(int(eidx))

            if terms >= 2 and np.linalg.norm(row) > 0.0:
                rows.append(row)
                flux_count += 1

        junction_nodes += 1

    if not rows:
        return np.zeros((0, 2 * nnodes), dtype=np.complex128), {
            "junction_nodes": 0,
            "junction_constraints": 0,
            "junction_panels": 0,
            "junction_trace_constraints": 0,
            "junction_flux_constraints": 0,
            "junction_orientation_conflict_nodes": int(orientation_conflict_nodes),
        }

    c_mat = np.vstack(rows)
    return c_mat, {
        "junction_nodes": int(junction_nodes),
        "junction_constraints": int(c_mat.shape[0]),
        "junction_panels": int(len(constrained_elems)),
        "junction_trace_constraints": int(trace_count),
        "junction_flux_constraints": int(flux_count),
        "junction_orientation_conflict_nodes": int(orientation_conflict_nodes),
    }


def _ensure_finite_linear_system(a_mat: np.ndarray, rhs: np.ndarray | None = None, label: str = "linear system") -> None:
    """Raise a clear error before calling LAPACK if the assembled system contains NaN/Inf."""

    a_eval = np.asarray(a_mat)
    if not np.all(np.isfinite(a_eval)):
        bad = np.argwhere(~np.isfinite(a_eval))
        first = tuple(int(v) for v in bad[0]) if bad.size else None
        raise ValueError(f"{label}: system matrix contains NaN/Inf at index {first}.")
    if rhs is None:
        return
    b_eval = np.asarray(rhs)
    if not np.all(np.isfinite(b_eval)):
        bad = np.argwhere(~np.isfinite(b_eval))
        first = tuple(int(v) for v in bad[0]) if bad.size else None
        raise ValueError(f"{label}: RHS contains NaN/Inf at index {first}.")



def _assemble_linear_mass_matrix(mesh: LinearMesh) -> np.ndarray:
    """Assemble the global consistent mass matrix for the linear boundary mesh."""

    nnodes = len(mesh.nodes)
    m_mat = np.zeros((nnodes, nnodes), dtype=np.complex128)
    for elem in mesh.elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        m_mat[np.ix_(ids, ids)] += _linear_mass_block(elem)
    return m_mat


@dataclass
class LinearCoupledNodeInfo:
    """Per-node coupled metadata for the global linear/Galerkin assembly."""

    active_region: int
    passive_region: int
    bc_kind: str
    robin_impedance: complex
    coeff_u_active: complex
    coeff_q_active: complex
    eps_phys: complex
    mu_phys: complex
    k_phys: complex
    q_plus_beta: complex
    q_plus_gamma: complex
    plus_region: int


def _build_linear_coupled_node_infos(
    mesh: LinearMesh,
    infos: List[PanelCoupledInfo],
) -> List[LinearCoupledNodeInfo]:
    """
    Derive one consistent coupled-interface record per nodal test/unknown DOF.

    The interface-aware linear mesh is expected to share a node only across elements with
    the same physical interface signature. We still verify that the incident elements agree
    on the metadata needed by the global nodal assembly.
    """

    incident: Dict[int, List[int]] = {}
    for eidx, elem in enumerate(mesh.elements):
        for nid in elem.node_ids:
            incident.setdefault(int(nid), []).append(int(eidx))

    def _complex_close(a: complex, b: complex, tol: float = 1.0e-10) -> bool:
        return abs(complex(a) - complex(b)) <= tol * max(1.0, abs(complex(a)), abs(complex(b)))

    node_infos: List[LinearCoupledNodeInfo | None] = [None] * len(mesh.nodes)
    for nid in range(len(mesh.nodes)):
        elem_ids = incident.get(int(nid), [])
        if not elem_ids:
            raise ValueError(f"Linear coupled node {nid} is not attached to any element.")
        ref = infos[int(elem_ids[0])]
        active_region = int(ref.minus_region if ref.minus_region >= 0 else ref.plus_region)
        passive_region = int(ref.plus_region if active_region == ref.minus_region else ref.minus_region)
        coeff_u_active, coeff_q_active = _region_side_trace_coefficients(ref, active_region)
        eps_phys = ref.eps_minus if active_region == ref.minus_region else ref.eps_plus
        mu_phys = ref.mu_minus if active_region == ref.minus_region else ref.mu_plus
        k_phys = ref.k_minus if active_region == ref.minus_region else ref.k_plus
        expected = {
            'active_region': active_region,
            'passive_region': passive_region,
            'bc_kind': str(ref.bc_kind),
            'robin_impedance': complex(ref.robin_impedance),
            'coeff_u_active': complex(coeff_u_active),
            'coeff_q_active': complex(coeff_q_active),
            'eps_phys': complex(eps_phys),
            'mu_phys': complex(mu_phys),
            'k_phys': complex(k_phys),
            'q_plus_beta': complex(ref.q_plus_beta),
            'q_plus_gamma': complex(ref.q_plus_gamma),
            'plus_region': int(ref.plus_region),
        }
        for eidx in elem_ids[1:]:
            info = infos[int(eidx)]
            active_chk = int(info.minus_region if info.minus_region >= 0 else info.plus_region)
            passive_chk = int(info.plus_region if active_chk == info.minus_region else info.minus_region)
            coeff_u_chk, coeff_q_chk = _region_side_trace_coefficients(info, active_chk)
            eps_chk = info.eps_minus if active_chk == info.minus_region else info.eps_plus
            mu_chk = info.mu_minus if active_chk == info.minus_region else info.mu_plus
            k_chk = info.k_minus if active_chk == info.minus_region else info.k_plus
            actual = {
                'active_region': active_chk,
                'passive_region': passive_chk,
                'bc_kind': str(info.bc_kind),
                'robin_impedance': complex(info.robin_impedance),
                'coeff_u_active': complex(coeff_u_chk),
                'coeff_q_active': complex(coeff_q_chk),
                'eps_phys': complex(eps_chk),
                'mu_phys': complex(mu_chk),
                'k_phys': complex(k_chk),
                'q_plus_beta': complex(info.q_plus_beta),
                'q_plus_gamma': complex(info.q_plus_gamma),
                'plus_region': int(info.plus_region),
            }
            if actual['active_region'] != expected['active_region'] or actual['passive_region'] != expected['passive_region'] or actual['bc_kind'] != expected['bc_kind'] or actual['plus_region'] != expected['plus_region']:
                raise ValueError(
                    "Linear/Galerkin nodal assembly encountered incompatible incident interface metadata "
                    f"at node {nid}."
                )
            for key in ('robin_impedance', 'coeff_u_active', 'coeff_q_active', 'eps_phys', 'mu_phys', 'k_phys', 'q_plus_beta', 'q_plus_gamma'):
                if not _complex_close(expected[key], actual[key]):
                    raise ValueError(
                        "Linear/Galerkin nodal assembly encountered inconsistent material coefficients "
                        f"at node {nid}."
                    )

        node_infos[nid] = LinearCoupledNodeInfo(**expected)

    return [ni for ni in node_infos if ni is not None]


def _build_linear_coupled_region_operators(
    mesh: LinearMesh,
    infos: List[PanelCoupledInfo],
    obs_order: int = 5,
    src_order: int = 5,
    far_ratio: float = 3.0,
) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Assemble reusable nodal S/K operators for each region and interface side.

    Returns `region_ops[region]['minus'|'plus'] = (S, K)` where the matrices already
    include only source elements whose minus/plus side belongs to the requested region.
    """

    region_to_k: Dict[int, complex] = {}
    for info in infos:
        if info.minus_region >= 0:
            region_to_k[int(info.minus_region)] = complex(info.k_minus)
        if info.plus_region >= 0:
            region_to_k[int(info.plus_region)] = complex(info.k_plus)

    nelems = len(mesh.elements)
    region_ops: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for region, k_region in region_to_k.items():
        k_eval = k_region if abs(k_region) > EPS else (EPS + 0.0j)
        minus_mask = np.fromiter((info.minus_region == region for info in infos), dtype=bool, count=nelems)
        plus_mask = np.fromiter((info.plus_region == region for info in infos), dtype=bool, count=nelems)
        region_ops[int(region)] = {
            'minus': _assemble_linear_operator_matrices(
                mesh=mesh,
                k0=k_eval,
                obs_normal_deriv=False,
                obs_order=obs_order,
                src_order=src_order,
                far_ratio=far_ratio,
                source_element_mask=minus_mask,
            ),
            'plus': _assemble_linear_operator_matrices(
                mesh=mesh,
                k0=k_eval,
                obs_normal_deriv=False,
                obs_order=obs_order,
                src_order=src_order,
                far_ratio=far_ratio,
                source_element_mask=plus_mask,
            ),
        }
    return region_ops


def _build_coupled_matrix_linear(
    mesh: LinearMesh,
    infos: List[PanelCoupledInfo],
    pol: str,
    obs_order: int = 5,
    src_order: int = 5,
) -> np.ndarray:
    """
    Assemble the experimental nodal linear/Galerkin coupled matrix.

    Unknown ordering is [U_trace_nodes, Q_minus_nodes]. This implementation keeps the
    same weak form as the element-by-element version, but assembles the reusable region/
    side operators once on the global nodal space and then combines them algebraically.
    That preserves the singular/near-singular treatment in `_assemble_linear_operator_matrices`
    while removing the expensive scalar pair-block rebuild inside the production path.
    """

    nnodes = len(mesh.nodes)
    a_mat = np.zeros((2 * nnodes, 2 * nnodes), dtype=np.complex128)
    if nnodes == 0:
        return a_mat

    node_infos = _build_linear_coupled_node_infos(mesh, infos)
    mass_mat = _assemble_linear_mass_matrix(mesh)
    region_ops = _build_linear_coupled_region_operators(
        mesh=mesh,
        infos=infos,
        obs_order=obs_order,
        src_order=src_order,
    )

    node_ids = np.arange(nnodes, dtype=int)
    q_plus_beta = np.asarray([ni.q_plus_beta for ni in node_infos], dtype=np.complex128)
    q_plus_gamma = np.asarray([ni.q_plus_gamma for ni in node_infos], dtype=np.complex128)
    active_regions = np.asarray([ni.active_region for ni in node_infos], dtype=int)
    passive_regions = np.asarray([ni.passive_region for ni in node_infos], dtype=int)
    bc_kinds = np.asarray([ni.bc_kind for ni in node_infos], dtype=object)
    robin_impedance = np.asarray([ni.robin_impedance for ni in node_infos], dtype=np.complex128)
    coeff_u_active = np.asarray([ni.coeff_u_active for ni in node_infos], dtype=np.complex128)
    coeff_q_active = np.asarray([ni.coeff_q_active for ni in node_infos], dtype=np.complex128)
    eps_phys = np.asarray([ni.eps_phys for ni in node_infos], dtype=np.complex128)
    mu_phys = np.asarray([ni.mu_phys for ni in node_infos], dtype=np.complex128)
    k_phys = np.asarray([ni.k_phys for ni in node_infos], dtype=np.complex128)

    def _apply_region_rows(rows: np.ndarray, region: int, row_offset: int) -> None:
        if rows.size == 0:
            return
        ops = region_ops.get(int(region))
        if ops is None:
            raise ValueError(f"Missing linear/Galerkin region operators for region {region}.")
        s_minus, k_minus = ops['minus']
        s_plus, k_plus = ops['plus']
        a_mat[np.ix_(row_offset + rows, node_ids)] += (
            0.5 * mass_mat[np.ix_(rows, node_ids)]
            + k_minus[np.ix_(rows, node_ids)]
            - k_plus[np.ix_(rows, node_ids)]
            + s_plus[np.ix_(rows, node_ids)] * q_plus_gamma[None, :]
        )
        a_mat[np.ix_(row_offset + rows, nnodes + node_ids)] += (
            -s_minus[np.ix_(rows, node_ids)]
            + s_plus[np.ix_(rows, node_ids)] * q_plus_beta[None, :]
        )

    for region in sorted(set(int(v) for v in active_regions)):
        rows = node_ids[active_regions == int(region)]
        _apply_region_rows(rows, int(region), row_offset=0)

    transmission_nodes = node_ids[bc_kinds == 'transmission']
    if transmission_nodes.size > 0:
        transmission_passive = passive_regions[transmission_nodes]
        for region in sorted(set(int(v) for v in transmission_passive if int(v) >= 0)):
            rows = transmission_nodes[transmission_passive == int(region)]
            _apply_region_rows(rows, int(region), row_offset=nnodes)

    bc_nodes = node_ids[bc_kinds != 'transmission']
    if bc_nodes.size > 0:
        zero_z = np.abs(robin_impedance[bc_nodes]) <= EPS
        pec_nodes = bc_nodes[zero_z]
        if pec_nodes.size > 0:
            if pol == 'TM':
                a_mat[np.ix_(nnodes + pec_nodes, node_ids)] += mass_mat[np.ix_(pec_nodes, node_ids)]
            else:
                a_mat[np.ix_(nnodes + pec_nodes, node_ids)] += (
                    mass_mat[np.ix_(pec_nodes, node_ids)] * coeff_u_active[pec_nodes][:, None]
                )
                a_mat[np.ix_(nnodes + pec_nodes, nnodes + node_ids)] += (
                    mass_mat[np.ix_(pec_nodes, node_ids)] * coeff_q_active[pec_nodes][:, None]
                )

        robin_nodes = bc_nodes[~zero_z]
        if robin_nodes.size > 0:
            alpha = np.asarray([
                _surface_robin_alpha(pol, eps_phys[i], mu_phys[i], k_phys[i], robin_impedance[i])
                for i in robin_nodes
            ], dtype=np.complex128)
            a_mat[np.ix_(nnodes + robin_nodes, node_ids)] += (
                mass_mat[np.ix_(robin_nodes, node_ids)] * (coeff_u_active[robin_nodes] + alpha)[:, None]
            )
            a_mat[np.ix_(nnodes + robin_nodes, nnodes + node_ids)] += (
                mass_mat[np.ix_(robin_nodes, node_ids)] * coeff_q_active[robin_nodes][:, None]
            )

    return a_mat

def prepare_linear_galerkin_foundation(
    geometry_snapshot: Dict[str, Any],
    frequency_ghz: float,
    polarization: str,
    geometry_units: str = "inches",
    material_base_dir: str | None = None,
    max_panels: int = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: float | None = None,
    node_snap_tol: float = 1.0e-9,
    obs_order: int = 8,
    src_order: int = 8,
) -> Dict[str, Any]:
    """
    Stage-1 drop-in helper for the upcoming linear-Galerkin coupled formulation.

    It validates geometry, builds the current panelization, upgrades that panelization
    into a continuous two-node linear mesh, derives per-element coupled material info,
    and assembles dense nodal S/K region operators.

    This function is intentionally separate from `solve_monostatic_rcs_2d` so the
    existing production solver remains unchanged until the full nodal coupled matrix
    assembly is ready.
    """

    freq_ghz = float(frequency_ghz)
    if (not math.isfinite(freq_ghz)) or freq_ghz <= 0.0:
        raise ValueError("frequency_ghz must be a positive finite value.")
    pol = _normalize_polarization(polarization)
    unit_scale = _unit_scale_to_meters(geometry_units)
    base_dir = material_base_dir or os.getcwd()
    mesh_freq_ghz = float(mesh_reference_ghz) if mesh_reference_ghz is not None else freq_ghz
    if (not math.isfinite(mesh_freq_ghz)) or mesh_freq_ghz <= 0.0:
        raise ValueError("mesh_reference_ghz must be a positive finite GHz value when provided.")
    lambda_min = C0 / (mesh_freq_ghz * 1e9)
    preflight = validate_geometry_snapshot_for_solver(geometry_snapshot, base_dir=base_dir)
    panels = _build_panels(
        geometry_snapshot=geometry_snapshot,
        meters_scale=unit_scale,
        min_wavelength=lambda_min,
        max_panels=max_panels,
    )
    materials = MaterialLibrary.from_entries(
        geometry_snapshot.get("ibcs", []) or [],
        geometry_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    for _msg in list(preflight.get('warnings', []) or []):
        materials.warn_once(str(_msg))

    mesh = _build_linear_mesh(panels, node_snap_tol=node_snap_tol)
    k0 = 2.0 * math.pi * (freq_ghz * 1e9) / C0
    infos = _build_linear_coupled_infos(mesh, materials, freq_ghz=freq_ghz, pol=pol, k0=k0)

    region_to_k: Dict[int, complex] = {}
    for info in infos:
        if info.minus_region >= 0:
            region_to_k[info.minus_region] = complex(info.k_minus)
        if info.plus_region >= 0:
            region_to_k[info.plus_region] = complex(info.k_plus)

    region_ops: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    cache: Dict[Tuple[float, float, bool], Tuple[np.ndarray, np.ndarray]] = {}
    for region, k_region in region_to_k.items():
        key = (round(float(np.real(k_region)), 12), round(float(np.imag(k_region)), 12), False)
        if key not in cache:
            cache[key] = _assemble_linear_operator_matrices(
                mesh=mesh,
                k0=k_region if abs(k_region) > EPS else (EPS + 0.0j),
                obs_normal_deriv=False,
                obs_order=obs_order,
                src_order=src_order,
            )
        region_ops[region] = cache[key]

    return {
        "panels": panels,
        "mesh": mesh,
        "materials": materials,
        "infos": infos,
        "region_ops": region_ops,
        "metadata": {
            "frequency_ghz": float(freq_ghz),
            "mesh_reference_ghz": float(mesh_freq_ghz),
            "polarization_internal": pol,
            "panel_count": len(panels),
            "linear_element_count": len(mesh.elements),
            "linear_node_count": len(mesh.nodes),
            "node_snap_tol_m": float(node_snap_tol),
            "obs_order": int(obs_order),
            "src_order": int(src_order),
            "warnings": list(materials.warnings),
            "preflight": dict(preflight),
            "status": "stage1-foundation",
        },
    }


def _medium_eta(eps: complex, mu: complex) -> complex:
    eps = _normalize_material_value(eps, 1.0 + 0.0j)
    mu = _normalize_material_value(mu, 1.0 + 0.0j)
    return ETA0 * cmath.sqrt(mu / eps)


def _medium_n(eps: complex, mu: complex) -> complex:
    eps = _normalize_material_value(eps, 1.0 + 0.0j)
    mu = _normalize_material_value(mu, 1.0 + 0.0j)
    return cmath.sqrt(eps * mu)


def _safe_complex_div(num: complex, den: complex, fallback: complex) -> complex:
    if abs(den) <= EPS:
        return fallback
    return num / den


def _snell_cos_t(eps1: complex, mu1: complex, eps2: complex, mu2: complex, cos_i: float) -> complex:
    c_i = max(0.0, min(1.0, float(abs(cos_i))))
    s_i2 = max(0.0, 1.0 - c_i * c_i)
    n1 = _medium_n(eps1, mu1)
    n2 = _medium_n(eps2, mu2)
    if abs(n2) <= EPS:
        n2 = 1.0 + 0.0j
    s_t2 = (n1 / n2) ** 2 * s_i2
    return cmath.sqrt(1.0 - s_t2)


def _projected_impedance(eps: complex, mu: complex, cos_theta: complex, pol: str) -> complex:
    eta = _medium_eta(eps, mu)
    if pol == "TE":
        return _safe_complex_div(eta, cos_theta, eta)
    return eta * cos_theta


def _parallel_impedance(z1: complex, z2: complex) -> complex:
    if abs(z1) <= EPS:
        return z2
    if abs(z2) <= EPS:
        return z1
    return _safe_complex_div(z1 * z2, z1 + z2, z1)


def _needs_coupled_formulation(panels: List[Panel]) -> bool:
    """Use the coupled trace formulation for all production solves."""

    _ = panels
    return True


def _region_medium(materials: MaterialLibrary, region_flag: int, freq_ghz: float) -> Tuple[complex, complex]:
    if region_flag <= 0:
        return 1.0 + 0.0j, 1.0 + 0.0j
    return materials.get_medium(region_flag, freq_ghz)


def _causal_medium_index(eps: complex, mu: complex) -> complex:
    """
    Choose refractive-index branch consistent with passive media in e^{-jwt}.

    Enforces a consistent sign choice so attenuation is physical.
    """

    n = _medium_n(eps, mu)
    if n.real < 0.0:
        n = -n
    # e^{-j omega t} convention prefers Im(n) <= 0 for passive attenuation.
    if n.imag > 0.0:
        n = -n
    if abs(n) <= EPS:
        return 1.0 + 0.0j
    return n


def _medium_wavenumber(
    k0: float,
    eps: complex,
    mu: complex,
) -> complex:
    """Complex medium wavenumber used directly inside integral kernels."""

    return complex(k0) * _causal_medium_index(eps, mu)


def _impedance_to_admittance(z_value: complex) -> complex:
    z_eval = _ensure_finite_complex(z_value, "Surface impedance")
    if abs(z_eval) <= EPS:
        return 0.0 + 0.0j
    return 1.0 / z_eval


def _surface_robin_alpha(
    pol: str,
    eps_medium: complex,
    mu_medium: complex,
    k_medium: complex,
    z_surface: complex,
) -> complex:
    """
    Return the scalar Robin coefficient alpha for q + alpha*u = 0.

    Branch semantics with direct TE/TM labeling:
    - TE: q - j*omega*eps * Zs * u = 0
    - TM: q + j*omega*mu / Zs * u = 0

    Using k*eta = omega*mu and k/eta = omega*eps, this becomes:
    - TE: alpha = -j * k * Zs / eta
    - TM: alpha =  j * k * eta / Zs
    """

    if abs(z_surface) <= EPS:
        return 0.0 + 0.0j
    eta_medium = _medium_eta(eps_medium, mu_medium)
    if pol == "TM":
        return 1j * complex(k_medium) * _safe_complex_div(eta_medium, z_surface, 0.0 + 0.0j)
    return -1j * complex(k_medium) * _safe_complex_div(z_surface, eta_medium, 0.0 + 0.0j)


def _region_side_trace_coefficients(info: PanelCoupledInfo, region_flag: int) -> Tuple[complex, complex]:
    """
    Map a region-side normal derivative to [u_trace, q_minus] coefficients.

    Returns (coeff_u, coeff_q) such that:
        q_region = coeff_u * u_trace + coeff_q * q_minus
    """

    if info.minus_region == region_flag:
        return 0.0 + 0.0j, 1.0 + 0.0j
    if info.plus_region == region_flag:
        return complex(info.q_plus_gamma), complex(info.q_plus_beta)
    raise ValueError("Requested region does not participate in this panel.")


def _q_plus_beta(
    pol: str,
    eps_minus: complex,
    mu_minus: complex,
    eps_plus: complex,
    mu_plus: complex,
) -> complex:
    """
    Scaling between minus-side and plus-side raw normal derivatives across interface.

    Branch semantics with direct TE/TM labeling:
    - TE uses the (1/eps) * du/dn continuity branch.
    - TM uses the (1/mu) * du/dn continuity branch.
    """

    if pol == "TM":
        return _safe_complex_div(mu_plus, mu_minus, 1.0 + 0.0j)
    return _safe_complex_div(eps_plus, eps_minus, 1.0 + 0.0j)


def _panel_effective_impedance(
    panel: Panel,
    materials: MaterialLibrary,
    freq_ghz: float,
    pol: str,
    cos_inc: float,
) -> complex:
    """
    Legacy/localized impedance approximation path used by non-coupled formulation.

    Coupled dielectric mode bypasses this and enforces interface physics globally.
    """

    if panel.seg_type == 1:
        z_card = materials.get_impedance(panel.ibc_flag, freq_ghz)
        return z_card

    if panel.seg_type == 2:
        if panel.ibc_flag > 0:
            return materials.get_impedance(panel.ibc_flag, freq_ghz)
        return 0.0 + 0.0j

    if panel.seg_type == 3:
        eps2, mu2 = materials.get_medium(panel.ipn1, freq_ghz)
        cos_t = _snell_cos_t(1.0 + 0.0j, 1.0 + 0.0j, eps2, mu2, cos_inc)
        z_int = _projected_impedance(eps2, mu2, cos_t, pol)
        if panel.ibc_flag > 0:
            z_card = materials.get_impedance(panel.ibc_flag, freq_ghz)
            return z_int + z_card
        return z_int

    if panel.seg_type == 4:
        if panel.ibc_flag > 0:
            return materials.get_impedance(panel.ibc_flag, freq_ghz)
        return 0.0 + 0.0j

    if panel.seg_type == 5:
        eps1, mu1 = materials.get_medium(panel.ipn1, freq_ghz)
        eps2, mu2 = materials.get_medium(panel.ipn2, freq_ghz)
        cos_i = complex(max(1e-6, min(1.0, abs(cos_inc))), 0.0)
        cos_t = _snell_cos_t(eps1, mu1, eps2, mu2, float(abs(cos_inc)))
        z1 = _projected_impedance(eps1, mu1, cos_i, pol)
        z2 = _projected_impedance(eps2, mu2, cos_t, pol)
        z_if = _parallel_impedance(z1, z2)
        if panel.ibc_flag > 0:
            z_card = materials.get_impedance(panel.ibc_flag, freq_ghz)
            return z_if + z_card
        return z_if

    if panel.ibc_flag > 0:
        return materials.get_impedance(panel.ibc_flag, freq_ghz)
    return 0.0 + 0.0j


def _build_coupled_panel_info(
    panels: List[Panel],
    materials: MaterialLibrary,
    freq_ghz: float,
    pol: str,
    k0: float,
) -> List[PanelCoupledInfo]:
    """
    Translate geometry TYPE/IBC/IPN flags into coupled interface algebra per panel.

    Project convention:
    - the drawn panel normal points toward the IPN1 side,
    - TYPE 3: plus/IPN1 = air, minus = dielectric(IPN1),
    - TYPE 5: plus/IPN1, minus/IPN2,
    - TYPE 4: plus/IPN1 = dielectric, minus = PEC/IBC side.

    The coupled assembly is allowed to use whichever side is the valid non-PEC side,
    so TYPE 4 remains solvable even though the PEC side is the minus side.
    """

    infos: List[PanelCoupledInfo] = []
    sheet_region_by_name: Dict[str, int] = {}
    next_sheet_region = 900_000

    for panel in panels:
        seg_type = panel.seg_type
        if seg_type == 3:
            if panel.ipn1 <= 0:
                raise ValueError(f"TYPE 3 panel '{panel.name}' requires IPN1 > 0.")
            plus_region = 0
            minus_region = panel.ipn1
            bc_kind = "transmission"
            plus_has_incident = True
            minus_has_incident = False
        elif seg_type == 5:
            if panel.ipn1 <= 0 or panel.ipn2 <= 0:
                raise ValueError(f"TYPE 5 panel '{panel.name}' requires IPN1 > 0 and IPN2 > 0.")
            plus_region = panel.ipn1
            minus_region = panel.ipn2
            bc_kind = "transmission"
            plus_has_incident = False
            minus_has_incident = False
        elif seg_type == 4:
            if panel.ipn1 <= 0:
                raise ValueError(f"TYPE 4 panel '{panel.name}' requires IPN1 > 0.")
            plus_region = panel.ipn1
            minus_region = -1
            bc_kind = "robin"
            plus_has_incident = False
            minus_has_incident = False
        elif seg_type == 2:
            minus_region = 0
            plus_region = -1
            bc_kind = "robin"
            minus_has_incident = True
            plus_has_incident = False
        elif seg_type == 1:
            if panel.ibc_flag <= 0:
                raise ValueError(
                    f"TYPE 1 panel '{panel.name}' requires IBC > 0 in coupled dielectric mode."
                )
            sheet_name = panel.name.strip() or "__type1_sheet__"
            sheet_region = sheet_region_by_name.get(sheet_name)
            if sheet_region is None:
                sheet_region = next_sheet_region
                sheet_region_by_name[sheet_name] = sheet_region
                next_sheet_region += 1
            minus_region = 0
            plus_region = sheet_region
            bc_kind = "transmission"
            minus_has_incident = True
            plus_has_incident = False
        else:
            minus_region = 0
            plus_region = -1
            bc_kind = "robin"
            minus_has_incident = True
            plus_has_incident = False

        eps_minus, mu_minus = _region_medium(materials, minus_region, freq_ghz)
        eps_plus, mu_plus = _region_medium(materials, plus_region, freq_ghz)
        k_minus = _medium_wavenumber(k0, eps_minus, mu_minus)
        k_plus = _medium_wavenumber(k0, eps_plus, mu_plus)
        if (
            abs(k_minus.imag) > 1e-10 or abs(k_plus.imag) > 1e-10
        ) and _complex_hankel_backend_name() == "native-series-asymptotic":
            raise RuntimeError(
                "Lossy dielectric media require SciPy or mpmath for trustworthy complex-Hankel evaluation. "
                "Install one of those backends before running production dielectric solves."
            )

        z_card = materials.get_impedance(panel.ibc_flag, freq_ghz) if panel.ibc_flag > 0 else 0.0 + 0.0j
        if bc_kind == "transmission":
            if seg_type == 1:
                if abs(z_card) <= EPS:
                    raise ValueError(
                        f"TYPE 1 panel '{panel.name}' has zero impedance; provide non-zero IBC for sheet mode."
                    )
                q_plus_beta = -1.0 + 0.0j
                q_plus_gamma = _impedance_to_admittance(z_card)
            else:
                q_plus_beta = _q_plus_beta(pol, eps_minus, mu_minus, eps_plus, mu_plus)
                q_plus_gamma = _impedance_to_admittance(z_card)
        else:
            q_plus_beta = _q_plus_beta(pol, eps_minus, mu_minus, eps_plus, mu_plus)
            q_plus_gamma = 0.0 + 0.0j

        infos.append(
            PanelCoupledInfo(
                seg_type=seg_type,
                plus_region=plus_region,
                minus_region=minus_region,
                plus_has_incident=plus_has_incident,
                minus_has_incident=minus_has_incident,
                eps_plus=eps_plus,
                mu_plus=mu_plus,
                eps_minus=eps_minus,
                mu_minus=mu_minus,
                k_plus=k_plus,
                k_minus=k_minus,
                q_plus_beta=q_plus_beta,
                q_plus_gamma=q_plus_gamma,
                bc_kind=bc_kind,
                robin_impedance=z_card if bc_kind == "robin" else 0.0 + 0.0j,
            )
        )

    return infos


def _green_2d(k0: complex | float, r: float) -> complex:
    """2D scalar Green's function G = j/4 * H0^(2)(k r)."""

    x = complex(k0) * max(r, EPS)
    if abs(x) <= 1e-12:
        x = 1e-12 + 0.0j
    return 0.25j * _hankel2_0(x)


def _dgreen_dn_obs(k0: complex | float, r_vec: np.ndarray, n_obs: np.ndarray) -> complex:
    """Normal derivative of Green's function w.r.t. observation point normal."""

    r = float(np.linalg.norm(r_vec))
    if r <= EPS:
        return 0.0 + 0.0j
    x = complex(k0) * r
    if abs(x) <= 1e-12:
        x = 1e-12 + 0.0j
    h1 = _hankel2_1(x)
    projection = float(np.dot(n_obs, r_vec) / r)
    return (-0.25j * complex(k0)) * h1 * projection


def _dgreen_dn_src(k0: complex | float, r_vec: np.ndarray, n_src: np.ndarray) -> complex:
    """Normal derivative of Green's function w.r.t. source panel normal."""

    r = float(np.linalg.norm(r_vec))
    if r <= EPS:
        return 0.0 + 0.0j
    x = complex(k0) * r
    if abs(x) <= 1e-12:
        x = 1e-12 + 0.0j
    h1 = _hankel2_1(x)
    projection = float(np.dot(n_src, r_vec) / r)
    return (0.25j * complex(k0)) * h1 * projection


def _quadrature_nodes(order: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    qx, qw = np.polynomial.legendre.leggauss(order)
    t = 0.5 * (qx + 1.0)
    w = 0.5 * qw
    return t, w


_QUAD_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _get_quadrature(order: int) -> Tuple[np.ndarray, np.ndarray]:
    o = int(order)
    if o not in _QUAD_CACHE:
        _QUAD_CACHE[o] = _quadrature_nodes(o)
    return _QUAD_CACHE[o]


def _near_singular_scheme(distance: float, panel_length: float) -> Tuple[int, int]:
    """
    Choose quadrature order and source-panel subdivision count.

    This improves near-singular accuracy when observation points approach a panel.
    """

    ratio = float(distance) / max(float(panel_length), EPS)
    if ratio < 0.25:
        return 64, 16
    if ratio < 0.60:
        return 56, 10
    if ratio < 1.50:
        return 40, 6
    if ratio < 3.00:
        return 28, 3
    return 16, 1


def _integrate_panel_generic(
    obs: np.ndarray,
    src: Panel,
    kernel_eval: Callable[[np.ndarray, np.ndarray], complex],
) -> complex:
    seg = src.p1 - src.p0
    distance = float(np.linalg.norm(obs - src.center))
    order, splits = _near_singular_scheme(distance, src.length)
    qt, qw = _get_quadrature(order)

    acc = 0.0 + 0.0j
    inv_splits = 1.0 / float(splits)
    for sidx in range(splits):
        t0 = float(sidx) * inv_splits
        dt = inv_splits
        for t, w in zip(qt, qw):
            u = t0 + dt * float(t)
            rp = src.p0 + u * seg
            acc += (dt * float(w)) * kernel_eval(obs, rp)
    return acc * src.length


def _single_layer_self_term(k0: complex | float, panel_length: float) -> complex:
    """
    Self-term for pulse-basis diagonal using singularity subtraction + correction.

    Base asymptotic piece is analytic; remainder is integrated numerically so this
    remains accurate beyond the small-argument regime.
    """

    l = max(float(panel_length), EPS)
    kz = complex(k0)
    x = kz * l / 4.0
    if abs(x) <= 1e-12:
        x = 1e-12 + 0.0j
    asym = (l / (2.0 * math.pi)) * (cmath.log(x) + EULER_GAMMA - 1.0) + 0.25j * l

    # Correction integral for finite kL effects:
    # 2 * ∫_0^{L/2} [G(r) - G_asym(r)] dr
    a = 0.5 * l
    kl = abs(kz) * l
    if kl < 0.5:
        order, splits = 24, 6
    elif kl < 3.0:
        order, splits = 36, 8
    elif kl < 10.0:
        order, splits = 48, 12
    else:
        order, splits = 64, 16

    qt, qw = _get_quadrature(order)
    corr_pos = 0.0 + 0.0j
    inv_splits = 1.0 / float(splits)
    for sidx in range(splits):
        r0 = a * float(sidx) * inv_splits
        dr = a * inv_splits
        for t, w in zip(qt, qw):
            r = r0 + dr * float(t)
            g = _green_2d(k0, r)
            z = kz * max(r, EPS) / 2.0
            if abs(z) <= 1e-12:
                z = 1e-12 + 0.0j
            g_asym = (1.0 / (2.0 * math.pi)) * (cmath.log(z) + EULER_GAMMA) + 0.25j
            corr_pos += dr * float(w) * (g - g_asym)

    return asym + 2.0 * corr_pos


def _integrate_single_layer(obs: np.ndarray, src: Panel, k0: complex | float, is_self: bool) -> complex:
    """Numerically integrate single-layer operator over one source panel."""

    if is_self:
        return _single_layer_self_term(k0, src.length)
    return _integrate_panel_generic(
        obs,
        src,
        lambda o, rp: _green_2d(k0, float(np.linalg.norm(o - rp))),
    )


def _integrate_kprime(obs: np.ndarray, n_obs: np.ndarray, src: Panel, k0: complex | float, is_self: bool) -> complex:
    """Numerically integrate observation-normal derivative operator (K')."""

    if is_self:
        return 0.0 + 0.0j
    return _integrate_panel_generic(
        obs,
        src,
        lambda o, rp: _dgreen_dn_obs(k0, o - rp, n_obs),
    )


def _integrate_k_source(obs: np.ndarray, src: Panel, k0: complex | float, is_self: bool) -> complex:
    """Numerically integrate source-normal derivative operator (K)."""

    if is_self:
        return 0.0 + 0.0j
    return _integrate_panel_generic(
        obs,
        src,
        lambda o, rp: _dgreen_dn_src(k0, o - rp, src.normal),
    )


def _observation_samples(panel: Panel, order: int = 2) -> List[Tuple[np.ndarray, float]]:
    """
    Observation-point averaging along each panel.

    This is still pulse-basis, but reduces pure midpoint-collocation bias.
    """

    o = max(1, int(order))
    if o == 1:
        return [(panel.center, 1.0)]
    qt, qw = _get_quadrature(o)
    seg = panel.p1 - panel.p0
    out: List[Tuple[np.ndarray, float]] = []
    for t, w in zip(qt, qw):
        out.append((panel.p0 + float(t) * seg, float(w)))
    return out


def _build_bem_matrices(
    panels: List[Panel],
    k0: complex,
    obs_normal_deriv: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build S and K/K' dense BEM operator matrices.

    obs_normal_deriv=True  → K' uses observation-panel normal (legacy EFIE/MFIE).
    obs_normal_deriv=False → K  uses source-panel normal (coupled dielectric).

    Far-field pairs (center-distance / panel-length >= 3): fully vectorised via
    scipy.special.hankel2 accepting array inputs.  This converts O(N^2 * Q)
    Python scalar calls into a single batched C/BLAS evaluation.

    Near-field pairs (the minority, O(N)): existing adaptive Gauss quadrature
    with near-singular correction is preserved exactly.
    """
    n = len(panels)
    s_mat = np.zeros((n, n), dtype=np.complex128)
    k_mat = np.zeros((n, n), dtype=np.complex128)
    if n == 0:
        return s_mat, k_mat

    centers = np.empty((n, 2), dtype=float)
    normals = np.empty((n, 2), dtype=float)
    p0s = np.empty((n, 2), dtype=float)
    segs = np.empty((n, 2), dtype=float)
    lengths = np.empty(n, dtype=float)
    for i, p in enumerate(panels):
        centers[i] = p.center
        normals[i] = p.normal
        p0s[i] = p.p0
        segs[i] = p.p1 - p.p0
        lengths[i] = p.length

    # Diagonal self-terms
    for m in range(n):
        s_mat[m, m] = _single_layer_self_term(k0, lengths[m])
    # k_mat diagonal remains zero (self-term of K/K' is 0)

    diff_cc = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N,N,2)
    dist_cc = np.linalg.norm(diff_cc, axis=-1)                       # (N,N)
    ratio_mat = dist_cc / np.maximum(lengths[np.newaxis, :], EPS)    # (N,N)

    idx = np.arange(n)
    off_diag = idx[:, None] != idx[None, :]
    FAR_RATIO = 3.0
    Q_FAR = 16
    far_mask = off_diag & (ratio_mat >= FAR_RATIO)
    near_mask = off_diag & (ratio_mat < FAR_RATIO)

    # --- Vectorised far-field path ---
    if np.any(far_mask) and _SCIPY_SPECIAL is not None:
        qt_far, qw_far = _get_quadrature(Q_FAR)
        # Source quadrature points: rq[src, q] = p0[src] + t_q * seg[src]
        rq = p0s[:, np.newaxis, :] + qt_far[np.newaxis, :, np.newaxis] * segs[:, np.newaxis, :]  # (N,Q,2)

        # Adaptive batch size to keep peak memory ≤ ~200 MB
        bytes_per_entry = 16  # complex128
        BATCH = max(8, min(128, int(2e8 / (n * Q_FAR * 2 * bytes_per_entry))))

        for m0 in range(0, n, BATCH):
            m1 = min(m0 + BATCH, n)
            B = m1 - m0
            far_b = far_mask[m0:m1, :]           # (B,N)
            if not np.any(far_b):
                continue

            obs_c = centers[m0:m1, :]            # (B,2)
            obs_n = normals[m0:m1, :]            # (B,2)

            # diff[b,n,q] = obs_c[b] - rq[n,q], pointing from source quad-pt to obs
            diff = obs_c[:, np.newaxis, np.newaxis, :] - rq[np.newaxis, :, :, :]  # (B,N,Q,2)
            dist = np.linalg.norm(diff, axis=-1)  # (B,N,Q)
            dist = np.maximum(dist, EPS)

            kr_abs = float(abs(complex(k0)))
            _k0_is_real = abs(complex(k0).imag) < 1e-10 * max(kr_abs, 1e-30)
            if _k0_is_real:
                # Fast path: real argument – j0/y0/j1/y1 are ~9× faster than hankel2
                kr_real = (float(complex(k0).real) * dist).ravel()
                kr_real = np.maximum(kr_real, 1e-12)
                h0 = (_SCIPY_SPECIAL.j0(kr_real) - 1j * _SCIPY_SPECIAL.y0(kr_real)).reshape(B, n, Q_FAR)
                h1 = (_SCIPY_SPECIAL.j1(kr_real) - 1j * _SCIPY_SPECIAL.y1(kr_real)).reshape(B, n, Q_FAR)
            else:
                # Lossy media: complex k – use robust complex-Hankel evaluation
                kr = (complex(k0) * dist).ravel()
                h0 = _hankel2_0_array(kr).reshape(B, n, Q_FAR)
                h1 = _hankel2_1_array(kr).reshape(B, n, Q_FAR)

            G = 0.25j * h0                                                  # (B,N,Q)
            S_b = lengths * np.einsum('q,bnq->bn', qw_far, G)              # (B,N)

            if obs_normal_deriv:
                # K'[b,n]: n_obs · (obs - src) / |obs - src|
                proj = np.einsum('bi,bnqi->bnq', obs_n, diff) / dist
                Ki = (-0.25j * complex(k0)) * h1 * proj
            else:
                # K[b,n]: n_src · (obs - src) / |obs - src|
                proj = np.sum(normals[np.newaxis, :, np.newaxis, :] * diff, axis=-1) / dist
                Ki = (0.25j * complex(k0)) * h1 * proj

            K_b = lengths * np.einsum('q,bnq->bn', qw_far, Ki)             # (B,N)

            s_mat[m0:m1, :] = np.where(far_b, S_b, s_mat[m0:m1, :])
            k_mat[m0:m1, :] = np.where(far_b, K_b, k_mat[m0:m1, :])

    # --- Near-field path ---
    # When scipy is available: vectorize each (order, splits) tier in one batch call.
    # When scipy is unavailable: fall back to scalar per-pair loop.
    active_mask = off_diag if _SCIPY_SPECIAL is None else near_mask

    if _SCIPY_SPECIAL is not None and np.any(active_mask):
        # Pre-compute 2-pt Gauss observation samples for all panels
        qt2, qw2 = _get_quadrature(2)
        obs_pts_all = p0s[:, np.newaxis, :] + qt2[np.newaxis, :, np.newaxis] * segs[:, np.newaxis, :]  # (N,2,2)

        _k0_is_real_nf = abs(complex(k0).imag) < 1e-10 * max(abs(complex(k0)), 1e-30)

        # Tier boundaries match _near_singular_scheme thresholds
        TIERS: List[Tuple[float, float, int, int]] = [
            (0.0,  0.25, 64, 16),
            (0.25, 0.60, 56, 10),
            (0.60, 1.50, 40,  6),
            (1.50, 3.00, 28,  3),
        ]
        for r_lo, r_hi, t_order, t_splits in TIERS:
            tier_mask = active_mask & (ratio_mat >= r_lo) & (ratio_mat < r_hi)
            if not np.any(tier_mask):
                continue
            tm, tn = np.where(tier_mask)

            # Expanded Gauss nodes/weights for this tier on [0,1]
            dt = 1.0 / t_splits
            qt_t, qw_t = _get_quadrature(t_order)
            t_eff_list: List[float] = []
            w_eff_list: List[float] = []
            for s_idx in range(t_splits):
                t0_s = s_idx * dt
                for tt, ww in zip(qt_t.tolist(), qw_t.tolist()):
                    t_eff_list.append(t0_s + dt * tt)
                    w_eff_list.append(dt * ww)
            t_eff = np.asarray(t_eff_list, dtype=float)   # (Q_eff,)
            w_eff = np.asarray(w_eff_list, dtype=float)   # (Q_eff,)
            Q_eff = len(t_eff)
            w2d = qw2[:, np.newaxis] * w_eff[np.newaxis, :]  # (2, Q_eff) combined weights

            # Source quad pts: (N_tier, Q_eff, 2)
            rq_t = p0s[tn, np.newaxis, :] + t_eff[np.newaxis, :, np.newaxis] * segs[tn, np.newaxis, :]

            # Observation 2-pt samples: (N_tier, 2, 2)
            obs_pts_t = obs_pts_all[tm]   # (N_tier, 2, 2)
            obs_n_t = normals[tm]         # (N_tier, 2)

            # diff[j, o, q, :] = obs_pts_t[j,o] - rq_t[j,q]
            diff_t = obs_pts_t[:, :, np.newaxis, :] - rq_t[:, np.newaxis, :, :]  # (N_tier,2,Q_eff,2)
            dist_t = np.maximum(np.linalg.norm(diff_t, axis=-1), EPS)            # (N_tier,2,Q_eff)

            flat_sz = dist_t.size
            if _k0_is_real_nf:
                kr_f = np.maximum(float(complex(k0).real) * dist_t, 1e-12).ravel()
                h0_f = (_SCIPY_SPECIAL.j0(kr_f) - 1j * _SCIPY_SPECIAL.y0(kr_f)).reshape(dist_t.shape)
                h1_f = (_SCIPY_SPECIAL.j1(kr_f) - 1j * _SCIPY_SPECIAL.y1(kr_f)).reshape(dist_t.shape)
            else:
                kr_f = (complex(k0) * dist_t).ravel()
                h0_f = np.asarray(_SCIPY_SPECIAL.hankel2(0, kr_f)).reshape(dist_t.shape)
                h1_f = np.asarray(_SCIPY_SPECIAL.hankel2(1, kr_f)).reshape(dist_t.shape)

            G_t = 0.25j * h0_f                                         # (N_tier,2,Q_eff)
            S_tier = lengths[tn] * np.einsum('oq,joq->j', w2d, G_t)   # (N_tier,)

            if obs_normal_deriv:
                proj_t = np.einsum('ji,joqi->joq', obs_n_t, diff_t) / dist_t
                Ki_t = (-0.25j * complex(k0)) * h1_f * proj_t
            else:
                src_n_t = normals[tn]                                   # (N_tier, 2)
                proj_t = np.einsum('ji,joqi->joq', src_n_t, diff_t) / dist_t
                Ki_t = (0.25j * complex(k0)) * h1_f * proj_t
            K_tier = lengths[tn] * np.einsum('oq,joq->j', w2d, Ki_t)  # (N_tier,)

            s_mat[tm, tn] = S_tier
            k_mat[tm, tn] = K_tier
    else:
        # Scalar fallback loop (no scipy, or empty near-field set)
        near_rows, near_cols = np.where(active_mask)
        for m, n_idx in zip(near_rows.tolist(), near_cols.tolist()):
            pm = panels[m]
            pn = panels[n_idx]
            obs_samples = _observation_samples(pm, order=2)
            s_val = 0.0 + 0.0j
            k_val = 0.0 + 0.0j
            for obs, w in obs_samples:
                s_val += float(w) * _integrate_single_layer(obs, pn, k0, is_self=False)
                if obs_normal_deriv:
                    k_val += float(w) * _integrate_kprime(obs, pm.normal, pn, k0, is_self=False)
                else:
                    k_val += float(w) * _integrate_k_source(obs, pn, k0, is_self=False)
            s_mat[m, n_idx] = s_val
            k_mat[m, n_idx] = k_val

    return s_mat, k_mat


def _build_operator_matrices(panels: List[Panel], k0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build legacy S and K' dense operators at one frequency."""
    return _build_bem_matrices(panels, complex(k0), obs_normal_deriv=True)


def _build_operator_matrices_coupled(panels: List[Panel], k0: complex | float) -> Tuple[np.ndarray, np.ndarray]:
    """Build coupled-formulation S and K dense operators at one medium k."""
    return _build_bem_matrices(panels, complex(k0), obs_normal_deriv=False)


def _propagation_direction_from_user_angle(elev_deg: float) -> np.ndarray:
    """
    Convert user "coming-from" angle convention to propagation direction.

    Convention:
    - 0 deg: coming from +x (right), propagating toward -x.
    - +90 deg: coming from +y (top), propagating toward -y.
    - -90 deg: coming from -y (bottom), propagating toward +y.
    """

    phi = math.radians(elev_deg)
    return np.asarray([-math.cos(phi), -math.sin(phi)], dtype=float)


def _incident_values(panels: List[Panel], k0: float, elev_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.array([p.center for p in panels], dtype=float)
    normals = np.array([p.normal for p in panels], dtype=float)
    u_mat, du_mat, cos_mat = _incident_values_many(centers, normals, k0, np.asarray([elev_deg], dtype=float))
    return u_mat[:, 0], du_mat[:, 0], cos_mat[:, 0]


def _incident_values_many(
    centers: np.ndarray,
    normals: np.ndarray,
    k0: float,
    elevations_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized incident traces for many elevations.

    Returns arrays of shape (N_panels, N_elevations): (u_inc, du_dn, |n·d|).
    """

    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    phi = np.deg2rad(elev)
    directions = np.stack([-np.cos(phi), -np.sin(phi)], axis=1)  # (E,2)
    phases = centers @ directions.T                               # (N,E)
    u_inc = np.exp((-1j * k0) * phases)
    dot_nd = normals @ directions.T                               # (N,E)
    du_dn = (-1j * k0) * dot_nd * u_inc
    cos_inc = np.abs(dot_nd)
    return (
        np.asarray(u_inc, dtype=np.complex128),
        np.asarray(du_dn, dtype=np.complex128),
        np.asarray(cos_inc, dtype=float),
    )


def _solve_linear_system(a_mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve Ax=b without silently degrading to a least-squares fallback."""

    if a_mat.shape[0] != a_mat.shape[1]:
        raise RuntimeError(
            "Aborting solve: encountered a non-square legacy linear system where a direct square solve was expected. "
            "Refusing to fall back to least-squares."
        )
    try:
        return np.linalg.solve(a_mat, rhs)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise RuntimeError(
            "Aborting solve: direct linear solve failed for the legacy system matrix. "
            "Refusing to fall back to least-squares."
        ) from exc


def _constraint_null_space(c_mat: np.ndarray) -> np.ndarray:
    """Compute a complex null-space basis Z such that C @ Z ~= 0."""

    c_eval = np.asarray(c_mat, dtype=np.complex128)
    if c_eval.ndim != 2:
        raise ValueError("Constraint matrix must be two-dimensional.")
    ncols = int(c_eval.shape[1])
    if ncols <= 0:
        raise ValueError("Constraint matrix must have at least one primal column.")
    if c_eval.shape[0] == 0 or c_eval.size == 0:
        return np.eye(ncols, dtype=np.complex128)

    if _SCIPY_LINALG is not None:
        try:
            _, svals, vh = _SCIPY_LINALG.svd(c_eval, full_matrices=True, check_finite=True)
        except Exception:
            _, svals, vh = np.linalg.svd(c_eval, full_matrices=True)
    else:
        _, svals, vh = np.linalg.svd(c_eval, full_matrices=True)

    svals = np.asarray(svals, dtype=float)
    sigma_max = float(np.max(svals)) if svals.size else 0.0
    tol = max(c_eval.shape) * max(sigma_max, 1.0) * np.finfo(float).eps * 16.0
    rank = int(np.sum(svals > tol))
    z_basis = np.asarray(vh[rank:, :].conj().T, dtype=np.complex128)
    if z_basis.ndim != 2 or z_basis.shape[0] != ncols:
        raise RuntimeError("Internal error: invalid null-space basis shape for exact constrained solve.")
    if z_basis.shape[1] == 0:
        raise RuntimeError(
            "Aborting solve: exact junction constraints eliminate all primal degrees of freedom."
        )
    return z_basis


def _prepare_linear_solver(
    a_mat: np.ndarray,
    constraint_mat: np.ndarray | None = None,
) -> PreparedLinearSolver:
    """
    Prepare reusable factorization for repeated solves with identical matrix.

    Unconstrained systems remain on the direct square-solve path. When
    `constraint_mat` is provided, the solver computes an exact null-space basis
    and solves the reduced least-squares problem over the constrained subspace,
    so junction constraints are enforced exactly rather than by weighted rows.
    """

    a_eval = np.asarray(a_mat, dtype=np.complex128)
    c_eval = None if constraint_mat is None else np.asarray(constraint_mat, dtype=np.complex128)
    if c_eval is not None and c_eval.size > 0:
        if c_eval.ndim != 2:
            raise ValueError("Constraint matrix must be two-dimensional.")
        if c_eval.shape[1] != a_eval.shape[1]:
            raise ValueError("Constraint matrix width does not match the primal system size.")
        z_basis = _constraint_null_space(c_eval)
        reduced_mat = np.asarray(a_eval @ z_basis, dtype=np.complex128)
        return PreparedLinearSolver(
            a_mat=a_eval,
            method="constrained_null_lstsq",
            null_basis=z_basis,
            reduced_mat=reduced_mat,
            constraint_mat=c_eval,
        )

    is_square = a_eval.shape[0] == a_eval.shape[1]
    if not is_square:
        raise RuntimeError(
            "Aborting solve: reusable prepared solver requires a square primal system. "
            "Use exact constraints through constraint_mat instead of a rectangular augmented matrix."
        )

    if _SCIPY_LINALG is not None:
        try:
            lu, piv = _SCIPY_LINALG.lu_factor(a_eval)
            return PreparedLinearSolver(a_mat=a_eval, method="scipy_lu", lu=lu, piv=piv)
        except Exception:
            pass

    return PreparedLinearSolver(a_mat=a_eval, method="numpy_solve")


def _solve_with_prepared_solver(prepared: PreparedLinearSolver, rhs: np.ndarray) -> np.ndarray:
    """Solve with a prepared linear-solver handle."""

    rhs_eval = np.asarray(rhs, dtype=np.complex128)
    if prepared.method == "scipy_lu" and _SCIPY_LINALG is not None and prepared.lu is not None and prepared.piv is not None:
        return _SCIPY_LINALG.lu_solve((prepared.lu, prepared.piv), rhs_eval)
    if prepared.method == "numpy_solve":
        return np.linalg.solve(prepared.a_mat, rhs_eval)
    if prepared.method == "constrained_null_lstsq":
        if prepared.null_basis is None or prepared.reduced_mat is None:
            raise RuntimeError("Aborting solve: constrained solver is missing its reduced-space data.")
        reduced_sol, *_ = np.linalg.lstsq(prepared.reduced_mat, rhs_eval, rcond=None)
        return np.asarray(prepared.null_basis @ reduced_sol, dtype=np.complex128)
    raise RuntimeError(
        f"Aborting solve: unsupported prepared solver method '{prepared.method}'."
    )


def _solve_many_with_prepared_solver(prepared: PreparedLinearSolver, rhs_list: List[np.ndarray]) -> List[np.ndarray]:
    """Solve A x_k = b_k for many right-hand-sides using one prepared handle."""

    if not rhs_list:
        return []
    rhs_mat = np.column_stack(rhs_list)
    sol_mat = _solve_with_prepared_solver(prepared, rhs_mat)
    if sol_mat.ndim == 1:
        sol_mat = sol_mat.reshape(-1, 1)
    return [np.asarray(sol_mat[:, i], dtype=np.complex128) for i in range(sol_mat.shape[1])]


def _residual_norm(a_mat: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    if denom <= EPS:
        denom = 1.0
    return float(np.linalg.norm(a_mat @ x - b) / denom)


def _residual_norm_many(a_mat: np.ndarray, x_mat: np.ndarray, b_mat: np.ndarray) -> np.ndarray:
    """Vectorized residual norms for matrix right-hand-sides."""

    x_eval = np.asarray(x_mat)
    b_eval = np.asarray(b_mat)
    if x_eval.ndim == 1:
        return np.asarray([_residual_norm(a_mat, x_eval, b_eval)], dtype=float)

    residual = a_mat @ x_eval - b_eval
    num = np.linalg.norm(residual, axis=0)
    den = np.linalg.norm(b_eval, axis=0)
    den = np.where(den <= EPS, 1.0, den)
    return np.asarray(num / den, dtype=float)


def _constraint_residual_norm_many(c_mat: np.ndarray | None, x_mat: np.ndarray) -> np.ndarray:
    """Absolute residual norms for exact linear constraints C x = 0."""

    if c_mat is None:
        x_eval = np.asarray(x_mat)
        cols = 1 if x_eval.ndim == 1 else int(x_eval.shape[1])
        return np.zeros(cols, dtype=float)
    c_eval = np.asarray(c_mat, dtype=np.complex128)
    if c_eval.size == 0:
        x_eval = np.asarray(x_mat)
        cols = 1 if x_eval.ndim == 1 else int(x_eval.shape[1])
        return np.zeros(cols, dtype=float)

    x_eval = np.asarray(x_mat, dtype=np.complex128)
    if x_eval.ndim == 1:
        return np.asarray([float(np.linalg.norm(c_eval @ x_eval))], dtype=float)
    return np.asarray(np.linalg.norm(c_eval @ x_eval, axis=0), dtype=float)


def _cond_estimate(a_mat: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(a_mat))
    except np.linalg.LinAlgError:
        return float("inf")


def _adaptive_cfie_eps(k0: float, panel_lengths: np.ndarray) -> float:
    """
    Choose CFIE blending strength from electrical panel size.

    This keeps stabilization weak on electrically small meshes and increases it
    when electrical size grows, reducing resonant conditioning issues.
    """

    if panel_lengths.size == 0:
        return CFIE_EPS
    l_mean = float(np.mean(panel_lengths))
    kh = abs(float(k0)) * max(l_mean, EPS)
    if kh < 0.20:
        eps = 2.0e-4
    elif kh < 0.80:
        eps = 5.0e-4
    elif kh < 2.50:
        eps = 1.0e-3
    elif kh < 6.00:
        eps = 2.0e-3
    else:
        eps = 5.0e-3
    return float(min(1.0e-2, max(1.0e-4, eps)))


def _normalize_rcs_normalization_mode(mode: str | None) -> str:
    """Accept only physical sigma_2d normalization aliases."""

    text = str(mode or "").strip().lower().replace("-", "_")
    if text in {"", "physical", "divide_by_k", "with_k", "k", "derived", "width", "sigma_2d"}:
        return RCS_NORM_MODE_PHYSICAL
    raise ValueError(
        f"Unsupported rcs_normalization_mode '{mode}'. This solver now supports only physical normalization "
        "sigma_2d = |A|^2 / (4k)."
    )


def _rcs_sigma_from_amp(
    amp_vec: np.ndarray,
    k_value: float,
) -> np.ndarray:
    """Apply physical 2D scattering-width normalization to the far-field amplitude."""

    amp_eval = np.asarray(amp_vec, dtype=np.complex128)
    scale = float(RCS_NORM_NUMERATOR) / max(float(k_value), EPS)
    sigma_lin = scale * (np.abs(amp_eval) ** 2)
    sigma_lin = np.where(np.isfinite(sigma_lin) & (sigma_lin >= EPS), sigma_lin, EPS)
    return np.asarray(sigma_lin, dtype=float)


def _resolve_worker_count(enabled: bool, requested: int, jobs: int) -> int:
    """
    Resolve thread-pool worker count for per-elevation parallel execution.

    Returns 1 when parallel execution is disabled or not useful.
    """

    count = int(max(0, jobs))
    if not enabled or count <= 1:
        return 1
    if int(requested) > 0:
        return max(1, min(int(requested), count))
    cpu = int(os.cpu_count() or 1)
    return max(1, min(cpu, count))


def evaluate_quality_gate(
    metadata: Dict[str, Any],
    thresholds: Dict[str, float | int] | None = None,
) -> Dict[str, Any]:
    """
    Evaluate a lightweight numeric quality gate from solver metadata.

    This does not prove correctness; it catches obvious numerical-risk runs.
    """

    defaults: Dict[str, float | int] = {
        "residual_norm_max": 1.0e-2,
        "constraint_residual_norm_max": 1.0e-8,
        "condition_est_max": 1.0e6,
        "warnings_max": 10,
    }
    merged = dict(defaults)
    if thresholds:
        merged.update(dict(thresholds))

    residual_limit = float(merged.get("residual_norm_max", defaults["residual_norm_max"]))
    constraint_limit = float(merged.get("constraint_residual_norm_max", defaults["constraint_residual_norm_max"]))
    condition_limit = float(merged.get("condition_est_max", defaults["condition_est_max"]))
    warnings_limit = int(merged.get("warnings_max", defaults["warnings_max"]))

    residual_value = float(metadata.get("residual_norm_max", 0.0) or 0.0)
    constraint_value = float(metadata.get("constraint_residual_norm_max", 0.0) or 0.0)
    condition_value = float(metadata.get("condition_est_max", 0.0) or 0.0)
    condition_computed = bool(metadata.get("condition_est_computed", True))
    warnings_count = len(list(metadata.get("warnings", []) or []))

    violations: List[str] = []
    if not math.isfinite(residual_value) or residual_value > residual_limit:
        violations.append(
            f"residual_norm_max={residual_value:.6g} exceeds limit {residual_limit:.6g}"
        )
    if int(metadata.get("junction_constraints", 0) or 0) > 0 and (
        (not math.isfinite(constraint_value)) or constraint_value > constraint_limit
    ):
        violations.append(
            f"constraint_residual_norm_max={constraint_value:.6g} exceeds limit {constraint_limit:.6g}"
        )
    if condition_computed and (not math.isfinite(condition_value) or condition_value > condition_limit):
        violations.append(
            f"condition_est_max={condition_value:.6g} exceeds limit {condition_limit:.6g}"
        )
    if warnings_count > warnings_limit:
        violations.append(
            f"warnings_count={warnings_count} exceeds limit {warnings_limit}"
        )

    return {
        "passed": len(violations) == 0,
        "thresholds": {
            "residual_norm_max": residual_limit,
            "constraint_residual_norm_max": constraint_limit,
            "condition_est_max": condition_limit,
            "warnings_max": warnings_limit,
        },
        "values": {
            "residual_norm_max": residual_value,
            "constraint_residual_norm_max": constraint_value,
            "condition_est_max": condition_value,
            "condition_est_computed": condition_computed,
            "warnings_count": warnings_count,
        },
        "violations": violations,
    }


def _build_system_matrix(
    panels: List[Panel],
    s_mat: np.ndarray,
    kp_mat: np.ndarray,
    z_eff: np.ndarray,
    pol: str,
    k0: float,
    cfie_eps: float,
    seg_types: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble legacy single-equation EFIE/MFIE-like system matrix A."""

    n = len(panels)
    if seg_types is None:
        seg_types = np.asarray([p.seg_type for p in panels], dtype=int)
    else:
        seg_types = np.asarray(seg_types, dtype=int)
    if seg_types.shape[0] != n:
        raise ValueError("seg_types length must match panel count.")

    z_eff = np.asarray(z_eff, dtype=np.complex128)
    if z_eff.shape[0] != n:
        raise ValueError("z_eff length must match panel count.")

    a_mat = np.zeros((n, n), dtype=np.complex128)
    tm_like = np.ones(n, dtype=bool) if pol == "TM" else (seg_types == 1)
    other = ~tm_like

    if np.any(tm_like):
        a_mat[tm_like, :] = s_mat[tm_like, :]
        tm_idx = np.flatnonzero(tm_like)
        a_mat[tm_idx, tm_idx] -= z_eff[tm_like]

    if np.any(other):
        z_other = z_eff[other]
        y_other = np.zeros_like(z_other, dtype=np.complex128)
        nz = np.abs(z_other) > 1e-9
        y_other[nz] = 1.0 / z_other[nz]

        blend = (1j * k0 * y_other + cfie_eps).reshape(-1, 1)
        a_mat[other, :] = kp_mat[other, :] + blend * s_mat[other, :]
        other_idx = np.flatnonzero(other)
        a_mat[other_idx, other_idx] -= 0.5 + cfie_eps * z_other

    return a_mat


def _build_system_rhs(
    panels: List[Panel],
    u_inc: np.ndarray,
    du_dn: np.ndarray,
    z_eff: np.ndarray,
    pol: str,
    k0: float,
    cfie_eps: float,
    seg_types: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble RHS vector for the legacy single-equation system."""

    n = len(panels)
    if seg_types is None:
        seg_types = np.asarray([p.seg_type for p in panels], dtype=int)
    else:
        seg_types = np.asarray(seg_types, dtype=int)
    if seg_types.shape[0] != n:
        raise ValueError("seg_types length must match panel count.")

    z_eff = np.asarray(z_eff, dtype=np.complex128)
    if z_eff.shape[0] != n:
        raise ValueError("z_eff length must match panel count.")

    rhs = np.zeros(n, dtype=np.complex128)
    tm_like = np.ones(n, dtype=bool) if pol == "TM" else (seg_types == 1)
    other = ~tm_like

    if np.any(tm_like):
        rhs[tm_like] = -u_inc[tm_like]

    if np.any(other):
        z_other = z_eff[other]
        y_other = np.zeros_like(z_other, dtype=np.complex128)
        nz = np.abs(z_other) > 1e-9
        y_other[nz] = 1.0 / z_other[nz]
        rhs[other] = -du_dn[other] - (1j * k0 * y_other + cfie_eps) * u_inc[other]

    return rhs


def _build_system_rhs_many(
    seg_types: np.ndarray,
    u_inc_mat: np.ndarray,
    du_dn_mat: np.ndarray,
    z_eff: np.ndarray,
    pol: str,
    k0: float,
    cfie_eps: float,
) -> np.ndarray:
    """
    Vectorized RHS assembly for many elevations with shared z_eff.

    Inputs/outputs use shape (N_panels, N_elevations).
    """

    seg_eval = np.asarray(seg_types, dtype=int).reshape(-1)
    u_eval = np.asarray(u_inc_mat, dtype=np.complex128)
    du_eval = np.asarray(du_dn_mat, dtype=np.complex128)
    z_eval = np.asarray(z_eff, dtype=np.complex128).reshape(-1)

    if u_eval.shape != du_eval.shape:
        raise ValueError("u_inc_mat and du_dn_mat must have the same shape.")
    if u_eval.shape[0] != seg_eval.shape[0] or u_eval.shape[0] != z_eval.shape[0]:
        raise ValueError("segment, field, and impedance dimensions do not match.")

    rhs = np.zeros_like(u_eval, dtype=np.complex128)
    tm_like = np.ones(seg_eval.shape[0], dtype=bool) if pol == "TM" else (seg_eval == 1)
    other = ~tm_like

    if np.any(tm_like):
        rhs[tm_like, :] = -u_eval[tm_like, :]

    if np.any(other):
        z_other = z_eval[other]
        y_other = np.zeros_like(z_other, dtype=np.complex128)
        nz = np.abs(z_other) > 1e-9
        y_other[nz] = 1.0 / z_other[nz]
        rhs[other, :] = -du_eval[other, :] - (1j * k0 * y_other[:, None] + cfie_eps) * u_eval[other, :]

    return rhs


def _build_system(
    panels: List[Panel],
    s_mat: np.ndarray,
    kp_mat: np.ndarray,
    u_inc: np.ndarray,
    du_dn: np.ndarray,
    z_eff: np.ndarray,
    pol: str,
    k0: float,
    cfie_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble legacy single-equation EFIE/MFIE-like linear system."""

    a_mat = _build_system_matrix(
        panels=panels,
        s_mat=s_mat,
        kp_mat=kp_mat,
        z_eff=z_eff,
        pol=pol,
        k0=k0,
        cfie_eps=cfie_eps,
    )
    rhs = _build_system_rhs(
        panels=panels,
        u_inc=u_inc,
        du_dn=du_dn,
        z_eff=z_eff,
        pol=pol,
        k0=k0,
        cfie_eps=cfie_eps,
    )

    return a_mat, rhs


def _backscatter_rcs(
    panels: List[Panel],
    sigma: np.ndarray,
    k0: float,
    elev_deg: float,
) -> Tuple[float, complex]:
    """Convert solved legacy current to monostatic 2D RCS and complex amplitude."""
    centers = np.array([p.center for p in panels], dtype=float)
    lengths = np.array([p.length for p in panels], dtype=float)
    sigma_lin_vec, amp_vec = _backscatter_rcs_many(
        centers=centers,
        lengths=lengths,
        sigma_mat=np.asarray(sigma, dtype=np.complex128).reshape(-1, 1),
        k0=k0,
        elevations_deg=np.asarray([elev_deg], dtype=float),
    )
    return float(sigma_lin_vec[0]), complex(amp_vec[0])


def _backscatter_rcs_many(
    centers: np.ndarray,
    lengths: np.ndarray,
    sigma_mat: np.ndarray,
    k0: float,
    elevations_deg: np.ndarray,
    normals: np.ndarray | None = None,
    seg_types: np.ndarray | None = None,
    pol: str | None = None,
    z_eff: np.ndarray | None = None,
    cfie_eps: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized legacy backscatter computation for many elevations.

    Returns (sigma_linear[E], amplitude[E]).

    Branch note:
    - TM / TYPE 1 branches use a sheet/current-like far-field sum.
    - TE on PEC/Robin-style legacy branches solves a combined-field density, so the
      far-field must include the double-layer directional factor (n·s) and the
      CFIE single-layer blend. Using the TM-style raw density sum here tends to
      wash out TE null structure and make it correlate too closely to TM.
    """

    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    phi = np.deg2rad(elev)
    scatter_dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)      # (E,2)
    phase_mat = np.exp((1j * k0) * (centers @ scatter_dirs.T))       # (N,E)

    sigma_eval = np.asarray(sigma_mat, dtype=np.complex128)
    if sigma_eval.ndim == 1:
        sigma_eval = sigma_eval.reshape(-1, 1)
    if sigma_eval.shape[0] != centers.shape[0]:
        raise ValueError("sigma_mat row count must match panel-center count.")
    if sigma_eval.shape[1] != phase_mat.shape[1]:
        raise ValueError("sigma_mat column count must match elevation count.")
    weighted = sigma_eval * np.asarray(lengths, dtype=float)[:, None]

    amp_terms = weighted * phase_mat

    if pol == "TE" and normals is not None and seg_types is not None:
        normal_eval = np.asarray(normals, dtype=float)
        seg_eval = np.asarray(seg_types, dtype=int).reshape(-1)
        if normal_eval.shape != centers.shape:
            raise ValueError("normals must have the same shape as centers.")
        if seg_eval.shape[0] != centers.shape[0]:
            raise ValueError("seg_types length must match panel-center count.")

        tm_like = (seg_eval == 1)
        other = ~tm_like
        if np.any(other):
            dot_scatter = normal_eval @ scatter_dirs.T
            if z_eff is None:
                blend = np.full_like(sigma_eval, complex(float(cfie_eps), 0.0))
            else:
                z_eval = np.asarray(z_eff, dtype=np.complex128)
                if z_eval.ndim == 1:
                    z_eval = z_eval.reshape(-1, 1)
                if z_eval.shape[0] != centers.shape[0]:
                    raise ValueError("z_eff row count must match panel-center count.")
                if z_eval.shape[1] == 1 and sigma_eval.shape[1] > 1:
                    z_eval = np.repeat(z_eval, sigma_eval.shape[1], axis=1)
                if z_eval.shape != sigma_eval.shape:
                    raise ValueError("z_eff shape must match sigma_mat shape, or be Nx1.")
                y_eval = np.zeros_like(z_eval, dtype=np.complex128)
                nz = np.abs(z_eval) > 1e-9
                y_eval[nz] = 1.0 / z_eval[nz]
                blend = (1j * k0) * y_eval + complex(float(cfie_eps), 0.0)

            far_factor = (1j * dot_scatter) + blend / complex(1j * k0 if abs(k0) > EPS else 1.0, 0.0)
            amp_terms[other, :] = weighted[other, :] * far_factor[other, :] * phase_mat[other, :]

    amp_vec = np.sum(amp_terms, axis=0)

    sigma_lin = _rcs_sigma_from_amp(amp_vec, k0)
    return np.asarray(sigma_lin, dtype=float), np.asarray(amp_vec, dtype=np.complex128)


def _incident_plane_wave(panels: List[Panel], k_air: float, elev_deg: float) -> np.ndarray:
    """Incident field trace sampled at panel centers for one elevation."""
    centers = np.array([p.center for p in panels], dtype=float)
    mat = _incident_plane_wave_many(centers, k_air, np.asarray([elev_deg], dtype=float))
    return mat[:, 0]


def _incident_plane_wave_many(
    centers: np.ndarray,
    k_air: float,
    elevations_deg: np.ndarray,
) -> np.ndarray:
    """Vectorized incident plane-wave traces for many elevations, shape (N,E)."""

    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    phi = np.deg2rad(elev)
    directions = np.stack([-np.cos(phi), -np.sin(phi)], axis=1)      # (E,2)
    return np.asarray(np.exp((-1j * k_air) * (centers @ directions.T)), dtype=np.complex128)


def _assemble_coupled_region_row(
    obs_idx: int,
    region_flag: int,
    s_mat: np.ndarray,
    k_mat: np.ndarray,
    infos: List[PanelCoupledInfo],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble one coupled BIE row for a chosen region at one observation panel.

    Returns coefficients for [u_trace, q_minus].
    """

    n = len(infos)
    row_u = np.zeros(n, dtype=np.complex128)
    row_q = np.zeros(n, dtype=np.complex128)
    row_u[obs_idx] += 0.5

    for j, info in enumerate(infos):
        s = s_mat[obs_idx, j]
        k = k_mat[obs_idx, j]
        if info.minus_region == region_flag:
            row_u[j] += k
            row_q[j] -= s
        if info.plus_region == region_flag:
            row_u[j] += -k + s * info.q_plus_gamma
            row_q[j] += s * info.q_plus_beta

    return row_u, row_q


def _junction_vertex_tol(panels: List[Panel]) -> float:
    coords = np.asarray([p for panel in panels for p in (panel.p0, panel.p1)], dtype=float)
    if coords.size == 0:
        return 1e-9
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    return max(1e-9, 1e-6 * max(diag, 1.0))


def _group_panel_vertices(panels: List[Panel], tol: float) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    grouped: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    inv_tol = 1.0 / max(tol, 1e-12)
    for idx, panel in enumerate(panels):
        key0 = (int(round(float(panel.p0[0]) * inv_tol)), int(round(float(panel.p0[1]) * inv_tol)))
        key1 = (int(round(float(panel.p1[0]) * inv_tol)), int(round(float(panel.p1[1]) * inv_tol)))
        # endpoint_sign = +1 for p0, -1 for p1 (edge direction away from node).
        grouped.setdefault(key0, []).append((idx, +1))
        grouped.setdefault(key1, []).append((idx, -1))
    return grouped


def _build_junction_trace_constraints(
    panels: List[Panel],
    infos: List[PanelCoupledInfo],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build linear constraints used to stabilize/regularize multi-surface junctions.

    Adds:
    - trace continuity constraints (u_i = u_ref),
    - region-wise flux balance constraints at shared nodes.
    """

    n = len(panels)
    tol = _junction_vertex_tol(panels)
    grouped = _group_panel_vertices(panels, tol)
    rows: List[np.ndarray] = []
    trace_count = 0
    flux_count = 0
    junction_nodes = 0
    constrained_panels: Set[int] = set()
    orientation_conflict_nodes = 0

    for incident in grouped.values():
        by_panel_sign: Dict[int, int] = {}
        for idx, endpoint_sign in incident:
            by_panel_sign[idx] = by_panel_sign.get(idx, 0) + int(endpoint_sign)
        unique_panels = sorted(by_panel_sign.keys())
        if len(unique_panels) < 2:
            continue
        seg_names = {panels[i].name for i in unique_panels}
        # True multi-surface junctions (>=3 panels) and explicit cross-segment joins (>=2 distinct segments).
        if len(unique_panels) < 3 and len(seg_names) < 2:
            continue
        if len(seg_names) >= 2:
            signs = [int(np.sign(by_panel_sign.get(i, 0))) for i in unique_panels]
            has_pos = any(s > 0 for s in signs)
            has_neg = any(s < 0 for s in signs)
            if not (has_pos and has_neg):
                orientation_conflict_nodes += 1

        ref = unique_panels[0]
        for other in unique_panels[1:]:
            row = np.zeros(2 * n, dtype=np.complex128)
            row[ref] = 1.0 + 0.0j
            row[other] = -1.0 + 0.0j
            rows.append(row)
            trace_count += 1
            constrained_panels.add(ref)
            constrained_panels.add(other)

        region_set: Set[int] = set()
        for idx in unique_panels:
            info = infos[idx]
            if info.minus_region >= 0:
                region_set.add(info.minus_region)
            if info.plus_region >= 0:
                region_set.add(info.plus_region)

        for region in sorted(region_set):
            row = np.zeros(2 * n, dtype=np.complex128)
            terms = 0
            for idx in unique_panels:
                endpoint_sign = by_panel_sign.get(idx, 0)
                if endpoint_sign == 0:
                    continue
                info = infos[idx]
                coeff_u = 0.0 + 0.0j
                coeff_q = 0.0 + 0.0j
                participates = False

                if info.minus_region == region:
                    coeff_q += 1.0 + 0.0j
                    participates = True
                if info.plus_region == region:
                    coeff_u += info.q_plus_gamma
                    coeff_q += info.q_plus_beta
                    participates = True
                if not participates:
                    continue

                w = complex(float(endpoint_sign), 0.0)
                row[idx] += w * coeff_u
                row[n + idx] += w * coeff_q
                terms += 1
                constrained_panels.add(idx)

            if terms >= 2:
                rows.append(row)
                flux_count += 1
        junction_nodes += 1

    if not rows:
        return np.zeros((0, 2 * n), dtype=np.complex128), {
            "junction_nodes": 0,
            "junction_constraints": 0,
            "junction_panels": 0,
            "junction_trace_constraints": 0,
            "junction_flux_constraints": 0,
            "junction_orientation_conflict_nodes": int(orientation_conflict_nodes),
        }

    c_mat = np.vstack(rows)
    return c_mat, {
        "junction_nodes": int(junction_nodes),
        "junction_constraints": int(c_mat.shape[0]),
        "junction_panels": int(len(constrained_panels)),
        "junction_trace_constraints": int(trace_count),
        "junction_flux_constraints": int(flux_count),
        "junction_orientation_conflict_nodes": int(orientation_conflict_nodes),
    }


def _augment_system_with_constraints(
    a_mat: np.ndarray,
    rhs: np.ndarray,
    c_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a square exact constrained-least-squares KKT system.

    This helper is retained for callers that explicitly need an array-only
    system. It enforces C x = 0 exactly by solving

        minimize ||A x - rhs||_2  subject to  C x = 0

    through the normal-equation KKT system

        [A^H A   C^H] [x]   [A^H rhs]
        [  C      0 ] [λ] = [   0   ]

    The production reuse path uses a null-space reduced solve instead, which
    avoids squaring the condition number.
    """

    a_eval = np.asarray(a_mat, dtype=np.complex128)
    rhs_eval = np.asarray(rhs, dtype=np.complex128)
    c_eval = np.asarray(c_mat, dtype=np.complex128)
    if c_eval.size == 0:
        return a_eval, rhs_eval
    if c_eval.ndim != 2 or c_eval.shape[1] != a_eval.shape[1]:
        raise ValueError(
            "Constraint matrix width does not match the primal system size."
        )

    normal = a_eval.conj().T @ a_eval
    kkt_top = np.hstack([normal, c_eval.conj().T])
    kkt_bottom = np.hstack([
        c_eval,
        np.zeros((c_eval.shape[0], c_eval.shape[0]), dtype=np.complex128),
    ])
    rhs_kkt = np.concatenate([
        a_eval.conj().T @ rhs_eval,
        np.zeros(c_eval.shape[0], dtype=np.complex128),
    ])
    return np.vstack([kkt_top, kkt_bottom]), rhs_kkt


def _build_coupled_matrix(
    panels: List[Panel],
    infos: List[PanelCoupledInfo],
    region_ops: Dict[int, Tuple[np.ndarray, np.ndarray]],
    pol: str,
) -> np.ndarray:
    """Assemble coupled dielectric system matrix A (unknowns: [u_trace, q_minus])."""

    n = len(panels)
    a_mat = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    row = 0

    for i, info in enumerate(infos):
        active_region = info.minus_region if info.minus_region >= 0 else info.plus_region
        if active_region < 0:
            raise ValueError("Coupled formulation requires at least one valid non-PEC medium side per panel.")

        s_active, k_active = region_ops[active_region]
        row_u, row_q = _assemble_coupled_region_row(i, active_region, s_active, k_active, infos)
        a_mat[row, :n] = row_u
        a_mat[row, n:] = row_q
        row += 1

        if info.bc_kind == "transmission":
            passive_region = info.plus_region if active_region == info.minus_region else info.minus_region
            if passive_region >= 0:
                s_passive, k_passive = region_ops[passive_region]
                row_u, row_q = _assemble_coupled_region_row(i, passive_region, s_passive, k_passive, infos)
                a_mat[row, :n] = row_u
                a_mat[row, n:] = row_q
                row += 1
                continue

        z = info.robin_impedance
        physical_region = active_region
        coeff_u, coeff_q = _region_side_trace_coefficients(info, physical_region)
        if abs(z) <= EPS:
            # PEC limits for the direct TE/TM labels used by this file:
            # - TM: u = 0 (Dirichlet branch)
            # - TE: q = 0 (Neumann branch)
            if pol == "TM":
                a_mat[row, i] = 1.0 + 0.0j
            else:
                a_mat[row, i] = coeff_u
                a_mat[row, n + i] = coeff_q
        else:
            eps_phys = info.eps_minus if physical_region == info.minus_region else info.eps_plus
            mu_phys = info.mu_minus if physical_region == info.minus_region else info.mu_plus
            k_phys = info.k_minus if physical_region == info.minus_region else info.k_plus
            alpha = _surface_robin_alpha(pol, eps_phys, mu_phys, k_phys, z)
            a_mat[row, i] = coeff_u + alpha
            a_mat[row, n + i] = coeff_q
        row += 1

    if row != 2 * n:
        raise RuntimeError(f"Internal coupled assembly mismatch: built {row} rows for {2 * n} unknowns.")
    return a_mat


def _build_coupled_rhs(
    panels: List[Panel],
    infos: List[PanelCoupledInfo],
    k_air: float,
    elev_deg: float,
) -> np.ndarray:
    """Assemble RHS for coupled dielectric system at one elevation."""

    centers = np.array([p.center for p in panels], dtype=float)
    u_inc_air = _incident_plane_wave_many(centers, k_air, np.asarray([elev_deg], dtype=float))
    rhs_mat = _build_coupled_rhs_many(infos=infos, u_inc_air=u_inc_air)
    return rhs_mat[:, 0]


def _build_coupled_rhs_many(
    infos: List[PanelCoupledInfo],
    u_inc_air: np.ndarray,
) -> np.ndarray:
    """
    Vectorized coupled RHS assembly for many elevations.

    Input `u_inc_air` has shape (N_panels, N_elevations); output is (2N, N_elevations).
    Row ordering matches _build_coupled_matrix(): first the active non-PEC side, then the
    opposite transmission side (if any) or the Robin/PEC boundary condition row.
    """

    n = len(infos)
    u_eval = np.asarray(u_inc_air, dtype=np.complex128)
    if u_eval.ndim == 1:
        u_eval = u_eval.reshape(-1, 1)
    if u_eval.shape[0] != n:
        raise ValueError("u_inc_air row count must match coupled panel count.")

    active_inc = np.zeros(n, dtype=bool)
    passive_inc = np.zeros(n, dtype=bool)
    for i, info in enumerate(infos):
        active_is_minus = info.minus_region >= 0
        active_inc[i] = info.minus_has_incident if active_is_minus else info.plus_has_incident
        if info.bc_kind == "transmission":
            passive_inc[i] = info.plus_has_incident if active_is_minus else info.minus_has_incident

    rhs = np.zeros((2 * n, u_eval.shape[1]), dtype=np.complex128)
    if np.any(active_inc):
        rhs[0::2, :] = np.where(active_inc[:, None], u_eval, 0.0 + 0.0j)
    if np.any(passive_inc):
        rhs[1::2, :] = np.where(passive_inc[:, None], u_eval, 0.0 + 0.0j)
    return rhs


def _build_coupled_system(
    panels: List[Panel],
    infos: List[PanelCoupledInfo],
    region_ops: Dict[int, Tuple[np.ndarray, np.ndarray]],
    pol: str,
    k_air: float,
    elev_deg: float,
    junction_constraints: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble coupled dielectric system (unknowns: [u_trace, q_minus])."""

    a_mat = _build_coupled_matrix(
        panels=panels,
        infos=infos,
        region_ops=region_ops,
        pol=pol,
    )
    rhs = _build_coupled_rhs(
        panels=panels,
        infos=infos,
        k_air=k_air,
        elev_deg=elev_deg,
    )

    if junction_constraints is not None and junction_constraints.size > 0:
        return _augment_system_with_constraints(a_mat, rhs, junction_constraints)
    return a_mat, rhs


def _backscatter_rcs_coupled(
    panels: List[Panel],
    infos: List[PanelCoupledInfo],
    u_trace: np.ndarray,
    q_minus: np.ndarray,
    k_air: float,
    elev_deg: float,
) -> Tuple[float, complex]:
    """Convert coupled solution traces/fluxes into monostatic 2D RCS and complex amplitude."""

    centers = np.array([p.center for p in panels], dtype=float)
    normals = np.array([p.normal for p in panels], dtype=float)
    lengths = np.array([p.length for p in panels], dtype=float)
    sigma_lin_vec, amp_vec = _backscatter_rcs_coupled_many(
        centers=centers,
        normals=normals,
        lengths=lengths,
        infos=infos,
        u_trace_mat=np.asarray(u_trace, dtype=np.complex128).reshape(-1, 1),
        q_minus_mat=np.asarray(q_minus, dtype=np.complex128).reshape(-1, 1),
        k_air=k_air,
        elevations_deg=np.asarray([elev_deg], dtype=float),
    )
    return float(sigma_lin_vec[0]), complex(amp_vec[0])


def _backscatter_rcs_coupled_many(
    centers: np.ndarray,
    normals: np.ndarray,
    lengths: np.ndarray,
    infos: List[PanelCoupledInfo],
    u_trace_mat: np.ndarray,
    q_minus_mat: np.ndarray,
    k_air: float,
    elevations_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized coupled backscatter computation for many elevations.

    Returns (sigma_linear[E], amplitude[E]).
    """

    u_eval = np.asarray(u_trace_mat, dtype=np.complex128)
    q_eval = np.asarray(q_minus_mat, dtype=np.complex128)
    if u_eval.ndim == 1:
        u_eval = u_eval.reshape(-1, 1)
    if q_eval.ndim == 1:
        q_eval = q_eval.reshape(-1, 1)
    if u_eval.shape != q_eval.shape:
        raise ValueError("u_trace_mat and q_minus_mat must have matching shapes.")
    if u_eval.shape[0] != len(infos):
        raise ValueError("Coupled solution rows must match coupled panel count.")

    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    phi = np.deg2rad(elev)
    scatter_dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)      # (E,2)
    phase_mat = np.exp((1j * k_air) * (centers @ scatter_dirs.T))    # (N,E)
    dot_scatter = normals @ scatter_dirs.T                            # (N,E)
    if u_eval.shape[1] != phase_mat.shape[1]:
        raise ValueError("Coupled solution column count must match elevation count.")

    beta = np.asarray([info.q_plus_beta for info in infos], dtype=np.complex128)
    gamma = np.asarray([info.q_plus_gamma for info in infos], dtype=np.complex128)
    minus_inc = np.asarray([info.minus_has_incident for info in infos], dtype=bool)
    plus_inc = np.asarray([info.plus_has_incident for info in infos], dtype=bool)

    q_plus = beta[:, None] * q_eval + gamma[:, None] * u_eval
    amp_terms = np.zeros_like(u_eval, dtype=np.complex128)

    if np.any(minus_inc):
        amp_terms[minus_inc, :] += (
            -q_eval[minus_inc, :]
            + 1j * k_air * dot_scatter[minus_inc, :] * u_eval[minus_inc, :]
        ) * lengths[minus_inc, None] * phase_mat[minus_inc, :]

    if np.any(plus_inc):
        amp_terms[plus_inc, :] += (
            q_plus[plus_inc, :]
            - 1j * k_air * dot_scatter[plus_inc, :] * u_eval[plus_inc, :]
        ) * lengths[plus_inc, None] * phase_mat[plus_inc, :]

    amp_vec = np.sum(amp_terms, axis=0)
    sigma_lin = _rcs_sigma_from_amp(amp_vec, k_air)
    return np.asarray(sigma_lin, dtype=float), np.asarray(amp_vec, dtype=np.complex128)


def _build_coupled_region_operators(
    panels: List[Panel],
    infos: List[PanelCoupledInfo],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Build/reuse per-region operators keyed by complex medium wavenumber."""

    region_to_k: Dict[int, complex] = {}
    for info in infos:
        if info.minus_region >= 0:
            region_to_k[info.minus_region] = complex(info.k_minus)
        if info.plus_region >= 0:
            region_to_k[info.plus_region] = complex(info.k_plus)

    by_k: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
    region_ops: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for region, k_region in region_to_k.items():
        key = (round(float(k_region.real), 12), round(float(k_region.imag), 12))
        if key not in by_k:
            k_eval = k_region if abs(k_region) > EPS else (EPS + 0.0j)
            by_k[key] = _build_operator_matrices_coupled(panels, k_eval)
        region_ops[region] = by_k[key]
    return region_ops


def solve_monostatic_rcs_2d(
    geometry_snapshot: Dict[str, Any],
    frequencies_ghz: List[float],
    elevations_deg: List[float],
    polarization: str,
    geometry_units: str = "inches",
    material_base_dir: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    quality_thresholds: Dict[str, float | int] | None = None,
    strict_quality_gate: bool = False,
    max_panels: int = MAX_PANELS_DEFAULT,
    compute_condition_number: bool = False,
    parallel_elevations: bool = True,
    max_elevation_workers: int = 0,
    reuse_angle_invariant_matrix: bool = True,
    mesh_reference_ghz: float | None = None,
    cfie_eps_override: float | None = None,
    rcs_normalization_mode: str = RCS_NORM_MODE_DEFAULT,
    basis_family: str = "pulse",
    testing_family: str = "collocation",
) -> Dict[str, Any]:
    """
    Main entry point for monostatic 2D RCS.

    Per frequency:
    - build operators,
    - per elevation assemble/solve linear system,
    - compute backscatter RCS and diagnostics.

    Angle convention (coming-from):
    - 0 deg: from right to left
    - +90 deg: from top to bottom
    - -90 deg: from bottom to top

    Performance controls:
    - compute_condition_number=False skips expensive per-angle condition estimates.
    - reuse_angle_invariant_matrix=True reuses one matrix factorization when A is
      elevation-invariant.
    - parallel_elevations=True enables per-elevation threading for angle-varying A.
    - mesh_reference_ghz sets a fixed panelization reference frequency for mesh
      generation (useful for cross-frequency apples-to-apples comparisons).
    - cfie_eps_override sets a fixed CFIE blend (disables adaptive CFIE for legacy
      formulation when provided).
    - rcs_normalization_mode is retained only for backward compatibility and now
      accepts physical normalization aliases only; stored values are always
      sigma_2d = |A|^2 / (4 k).
    - basis_family/testing_family support the production pulse/collocation path and
      a hardened linear/Galerkin coupled path.
    - The linear path now splits mixed-interface shared nodes automatically.
    - True branching multi-surface nodes still fall back to pulse/collocation.
    """

    if not frequencies_ghz:
        raise ValueError("At least one frequency is required.")
    if not elevations_deg:
        raise ValueError("At least one elevation angle is required.")

    frequencies = [float(f) for f in frequencies_ghz]
    elevations = [float(e) for e in elevations_deg]
    basis_mode = str(basis_family or "pulse").strip().lower()
    testing_mode = str(testing_family or "collocation").strip().lower()
    requested_basis_mode = basis_mode
    requested_testing_mode = testing_mode
    if (basis_mode, testing_mode) not in {("pulse", "collocation"), ("linear", "galerkin")}:
        raise NotImplementedError(
            "Unsupported discretization family. Supported pairs are pulse/collocation and "
            "linear/galerkin."
        )
    linear_galerkin_requested = (basis_mode, testing_mode) == ("linear", "galerkin")
    linear_galerkin_mode = linear_galerkin_requested
    effective_basis_mode = basis_mode
    effective_testing_mode = testing_mode
    discretization_fallback_used = False
    discretization_fallback_reason: str | None = None
    if any(f <= 0.0 for f in frequencies):
        raise ValueError("Frequencies must be positive GHz values.")
    mesh_ref_ghz: float | None = None
    if mesh_reference_ghz is not None:
        mesh_ref_ghz = float(mesh_reference_ghz)
        if (not math.isfinite(mesh_ref_ghz)) or mesh_ref_ghz <= 0.0:
            raise ValueError("mesh_reference_ghz must be a positive finite GHz value.")

    cfie_eps_fixed: float | None = None
    if cfie_eps_override is not None:
        cfie_eps_fixed = float(cfie_eps_override)
        if (not math.isfinite(cfie_eps_fixed)) or cfie_eps_fixed <= 0.0:
            raise ValueError("cfie_eps_override must be a positive finite value.")
    rcs_norm_mode = _normalize_rcs_normalization_mode(rcs_normalization_mode)
    _raise_if_untrusted_math_backends()

    pol = _normalize_polarization(polarization)
    unit_scale = _unit_scale_to_meters(geometry_units)

    base_dir = material_base_dir or os.getcwd()
    preflight_report = validate_geometry_snapshot_for_solver(geometry_snapshot, base_dir=base_dir)
    materials = MaterialLibrary.from_entries(
        geometry_snapshot.get("ibcs", []) or [],
        geometry_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    for _msg in list(preflight_report.get('warnings', []) or []):
        materials.warn_once(str(_msg))

    samples: List[Dict[str, Any]] = []
    total_steps = len(frequencies) * (len(elevations) + 1)
    done_steps = 0

    residual_values: List[float] = []
    constraint_residual_values: List[float] = []
    cond_values: List[float] = []
    cfie_eps_values: List[float] = []
    mesh_reference_values: List[float] = []
    panel_count_values: List[int] = []
    panel_length_min_values: List[float] = []
    panel_length_max_values: List[float] = []
    elevations_arr = np.asarray(elevations, dtype=float)
    reused_matrix_solve_count = 0
    parallel_elevation_solve_count = 0
    max_parallel_workers_used = 1
    coupled_mode_global: bool | None = None
    formulation_label = "2D BIE/MoM pulse-point EFIE/MFIE"
    junction_stats = {
        "junction_nodes": 0,
        "junction_constraints": 0,
        "junction_panels": 0,
        "junction_trace_constraints": 0,
        "junction_flux_constraints": 0,
        "junction_orientation_conflict_nodes": 0,
    }

    def emit_progress(message: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(done_steps, total_steps, message)
        except Exception:
            pass

    emit_progress("Initializing solver")

    for freq_ghz in frequencies:
        freq_hz = freq_ghz * 1e9
        k0 = 2.0 * math.pi * freq_hz / C0
        mesh_freq_ghz = mesh_ref_ghz if mesh_ref_ghz is not None else float(freq_ghz)
        lambda_min = C0 / (mesh_freq_ghz * 1e9)
        panels = _build_panels(
            geometry_snapshot,
            unit_scale,
            lambda_min,
            max_panels=max_panels,
        )
        panel_lengths = np.asarray([p.length for p in panels], dtype=float)
        panel_centers = np.asarray([p.center for p in panels], dtype=float)
        panel_normals = np.asarray([p.normal for p in panels], dtype=float)
        panel_seg_types = np.asarray([p.seg_type for p in panels], dtype=int)
        mesh_reference_values.append(float(mesh_freq_ghz))
        panel_count_values.append(int(len(panels)))
        panel_length_min_values.append(float(np.min(panel_lengths)) if len(panel_lengths) else 0.0)
        panel_length_max_values.append(float(np.max(panel_lengths)) if len(panel_lengths) else 0.0)
        coupled_mode = _needs_coupled_formulation(panels)
        if coupled_mode_global is None:
            coupled_mode_global = bool(coupled_mode)
        elif bool(coupled_mode_global) != bool(coupled_mode):
            raise RuntimeError("Internal error: coupled formulation mode changed across frequency-dependent re-meshing.")

        if coupled_mode:
            if linear_galerkin_mode:
                formulation_label = "2D BIE/MoM coupled dielectric trace formulation (linear Galerkin)"
                preview_infos = _build_coupled_panel_info(panels, materials, freq_ghz, pol, k0)
                mesh, linear_mesh_stats_local = _build_linear_mesh_interface_aware(panels, preview_infos)
                coupled_infos = _build_linear_coupled_infos(mesh, materials, freq_ghz, pol, k0)
                linear_mesh_stats_local = dict(linear_mesh_stats_local)
                linear_mesh_stats_local.update(_linear_coupled_node_report(mesh, coupled_infos))
                done_steps += 1
                emit_progress(f"Assembled linear/Galerkin coupled operators at {freq_ghz:g} GHz")

                nnodes = len(mesh.nodes)
                linear_junction_constraints, linear_junction_stats = _build_linear_junction_constraints(
                    mesh,
                    coupled_infos,
                )
                junction_stats.update(linear_mesh_stats_local)
                junction_stats.update(linear_junction_stats)
                if int(linear_junction_stats.get("junction_orientation_conflict_nodes", 0)) > 0:
                    raise ValueError(
                        "Detected "
                        f"{int(linear_junction_stats.get('junction_orientation_conflict_nodes', 0))} cross-segment "
                        "junction node(s) with inconsistent segment orientation. Fix the geometry so shared "
                        "junctions have a physically consistent plus/minus side assignment before solving."
                    )
                if linear_junction_constraints.size > 0:
                    formulation_label = "2D BIE/MoM coupled dielectric trace formulation (linear Galerkin + junction constraints)"
                    materials.warn_once(
                        (
                            "Applied "
                            f"{int(linear_junction_stats.get('junction_constraints', 0))} linear/Galerkin junction constraint(s) "
                            f"(trace={int(linear_junction_stats.get('junction_trace_constraints', 0))}, "
                            f"flux={int(linear_junction_stats.get('junction_flux_constraints', 0))}) "
                            f"across {int(linear_junction_stats.get('junction_nodes', 0))} node(s)."
                        )
                    )

                a_core = _build_coupled_matrix_linear(
                    mesh=mesh,
                    infos=coupled_infos,
                    pol=pol,
                )

                rhs_mat = _build_coupled_rhs_many_linear(
                    mesh=mesh,
                    infos=coupled_infos,
                    k_air=k0,
                    elevations_deg=elevations_arr,
                )
                a_mat = a_core
                constraint_mat = linear_junction_constraints if linear_junction_constraints.size > 0 else None
                _ensure_finite_linear_system(a_mat, rhs_mat, label="linear/Galerkin coupled system")
                prepared = _prepare_linear_solver(a_mat, constraint_mat=constraint_mat)
                if compute_condition_number:
                    cond_eval_mat = prepared.reduced_mat if prepared.reduced_mat is not None else a_mat
                    cond_values.append(_cond_estimate(cond_eval_mat))
                sol_mat = _solve_with_prepared_solver(prepared, rhs_mat)
                if sol_mat.ndim == 1:
                    sol_mat = sol_mat.reshape(-1, 1)

                primal_sol_mat = sol_mat[: 2 * nnodes, :]
                reused_matrix_solve_count += len(elevations)
                max_parallel_workers_used = max(max_parallel_workers_used, 1)
                residual_vec = _residual_norm_many(a_mat, primal_sol_mat, rhs_mat)
                constraint_residual_vec = _constraint_residual_norm_many(constraint_mat, primal_sol_mat)
                rcs_lin_vec, amp_vec = _backscatter_rcs_coupled_many_linear(
                    mesh=mesh,
                    infos=coupled_infos,
                    u_trace_nodes_mat=primal_sol_mat[:nnodes, :],
                    q_minus_nodes_mat=primal_sol_mat[nnodes:, :],
                    k_air=k0,
                    elevations_deg=elevations_arr,
                )
                rcs_db_vec = 10.0 * np.log10(rcs_lin_vec)

                for idx, elev_deg in enumerate(elevations):
                    amp_val = complex(amp_vec[idx])
                    residual_local = float(residual_vec[idx])
                    samples.append(
                        {
                            "frequency_ghz": float(freq_ghz),
                            "theta_inc_deg": float(elev_deg),
                            "theta_scat_deg": float(elev_deg),
                            "rcs_linear": float(rcs_lin_vec[idx]),
                            "rcs_db": float(rcs_db_vec[idx]),
                            "rcs_amp_real": float(np.real(amp_val)),
                            "rcs_amp_imag": float(np.imag(amp_val)),
                            "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                            "linear_residual": residual_local,
                        }
                    )
                    residual_values.append(residual_local)
                    constraint_residual_values.append(float(constraint_residual_vec[idx]))
                    done_steps += 1
                    emit_progress(f"Solved {freq_ghz:g} GHz at {elev_deg:g} deg")
                continue

            formulation_label = "2D BIE/MoM coupled dielectric trace formulation"
            coupled_infos = _build_coupled_panel_info(panels, materials, freq_ghz, pol, k0)
            junction_constraints, junction_stats = _build_junction_trace_constraints(panels, infos=coupled_infos)
            if junction_stats["junction_constraints"] > 0:
                materials.warn_once(
                    (
                        "Applied "
                        f"{junction_stats['junction_constraints']} junction constraint(s) "
                        f"(trace={junction_stats['junction_trace_constraints']}, "
                        f"flux={junction_stats['junction_flux_constraints']}) "
                        f"across {junction_stats['junction_nodes']} node(s)."
                    )
                )
            if int(junction_stats.get("junction_orientation_conflict_nodes", 0)) > 0:
                raise ValueError(
                    "Detected "
                    f"{int(junction_stats.get('junction_orientation_conflict_nodes', 0))} cross-segment "
                    "junction node(s) with inconsistent segment orientation. Fix the geometry so shared "
                    "junctions have a physically consistent plus/minus side assignment before solving."
                )
            region_ops = _build_coupled_region_operators(panels, coupled_infos)
            done_steps += 1
            emit_progress(f"Assembled coupled operators at {freq_ghz:g} GHz")

            n_panels = len(panels)
            a_core = _build_coupled_matrix(
                panels=panels,
                infos=coupled_infos,
                region_ops=region_ops,
                pol=pol,
            )
            a_mat = a_core
            constraint_mat = junction_constraints if junction_constraints.size > 0 else None

            rhs_mat = _build_coupled_rhs_many(
                infos=coupled_infos,
                u_inc_air=_incident_plane_wave_many(panel_centers, k0, elevations_arr),
            )
            _ensure_finite_linear_system(a_mat, rhs_mat, label="coupled pulse/collocation system")
            prepared = _prepare_linear_solver(a_mat, constraint_mat=constraint_mat)
            if compute_condition_number:
                cond_eval_mat = prepared.reduced_mat if prepared.reduced_mat is not None else a_mat
                cond_values.append(_cond_estimate(cond_eval_mat))

            sol_mat = _solve_with_prepared_solver(prepared, rhs_mat)
            if sol_mat.ndim == 1:
                sol_mat = sol_mat.reshape(-1, 1)

            primal_sol_mat = sol_mat[: 2 * n_panels, :]

            reused_matrix_solve_count += len(elevations)
            max_parallel_workers_used = max(max_parallel_workers_used, 1)

            residual_vec = _residual_norm_many(a_mat, primal_sol_mat, rhs_mat)
            constraint_residual_vec = _constraint_residual_norm_many(constraint_mat, primal_sol_mat)
            rcs_lin_vec, amp_vec = _backscatter_rcs_coupled_many(
                centers=panel_centers,
                normals=panel_normals,
                lengths=panel_lengths,
                infos=coupled_infos,
                u_trace_mat=primal_sol_mat[:n_panels, :],
                q_minus_mat=primal_sol_mat[n_panels : 2 * n_panels, :],
                k_air=k0,
                elevations_deg=elevations_arr,
            )
            rcs_db_vec = 10.0 * np.log10(rcs_lin_vec)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append(
                    {
                        "frequency_ghz": float(freq_ghz),
                        "theta_inc_deg": float(elev_deg),
                        "theta_scat_deg": float(elev_deg),
                        "rcs_linear": float(rcs_lin_vec[idx]),
                        "rcs_db": float(rcs_db_vec[idx]),
                        "rcs_amp_real": float(np.real(amp_val)),
                        "rcs_amp_imag": float(np.imag(amp_val)),
                        "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                        "linear_residual": residual_local,
                    }
                )
                residual_values.append(residual_local)
                constraint_residual_values.append(float(constraint_residual_vec[idx]))
                done_steps += 1
                emit_progress(f"Solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        s_mat, kp_mat = _build_operator_matrices(panels, k0)
        cfie_eps_freq = (
            float(cfie_eps_fixed)
            if cfie_eps_fixed is not None
            else _adaptive_cfie_eps(k0, panel_lengths)
        )
        cfie_eps_values.append(float(cfie_eps_freq))
        done_steps += 1
        emit_progress(f"Assembled operators at {freq_ghz:g} GHz")

        angle_invariant_matrix = bool(
            reuse_angle_invariant_matrix and not any(p.seg_type in {3, 5} for p in panels)
        )

        if angle_invariant_matrix:
            z_eff = np.zeros(len(panels), dtype=np.complex128)
            for i, p in enumerate(panels):
                z_eff[i] = _panel_effective_impedance(p, materials, freq_ghz, pol, 1.0)

            a_mat = _build_system_matrix(
                panels=panels,
                s_mat=s_mat,
                kp_mat=kp_mat,
                z_eff=z_eff,
                pol=pol,
                k0=k0,
                cfie_eps=cfie_eps_freq,
            )
            if compute_condition_number:
                cond_values.append(_cond_estimate(a_mat))
            prepared = _prepare_linear_solver(a_mat)

            u_inc_mat, du_dn_mat, _ = _incident_values_many(panel_centers, panel_normals, k0, elevations_arr)
            rhs_mat = _build_system_rhs_many(
                seg_types=panel_seg_types,
                u_inc_mat=u_inc_mat,
                du_dn_mat=du_dn_mat,
                z_eff=z_eff,
                pol=pol,
                k0=k0,
                cfie_eps=cfie_eps_freq,
            )
            sol_mat = _solve_with_prepared_solver(prepared, rhs_mat)
            if sol_mat.ndim == 1:
                sol_mat = sol_mat.reshape(-1, 1)

            reused_matrix_solve_count += len(elevations)
            max_parallel_workers_used = max(max_parallel_workers_used, 1)

            residual_vec = _residual_norm_many(a_mat, sol_mat, rhs_mat)
            rcs_lin_vec, amp_vec = _backscatter_rcs_many(
                centers=panel_centers,
                lengths=panel_lengths,
                sigma_mat=sol_mat,
                k0=k0,
                elevations_deg=elevations_arr,
                normals=panel_normals,
                seg_types=panel_seg_types,
                pol=pol,
                z_eff=z_eff,
                cfie_eps=cfie_eps_freq,
            )
            rcs_db_vec = 10.0 * np.log10(rcs_lin_vec)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append(
                    {
                        "frequency_ghz": float(freq_ghz),
                        "theta_inc_deg": float(elev_deg),
                        "theta_scat_deg": float(elev_deg),
                        "rcs_linear": float(rcs_lin_vec[idx]),
                        "rcs_db": float(rcs_db_vec[idx]),
                        "rcs_amp_real": float(np.real(amp_val)),
                        "rcs_amp_imag": float(np.imag(amp_val)),
                        "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                        "linear_residual": residual_local,
                    }
                )
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"Solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        u_inc_all, du_dn_all, cos_inc_all = _incident_values_many(panel_centers, panel_normals, k0, elevations_arr)

        def _solve_one_elevation(idx: int) -> Tuple[int, np.ndarray, np.ndarray, float, float | None]:
            u_inc = u_inc_all[:, idx]
            du_dn = du_dn_all[:, idx]
            cos_inc = cos_inc_all[:, idx]
            z_local = np.zeros(len(panels), dtype=np.complex128)
            for i, p in enumerate(panels):
                z_local[i] = _panel_effective_impedance(p, materials, freq_ghz, pol, cos_inc[i])
            a_local = _build_system_matrix(
                panels=panels,
                s_mat=s_mat,
                kp_mat=kp_mat,
                z_eff=z_local,
                pol=pol,
                k0=k0,
                cfie_eps=cfie_eps_freq,
                seg_types=panel_seg_types,
            )
            rhs_local = _build_system_rhs(
                panels=panels,
                u_inc=u_inc,
                du_dn=du_dn,
                z_eff=z_local,
                pol=pol,
                k0=k0,
                cfie_eps=cfie_eps_freq,
                seg_types=panel_seg_types,
            )
            cond_local = _cond_estimate(a_local) if compute_condition_number else None
            sigma_local = _solve_linear_system(a_local, rhs_local)
            residual_local = _residual_norm(a_local, sigma_local, rhs_local)
            return (
                idx,
                np.asarray(sigma_local, dtype=np.complex128),
                np.asarray(z_local, dtype=np.complex128),
                float(residual_local),
                cond_local,
            )

        workers = _resolve_worker_count(
            enabled=parallel_elevations,
            requested=max_elevation_workers,
            jobs=len(elevations),
        )
        max_parallel_workers_used = max(max_parallel_workers_used, workers)

        sigma_cols: List[np.ndarray] = []
        z_cols: List[np.ndarray] = []
        residual_ordered: List[float] = []

        if workers > 1:
            parallel_elevation_solve_count += len(elevations)
            staged: List[Tuple[int, np.ndarray, np.ndarray, float, float | None]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                future_map = {
                    ex.submit(_solve_one_elevation, idx): (idx, elev_deg)
                    for idx, elev_deg in enumerate(elevations)
                }
                for fut in concurrent.futures.as_completed(future_map):
                    idx, elev_deg = future_map[fut]
                    i_out, sigma_out, z_out, residual_out, cond_out = fut.result()
                    staged.append((i_out, sigma_out, z_out, residual_out, cond_out))
                    done_steps += 1
                    emit_progress(f"Solved {freq_ghz:g} GHz at {elev_deg:g} deg")

            staged.sort(key=lambda t: t[0])
            for _, sigma_out, z_out, residual_out, cond_out in staged:
                sigma_cols.append(np.asarray(sigma_out, dtype=np.complex128))
                z_cols.append(np.asarray(z_out, dtype=np.complex128))
                residual_ordered.append(float(residual_out))
                if cond_out is not None:
                    cond_values.append(float(cond_out))
        else:
            for idx, elev_deg in enumerate(elevations):
                _, sigma_out, z_out, residual_out, cond_out = _solve_one_elevation(idx)
                sigma_cols.append(np.asarray(sigma_out, dtype=np.complex128))
                z_cols.append(np.asarray(z_out, dtype=np.complex128))
                residual_ordered.append(float(residual_out))
                if cond_out is not None:
                    cond_values.append(float(cond_out))
                done_steps += 1
                emit_progress(f"Solved {freq_ghz:g} GHz at {elev_deg:g} deg")

        sigma_mat = np.column_stack(sigma_cols) if sigma_cols else np.zeros((len(panels), 0), dtype=np.complex128)
        z_eff_mat = np.column_stack(z_cols) if z_cols else np.zeros((len(panels), 0), dtype=np.complex128)
        rcs_lin_vec, amp_vec = _backscatter_rcs_many(
            centers=panel_centers,
            lengths=panel_lengths,
            sigma_mat=sigma_mat,
            k0=k0,
            elevations_deg=elevations_arr,
            normals=panel_normals,
            seg_types=panel_seg_types,
            pol=pol,
            z_eff=z_eff_mat,
            cfie_eps=cfie_eps_freq,
        )
        rcs_db_vec = 10.0 * np.log10(rcs_lin_vec)

        for idx, elev_deg in enumerate(elevations):
            amp_val = complex(amp_vec[idx])
            residual_local = float(residual_ordered[idx])
            samples.append(
                {
                    "frequency_ghz": float(freq_ghz),
                    "theta_inc_deg": float(elev_deg),
                    "theta_scat_deg": float(elev_deg),
                    "rcs_linear": float(rcs_lin_vec[idx]),
                    "rcs_db": float(rcs_db_vec[idx]),
                    "rcs_amp_real": float(np.real(amp_val)),
                    "rcs_amp_imag": float(np.imag(amp_val)),
                    "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                    "linear_residual": residual_local,
                }
            )
            residual_values.append(residual_local)

    done_steps = total_steps
    emit_progress("Completed")

    user_pol = _canonical_user_polarization_label(polarization)
    user_pol_alias = 'VV' if user_pol == 'TE' else 'HH'
    if discretization_fallback_used:
        raise RuntimeError(
            "Aborting solve: discretization fallback was triggered"
            + (f" ({discretization_fallback_reason})." if discretization_fallback_reason else ".")
        )

    result = {
        "title": geometry_snapshot.get("title", "Geometry"),
        "scattering_mode": "monostatic",
        "polarization": user_pol,
        "polarization_export": user_pol,
        "polarization_alias": user_pol_alias,
        "solver_polarization_internal": pol,
        "samples": samples,
        "metadata": {
            "formulation": formulation_label,
            "coupled_dielectric_mode": bool(coupled_mode_global) if coupled_mode_global is not None else False,
            "basis_family_requested": requested_basis_mode,
            "testing_family_requested": requested_testing_mode,
            "basis_family": effective_basis_mode,
            "testing_family": effective_testing_mode,
            "discretization_fallback_used": bool(discretization_fallback_used),
            "discretization_fallback_reason": discretization_fallback_reason,
            "geometry_units_in": geometry_units,
            "polarization_user": user_pol,
            "polarization_alias": user_pol_alias,
            "rcs_normalization_mode": rcs_norm_mode,
            "rcs_linear_quantity": "sigma_2d",
            "rcs_normalization_formula": f"sigma = ({RCS_NORM_NUMERATOR:.12g} / k) * |A|^2",
            "mesh_reference_ghz": float(mesh_ref_ghz) if mesh_ref_ghz is not None else None,
            "mesh_reference_mode": "fixed" if mesh_ref_ghz is not None else "per_frequency",
            "mesh_reference_ghz_min": float(np.min(mesh_reference_values)) if mesh_reference_values else 0.0,
            "mesh_reference_ghz_max": float(np.max(mesh_reference_values)) if mesh_reference_values else 0.0,
            "panel_count": int(np.max(panel_count_values)) if panel_count_values else 0,
            "panel_count_min": int(np.min(panel_count_values)) if panel_count_values else 0,
            "panel_count_max": int(np.max(panel_count_values)) if panel_count_values else 0,
            "panel_length_min_m": float(np.min(panel_length_min_values)) if panel_length_min_values else 0.0,
            "panel_length_max_m": float(np.max(panel_length_max_values)) if panel_length_max_values else 0.0,
            "frequency_count": len(frequencies),
            "elevation_count": len(elevations),
            "max_panels_limit": int(max(1, int(max_panels))),
            "no_accuracy_fallbacks_enforced": True,
            "bessel_backend": _BESSEL.backend_name,
            "complex_hankel_backend": _complex_hankel_backend_name(),
            "junction_nodes": int(junction_stats.get("junction_nodes", 0)),
            "junction_constraints": int(junction_stats.get("junction_constraints", 0)),
            "junction_panels": int(junction_stats.get("junction_panels", 0)),
            "junction_trace_constraints": int(junction_stats.get("junction_trace_constraints", 0)),
            "junction_flux_constraints": int(junction_stats.get("junction_flux_constraints", 0)),
            "junction_orientation_conflict_nodes": int(
                junction_stats.get("junction_orientation_conflict_nodes", 0)
            ),
            "linear_node_count": int(junction_stats.get("linear_node_count", 0)),
            "linear_element_count": int(junction_stats.get("linear_element_count", 0)),
            "linear_geometric_node_count": int(junction_stats.get("linear_geometric_node_count", 0)),
            "linear_interface_split_nodes": int(junction_stats.get("linear_interface_split_nodes", 0)),
            "linear_branching_nodes": int(junction_stats.get("linear_branching_nodes", 0)),
            "linear_mixed_interface_nodes": int(junction_stats.get("linear_mixed_interface_nodes", 0)),
            "linear_unsupported_nodes": int(junction_stats.get("linear_unsupported_nodes", 0)),
            "residual_norm_max": float(np.max(residual_values)) if residual_values else 0.0,
            "residual_norm_mean": float(np.mean(residual_values)) if residual_values else 0.0,
            "constraint_residual_norm_max": float(np.max(constraint_residual_values)) if constraint_residual_values else 0.0,
            "constraint_residual_norm_mean": float(np.mean(constraint_residual_values)) if constraint_residual_values else 0.0,
            "condition_est_computed": bool(compute_condition_number),
            "condition_est_count": int(len(cond_values)),
            "condition_est_max": float(np.max(cond_values)) if cond_values else 0.0,
            "condition_est_mean": float(np.mean(cond_values)) if cond_values else 0.0,
            "reuse_angle_invariant_matrix": bool(reuse_angle_invariant_matrix),
            "matrix_reuse_solve_count": int(reused_matrix_solve_count),
            "parallel_elevations_enabled": bool(parallel_elevations),
            "parallel_elevation_solve_count": int(parallel_elevation_solve_count),
            "parallel_elevation_workers_used": int(max(1, max_parallel_workers_used)),
            "cfie_eps_override": float(cfie_eps_fixed) if cfie_eps_fixed is not None else None,
            "cfie_eps_min": float(np.min(cfie_eps_values)) if cfie_eps_values else 0.0,
            "cfie_eps_max": float(np.max(cfie_eps_values)) if cfie_eps_values else 0.0,
            "cfie_eps_mean": float(np.mean(cfie_eps_values)) if cfie_eps_values else 0.0,
            "warnings": list(materials.warnings),
            "preflight": dict(preflight_report),
        },
    }

    metadata = result.get("metadata", {}) or {}
    quality_gate = evaluate_quality_gate(metadata, thresholds=quality_thresholds)
    metadata["quality_gate"] = quality_gate
    result["metadata"] = metadata

    if strict_quality_gate and not bool(quality_gate.get("passed", False)):
        msg = "; ".join(quality_gate.get("violations", []) or ["quality gate failed"])
        raise ValueError(f"Quality gate failed: {msg}")

    return result