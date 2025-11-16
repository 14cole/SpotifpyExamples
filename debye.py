"""Python equivalents of the legacy CEPSDBG / CEPS2DBX routines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import List

from .rop import RopData

_A0 = 5.563249e-2
_MAX_GAMMA = 10.0
_LOG_BLOCK = 120
_LOG_POINTS = 601


@dataclass
class DebyeControls:
    """Mirrors the FORTRAN parameter control semantics."""

    fres: float = 0.0
    deps: float = 0.0
    epsv: float = 0.0
    gamma: float = 0.0
    sige: float = 0.0


@dataclass
class DebyeFitResult:
    """Holds the outcome of the translated CEPS2DBX optimization."""

    fres: float
    deps: float
    epsv: float
    gamma: float
    sige: float
    rms: float
    sse: float
    iflag: int
    iterations: int
    fitted: List[complex]
    residual_epsr: List[float]
    residual_epsi: List[float]
    log_frequencies: List[float]
    log_ceps: List[complex]

    @property
    def converged(self) -> bool:
        return self.iflag == 0


def compute_debye_ceps(
    fghz: float,
    fres: float,
    deps: float,
    epsv: float,
    gamma: float,
    sige: float,
) -> complex:
    """Reimplements CEPSDBG in Python."""

    gamma = abs(gamma)
    if gamma > _MAX_GAMMA:
        gamma = _MAX_GAMMA

    x = fghz / fres
    x2 = x * x
    u = 1.0 / (1.0 + x2)
    v = x * u

    if gamma != 0.0:
        dv = 10.0 ** gamma
        xp = x * dv
        xm = x / dv
        up = 1.0 / (1.0 + xp * xp)
        um = 1.0 / (1.0 + xm * xm)
        vp = xp * up
        vm = xm * um
        u = (u + up + um) / 3.0
        v = (v + vp + vm) / 3.0

    er = epsv + deps * u
    dei = sige / (fghz * _A0)
    ei = -deps * v - dei
    return complex(er, ei)


def fit_debye(
    data: RopData,
    controls: DebyeControls,
    max_iterations: int = 5000,
) -> DebyeFitResult:
    """Python translation of CEPS2DBX."""

    n = data.count
    if n == 0:
        raise ValueError("Material file does not contain any samples")

    epsr = data.eps_real
    epsi_measured = data.eps_imag
    epsim = [-val for val in epsi_measured]
    measured = [complex(er, ei) for er, ei in zip(epsr, epsi_measured)]

    idx_epsim_max = max(range(n), key=lambda idx: epsim[idx])
    idx_epsr_max = max(range(n), key=lambda idx: epsr[idx])
    idx_epsr_min = min(range(n), key=lambda idx: epsr[idx])

    fres = abs(controls.fres) if controls.fres != 0.0 else data.frequencies_ghz[idx_epsim_max]
    deps = abs(controls.deps) if controls.deps != 0.0 else epsr[idx_epsr_max] - epsr[idx_epsr_min]
    epsv = abs(controls.epsv) if controls.epsv != 0.0 else epsr[-1]
    gamma = abs(controls.gamma) if controls.gamma != 0.0 else 0.3
    sige = abs(controls.sige) if controls.sige != 0.0 else 1e-10

    freqs = data.frequencies_ghz

    def objective(f_val: float, d_val: float, e_val: float, g_val: float, s_val: float) -> float:
        total = 0.0
        for freq, meas in zip(freqs, measured):
            ceps = compute_debye_ceps(freq, f_val, d_val, e_val, g_val, s_val)
            diff = meas - ceps
            total += diff.real * diff.real + diff.imag * diff.imag
        return total

    smin = objective(fres, deps, epsv, gamma, sige)
    iflag = 0
    iteration = 0
    pf = 1e-5
    sf = 1.0

    def sweep_parameter(value: float, delta: float, setter, current_sse: float) -> float:
        if delta == 0.0:
            return current_sse
        setter(value + delta)
        plus_val = objective(fres, deps, epsv, gamma, sige)
        setter(value - delta)
        minus_val = objective(fres, deps, epsv, gamma, sige)
        setter(value)
        best = min(current_sse, plus_val, minus_val)
        if best == plus_val:
            setter(value + delta)
        elif best == minus_val:
            setter(value - delta)
        return best

    while True:
        adjust_down = False
        for _ in range(10):
            iteration += 1
            if iteration > max_iterations:
                iflag = 1
                adjust_down = False
                break

            sref = smin

            if controls.fres <= 0.0:
                df = pf * fres if fres != 0.0 else pf

                def set_fres(val: float) -> None:
                    nonlocal fres
                    fres = val

                smin = sweep_parameter(fres, df, set_fres, smin)

            if controls.deps <= 0.0:
                dd = max(pf, pf * abs(deps))

                def set_deps(val: float) -> None:
                    nonlocal deps
                    deps = val

                smin = sweep_parameter(deps, dd, set_deps, smin)

            if controls.epsv <= 0.0:
                de = max(pf, pf * abs(epsv))

                def set_epsv(val: float) -> None:
                    nonlocal epsv
                    epsv = val

                smin = sweep_parameter(epsv, de, set_epsv, smin)

            if controls.gamma <= 0.0:
                dg = max(pf, pf * abs(gamma))

                def set_gamma(val: float) -> None:
                    nonlocal gamma
                    gamma = abs(val)

                smin = sweep_parameter(gamma, dg, set_gamma, smin)
                gamma = abs(gamma)

            if controls.sige <= 0.0:
                ds = max(pf, pf * abs(sige))

                def set_sige(val: float) -> None:
                    nonlocal sige
                    sige = max(0.0, val)

                smin = sweep_parameter(sige, ds, set_sige, smin)

            if smin >= sref:
                adjust_down = True
                break

        if iflag == 1:
            break

        if adjust_down:
            if sf < 512.1:
                sf *= 2.0
                pf /= 2.0
                continue
            break
        else:
            sf /= 2.0
            pf *= 2.0
            continue

    if sige < 1e-8:
        sige = 0.0
    if abs(fres) > 999.9999:
        fres = 999.9999

    fitted = [
        compute_debye_ceps(freq, fres, deps, epsv, gamma, sige)
        for freq in data.frequencies_ghz
    ]
    epsrd = [val.real for val in fitted]
    epsid = [-val.imag for val in fitted]
    residual_epsr = [er_m - er_d for er_m, er_d in zip(epsr, epsrd)]
    residual_epsi = [ei_m - ei_d for ei_m, ei_d in zip(epsim, epsid)]

    log_freqs = _log_frequency_grid()
    log_ceps = [
        compute_debye_ceps(freq, fres, deps, epsv, gamma, sige)
        for freq in log_freqs
    ]

    rms = sqrt(smin / n)

    return DebyeFitResult(
        fres=fres,
        deps=deps,
        epsv=epsv,
        gamma=gamma,
        sige=sige,
        rms=rms,
        sse=smin,
        iflag=iflag,
        iterations=iteration,
        fitted=fitted,
        residual_epsr=residual_epsr,
        residual_epsi=residual_epsi,
        log_frequencies=log_freqs,
        log_ceps=log_ceps,
    )


def generate_dbe_report(
    data: RopData,
    controls: DebyeControls,
    result: DebyeFitResult,
    *,
    input_file: Path | str,
    output_file: Path | str,
    timestamp: datetime | None = None,
) -> str:
    """Create a text report that mirrors the .DBE output format."""

    timestamp = timestamp or datetime.now()
    date_text = timestamp.strftime("%d-%b-%Y")
    time_text = timestamp.strftime("%H:%M:%S.%f")[:11]

    lines: List[str] = []
    lines.append(f"DATE: {date_text};  TIME: {time_text}")
    lines.append("")
    lines.append(f"DEBYE LSF FILE:  {Path(output_file)}")
    lines.append(f"INPUT DATA FILE: {Path(input_file)}")
    if data.sample_id:
        lines.append(f"SAMPLE ID:       {data.sample_id}")
    lines.append("")
    start_freq = data.frequencies_ghz[0]
    stop_freq = data.frequencies_ghz[-1]
    lines.append(
        f"FGHZ START, STOP, AND NUMBER: {start_freq:9.3f};"
        f"{stop_freq:9.3f};{data.count:7d}"
    )
    lines.append("")
    lines.extend(
        [
            'PARAMETER CONTROL;  IF PARAMETER (WITH "0" SUFFIX) IS:',
            '= 0 - LET THE COMPUTER ESTIMATE THE LEAST SQUARE FIT.  ',
            '> 0 - KEEP THE PARAMETER FIXED AT THAT VALUE.',
            '      TO FIX PARAMETER TO ZERO, SET IT EQUAL TO 1.E-8.',
            '< 0 - USE THE NEGATIVE OF THAT VALUE FOR INITIAL GUESS.',
            "",
            "       FRES0       DEPS0       EPSV0       GAMM0       SIGE0",
            (
                f"{controls.fres:11.4f}{controls.deps:11.4f}"
                f"{controls.epsv:11.4f}{controls.gamma:11.4f}{controls.sige:11.4f}"
            ),
            "",
            "LEAST SQUARE FIT TO DEBYE PARAMETERS:",
            "      FRES      DEPS      EPSV     GAMMA      SIGE       RMS",
            (
                f"{result.fres:10.4f}{result.deps:10.4f}{result.epsv:10.4f}"
                f"{result.gamma:10.4f}{result.sige:10.5f}{result.rms:10.4f}"
            ),
            "",
            "COMPARISON OF MEASURED (_M) AND LSF DEBYE (_D) CEPS VS FREQUENCY:",
            "   #      FGHZ      EPSR_M    EPSR_D    DEL_ER      EPSI_M    EPSI_D    DEL_EI",
        ]
    )

    epsr = data.eps_real
    epsi = data.eps_imag
    epsrd = [val.real for val in result.fitted]
    epsid_positive = [-val.imag for val in result.fitted]
    epsi_fit = [-val for val in epsid_positive]

    for idx in range(data.count):
        freq = data.frequencies_ghz[idx]
        er_m = epsr[idx]
        er_d = epsrd[idx]
        dr = result.residual_epsr[idx]
        ei_m = epsi[idx]
        ei_d = epsi_fit[idx]
        di = result.residual_epsi[idx]
        lines.append(
            f"{idx + 1:4d}  {freq:9.3f}  {er_m:8.3f}  {er_d:8.3f}  {dr:8.3f}"
            f"    {ei_m:8.3f}  {ei_d:8.3f}  {di:8.3f}"
        )

    lines.append("")
    lines.append("LSF DEBYE CEPS VS FREQUENCY FROM .01 TO 1000 GHZ:")
    lines.append("   #      FGHZ      EPSR      EPSI")
    for idx, (freq, ceps) in enumerate(zip(result.log_frequencies, result.log_ceps), start=1):
        lines.append(f"{idx:4d}  {freq:10.4f}  {ceps.real:10.4f}  {ceps.imag:10.4f}")

    return "\n".join(lines).rstrip() + "\n"


def _log_frequency_grid() -> List[float]:
    values = [0.0] * _LOG_POINTS
    anchors = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    starts = [0, 120, 240, 360, 480, 600]
    for idx, value in zip(starts, anchors):
        values[idx] = value
    dv = 10.0 ** (1.0 / _LOG_BLOCK)
    for offset in range(_LOG_BLOCK - 1):
        values[1 + offset] = values[offset] * dv
        values[121 + offset] = values[120 + offset] * dv
        values[241 + offset] = values[240 + offset] * dv
        values[361 + offset] = values[360 + offset] * dv
        values[481 + offset] = values[480 + offset] * dv
    return values
