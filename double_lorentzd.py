"""Double Lorentz (Debye + Lorentz) solver translated from CE2DL2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isclose, sqrt
from pathlib import Path
from typing import List, Literal, Tuple

from .debye import _log_frequency_grid
from .rop import RopData

QuantityKind = Literal["eps", "mu"]

_A0 = 5.563249e-2
_MIN_FREQ = 1e-4
_MAX_FREQ = 999.9999
_MIN_DEPS = 1e-4
_MAX_DEPS = 999.9999
_MAX_GAMMA = 10.0
_MIN_LOR_GAMMA = 0.001
_MAX_LOR_GAMMA = 100.0
_SENTINEL_ZERO = 1e-8


@dataclass
class DoubleLorentzControls:
    """Control values that mirror the FORTRAN parameter semantics."""

    fres1: float = 0.0
    deps1: float = 0.0
    gamma1: float = 0.0
    fres2: float = 0.0
    deps2: float = 0.0
    gamma2: float = 0.0
    epsv: float = 0.0
    sige: float = 0.0


@dataclass
class DoubleLorentzResult:
    """Holds the outcome of the translated CE2DL2 optimization."""

    fres1: float
    deps1: float
    gamma1: float
    fres2: float
    deps2: float
    gamma2: float
    epsv: float
    sige: float
    rms: float
    sse: float
    iterations: int
    iflag: int
    fitted: List[complex]
    residual_real: List[float]
    residual_imag: List[float]
    log_frequencies: List[float]
    log_values: List[complex]

    @property
    def converged(self) -> bool:
        return self.iflag == 0


def fit_double_lorentz(
    data: RopData,
    quantity: QuantityKind,
    controls: DoubleLorentzControls,
    max_iterations: int = 5000,
) -> DoubleLorentzResult:
    """Python translation of the CE2DL2 least-squares solver."""

    real_vals, imag_vals = _select_quantity_arrays(data, quantity)
    count = len(real_vals)
    if count == 0:
        raise ValueError("Material file does not contain any samples")

    freqs = data.frequencies_ghz
    neg_imag = [-val for val in imag_vals]
    idx_epsix_max = max(range(count), key=lambda idx: neg_imag[idx])
    idx_real_max = max(range(count), key=lambda idx: real_vals[idx])
    idx_real_min = min(range(count), key=lambda idx: real_vals[idx])

    fres1, fres2, deps1, deps2, gamma1, gamma2 = _initial_parameters(
        freqs,
        real_vals,
        controls,
        idx_epsix_max,
        idx_real_max,
        idx_real_min,
    )

    if controls.epsv == 0.0:
        epsv = real_vals[-1]
    else:
        epsv = abs(controls.epsv)

    if controls.sige == 0.0:
        sige = 1e-8
    else:
        sige = abs(controls.sige)

    measured = [complex(er, ei) for er, ei in zip(real_vals, imag_vals)]

    def objective() -> float:
        total = 0.0
        for freq, meas in zip(freqs, measured):
            val = compute_debye_lorentz_ceps(
                freq,
                fres1,
                deps1,
                gamma1,
                fres2,
                deps2,
                gamma2,
                epsv,
                sige,
            )
            diff = meas - val
            total += diff.real * diff.real + diff.imag * diff.imag
        return total

    smin = objective()
    sf = 1.0
    pf = 1e-2
    iflag = 0
    iteration = 0

    def _clamp_freq(value: float) -> float:
        return min(max(value, _MIN_FREQ), _MAX_FREQ)

    def _clamp_deps(value: float) -> float:
        return min(max(value, _MIN_DEPS), _MAX_DEPS)

    def _clamp_gamma(value: float) -> float:
        value = abs(value)
        if value > _MAX_GAMMA:
            value = _MAX_GAMMA
        if value < 1e-4:
            value = 0.0
        return value

    def _clamp_gamma2(value: float) -> float:
        value = abs(value)
        if value < _MIN_LOR_GAMMA:
            value = 0.0
        if value > _MAX_LOR_GAMMA:
            value = _MAX_LOR_GAMMA
        return value

    def _clamp_epsv(value: float) -> float:
        return min(max(value, 0.0), _MAX_DEPS)

    def _clamp_sige(value: float) -> float:
        value = max(value, 0.0)
        if value > 100.0:
            value = 100.0
        return value

    def sweep(value_getter, value_setter, delta: float, clamp) -> float:
        nonlocal smin
        current = smin
        best_value = value_getter()
        if delta != 0.0:
            orig = value_getter()
            value_setter(orig + delta)
            plus_val = objective()
            if plus_val < current:
                current = plus_val
                best_value = orig + delta
            value_setter(orig - delta)
            minus_val = objective()
            if minus_val < current:
                current = minus_val
                best_value = orig - delta
            value_setter(orig)
        value_setter(clamp(best_value))
        smin = objective()
        return smin

    while True:
        adjust_down = False
        for _ in range(10):
            iteration += 1
            if iteration > max_iterations:
                iflag = 1
                break

            sref = smin

            if controls.fres1 <= 0.0:
                df = pf * fres1 if fres1 != 0.0 else pf

                def set_fres1(val: float) -> None:
                    nonlocal fres1
                    fres1 = val

                sweep(lambda: fres1, set_fres1, df, _clamp_freq)

            if controls.deps1 <= 0.0:
                dd = max(pf, pf * abs(deps1))

                def set_deps1(val: float) -> None:
                    nonlocal deps1
                    deps1 = val

                sweep(lambda: deps1, set_deps1, dd, _clamp_deps)

            if controls.gamma1 <= 0.0:
                dg = max(pf, pf * abs(gamma1))

                def set_gamma1(val: float) -> None:
                    nonlocal gamma1
                    gamma1 = val

                sweep(lambda: gamma1, set_gamma1, dg, _clamp_gamma)

            if controls.fres2 <= 0.0:
                df2 = pf * fres2 if fres2 != 0.0 else pf

                def set_fres2(val: float) -> None:
                    nonlocal fres2
                    fres2 = val

                sweep(lambda: fres2, set_fres2, df2, _clamp_freq)

            if controls.deps2 <= 0.0:
                dd2 = max(pf, pf * abs(deps2))

                def set_deps2(val: float) -> None:
                    nonlocal deps2
                    deps2 = val

                sweep(lambda: deps2, set_deps2, dd2, _clamp_deps)

            if controls.gamma2 <= 0.0:
                dg2 = max(pf, pf * abs(gamma2))

                def set_gamma2(val: float) -> None:
                    nonlocal gamma2
                    gamma2 = val

                sweep(lambda: gamma2, set_gamma2, dg2, _clamp_gamma2)

            if controls.epsv <= 0.0:
                de = max(pf, pf * abs(epsv))

                def set_epsv(val: float) -> None:
                    nonlocal epsv
                    epsv = val

                sweep(lambda: epsv, set_epsv, de, _clamp_epsv)

            if controls.sige <= 0.0:
                ds = max(pf / 10.0, pf * max(sige, 1e-8))

                def set_sige(val: float) -> None:
                    nonlocal sige
                    sige = val

                sweep(lambda: sige, set_sige, ds, _clamp_sige)

            if smin >= sref:
                adjust_down = True
                break

        if iflag == 1:
            break

        if adjust_down:
            if sf < 1024.1:
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

    fitted = [
        compute_debye_lorentz_ceps(
            freq,
            fres1,
            deps1,
            gamma1,
            fres2,
            deps2,
            gamma2,
            epsv,
            sige,
        )
        for freq in freqs
    ]

    residual_real = [real_vals[i] - fitted[i].real for i in range(count)]
    residual_imag = [imag_vals[i] - fitted[i].imag for i in range(count)]

    log_freqs = _log_frequency_grid()
    log_values = [
        compute_debye_lorentz_ceps(
            freq,
            fres1,
            deps1,
            gamma1,
            fres2,
            deps2,
            gamma2,
            epsv,
            sige,
        )
        for freq in log_freqs
    ]

    rms = sqrt(smin / count)

    return DoubleLorentzResult(
        fres1=fres1,
        deps1=deps1,
        gamma1=gamma1,
        fres2=fres2,
        deps2=deps2,
        gamma2=gamma2,
        epsv=epsv,
        sige=sige,
        rms=rms,
        sse=smin,
        iterations=iteration,
        iflag=iflag,
        fitted=fitted,
        residual_real=residual_real,
        residual_imag=residual_imag,
        log_frequencies=log_freqs,
        log_values=log_values,
    )


def compute_debye_lorentz_ceps(
    fghz: float,
    fres_debye: float,
    deps_debye: float,
    gamma_debye: float,
    fres_lorentz: float,
    deps_lorentz: float,
    gamma_lorentz: float,
    epsv: float,
    sige: float,
) -> complex:
    """Reimplements CEPSDL2 in Python."""

    freq = max(fghz, 1e-6)
    fres_debye = max(abs(fres_debye), 1e-4)
    fres_lorentz = max(abs(fres_lorentz), 1e-4)
    gamma_d = min(abs(gamma_debye), _MAX_GAMMA)
    gamma_l = abs(gamma_lorentz)
    if gamma_l > _MAX_LOR_GAMMA:
        gamma_l = _MAX_LOR_GAMMA
    if gamma_l < _MIN_LOR_GAMMA:
        gamma_l = _MIN_LOR_GAMMA

    base = complex(epsv, -sige / (freq * _A0))

    x1 = freq / fres_debye
    if gamma_d == 0.0:
        c1 = deps_debye / (1.0 + 1j * x1)
    else:
        dv = 10.0 ** gamma_d
        xm = x1 / dv
        xp = x1 * dv
        c1 = deps_debye * (
            1.0 / (1.0 + 1j * xm)
            + 1.0 / (1.0 + 1j * x1)
            + 1.0 / (1.0 + 1j * xp)
        ) / 3.0

    x2 = freq / fres_lorentz
    c2 = deps_lorentz / (1.0 - x2 * (x2 - 1j * gamma_l))

    return base + c1 + c2


def generate_dlm_report(
    data: RopData,
    quantity: QuantityKind,
    controls: DoubleLorentzControls,
    result: DoubleLorentzResult,
    *,
    input_file: Path | str,
    output_file: Path | str,
    timestamp: datetime | None = None,
) -> str:
    """Emit a text report equivalent to the FORTRAN .DLE/.DLM output."""

    timestamp = timestamp or datetime.now()
    date_text = timestamp.strftime("%d-%b-%Y")
    time_text = timestamp.strftime("%H:%M:%S.%f")[:11]

    real_vals, imag_vals = _select_quantity_arrays(data, quantity)
    heading = "CEPS" if quantity == "eps" else "CMU"
    lines: List[str] = []
    lines.append(f"DATE: {date_text};  TIME: {time_text}")
    lines.append("")
    lines.append(f"DEBYE-LORENTZIAN LSF FILE:  {Path(output_file)}")
    lines.append(f"INPUT DATA FILE: {Path(input_file)}")
    if data.sample_id:
        lines.append(f"SAMPLE ID:       {data.sample_id}")
    lines.append("")
    lines.append(
        f"FGHZ START, STOP, AND NUMBER: {data.frequencies_ghz[0]:9.3f};"
        f"{data.frequencies_ghz[-1]:9.3f};{len(real_vals):7d}"
    )
    lines.append("")

    if quantity == "eps":
        control_header = (
            "    FRES0D    DEPS0D    GAMM0D    FRES0L    DEPS0L    GAMM0L"
            "     EPSV0     SIGE0"
        )
        fit_header = (
            "     FRESD     DEPSD     GAMMD     FRESL     DEPSL     GAMML"
            "      EPSV      SIGE       RMS"
        )
        control_values = (
            f"{controls.fres1:11.4f}{controls.deps1:11.4f}{controls.gamma1:11.4f}"
            f"{controls.fres2:11.4f}{controls.deps2:11.4f}{controls.gamma2:11.4f}"
            f"{controls.epsv:11.4f}{controls.sige:11.4f}"
        )
        fit_values = (
            f"{result.fres1:11.4f}{result.deps1:11.4f}{result.gamma1:11.4f}"
            f"{result.fres2:11.4f}{result.deps2:11.4f}{result.gamma2:11.4f}"
            f"{result.epsv:11.4f}{result.sige:11.5f}{result.rms:10.4f}"
        )
        table_header = (
            "   #      FGHZ      EPSR_M    EPSR_DL    DEL_ER"
            "      EPSI_M    EPSI_DL    DEL_EI"
        )
        log_header = "LSF DEBYE-LORENTZIAN CEPS FROM .01 TO 1000 GHZ:"
        item_labels = ("EPSR", "EPSI")
    else:
        control_header = (
            "    FR_M0D    DMUR0D    GAMM0D    FR_M0L    DMUR0L    GAMM0L"
            "     MURV0     SIGM0"
        )
        fit_header = (
            "     FR_M1     DMUR1     GAMM1     FR_M2     DMUR2     GAMM2"
            "      MURV      SIGM       RMS"
        )
        control_values = (
            f"{controls.fres1:11.4f}{controls.deps1:11.4f}{controls.gamma1:11.4f}"
            f"{controls.fres2:11.4f}{controls.deps2:11.4f}{controls.gamma2:11.4f}"
            f"{controls.epsv:11.4f}{controls.sige:11.4f}"
        )
        fit_values = (
            f"{result.fres1:11.4f}{result.deps1:11.4f}{result.gamma1:11.4f}"
            f"{result.fres2:11.4f}{result.deps2:11.4f}{result.gamma2:11.4f}"
            f"{result.epsv:11.4f}{result.sige:11.5f}{result.rms:10.4f}"
        )
        table_header = (
            "   #      FGHZ       MUR_M     MUR_DL    DEL_MUR"
            "      MUI_M     MUI_DL    DEL_MUI"
        )
        log_header = "LSF DEBYE-LORENTZIAN CMU FROM .01 TO 1000 GHZ:"
        item_labels = ("MUR", "MUI")

    lines.append('PARAMETER CONTROL;  IF PARAMETER (WITH "0" SUFFIX) IS:')
    lines.append('= 0 - LET THE COMPUTER ESTIMATE THE LEAST SQUARE FIT.  ')
    lines.append('> 0 - KEEP THE PARAMETER FIXED AT THAT VALUE.')
    lines.append('      TO FIX PARAMETER TO ZERO, SET IT EQUAL TO 1.E-8.')
    lines.append('< 0 - USE THE NEGATIVE OF THAT VALUE FOR INITIAL GUESS.')
    lines.append("")
    lines.append(control_header)
    lines.append(control_values)
    lines.append("")
    lines.append('LEAST SQUARE FIT TO DEBYE-LORENTZIAN PARAMETERS:')
    lines.append(fit_header)
    lines.append(fit_values)
    lines.append("")
    lines.append(
        f'COMPARISON OF MEASURED (_M) AND LSF DEBYE-LORENTZIAN (_DL) {heading} VS FREQUENCY:'
    )
    lines.append(table_header)

    for idx, (freq, measured_r, fitted_complex, d_r, measured_i, fitted_i, d_i) in enumerate(
        zip(
            data.frequencies_ghz,
            real_vals,
            result.fitted,
            result.residual_real,
            imag_vals,
            (val.imag for val in result.fitted),
            result.residual_imag,
        ),
        start=1,
    ):
        lines.append(
            f"{idx:4d}  {freq:9.3f}  {measured_r:8.3f}  {fitted_complex.real:8.3f}"
            f"  {d_r:8.3f}    {measured_i:8.3f}  {fitted_i:8.3f}  {d_i:8.3f}"
        )

    lines.append("")
    lines.append(log_header)
    lines.append("   #      FGHZ      {0}      {1}".format(*item_labels))
    for idx, (freq, value) in enumerate(zip(result.log_frequencies, result.log_values), start=1):
        lines.append(f"{idx:4d}  {freq:10.4f}  {value.real:10.4f}  {value.imag:10.4f}")

    return "\n".join(lines).rstrip() + "\n"


def _select_quantity_arrays(data: RopData, quantity: QuantityKind) -> Tuple[List[float], List[float]]:
    if quantity == "eps":
        return data.eps_real, data.eps_imag
    if quantity == "mu":
        return data.mu_real, data.mu_imag
    raise ValueError(f"Unsupported quantity kind: {quantity}")


def _is_sentinel_zero(value: float) -> bool:
    return isclose(value, _SENTINEL_ZERO, rel_tol=0.0, abs_tol=1e-12)


def _initial_parameters(
    freqs: List[float],
    real_vals: List[float],
    controls: DoubleLorentzControls,
    idx_epsix_max: int,
    idx_real_max: int,
    idx_real_min: int,
) -> Tuple[float, float, float, float, float, float]:
    freq_peak = freqs[idx_epsix_max]

    if _is_sentinel_zero(controls.deps2):
        fres1 = abs(controls.fres1) if controls.fres1 != 0.0 else freq_peak
        deps1 = abs(controls.deps1) if controls.deps1 != 0.0 else real_vals[idx_real_max] - real_vals[idx_real_min]
        gamma1 = abs(controls.gamma1) if controls.gamma1 != 0.0 else 0.3
        fres2 = freq_peak
        deps2 = _SENTINEL_ZERO
        gamma2 = 0.3
        return fres1, fres2, deps1, deps2, gamma1, gamma2

    if _is_sentinel_zero(controls.deps1):
        fres2 = abs(controls.fres2) if controls.fres2 != 0.0 else freq_peak
        deps2 = abs(controls.deps2) if controls.deps2 != 0.0 else real_vals[idx_real_max] - real_vals[idx_real_min]
        gamma2 = abs(controls.gamma2) if controls.gamma2 != 0.0 else 0.3
        fres1 = freq_peak
        deps1 = _SENTINEL_ZERO
        gamma1 = 0.3
        return fres1, fres2, deps1, deps2, gamma1, gamma2

    span = (freqs[-1] - freqs[0]) / 3.0
    fres1 = abs(controls.fres1) if controls.fres1 != 0.0 else span
    fres2 = abs(controls.fres2) if controls.fres2 != 0.0 else 2 * span

    dv = (real_vals[idx_real_max] - real_vals[idx_real_min]) / 2.0
    deps1 = abs(controls.deps1) if controls.deps1 != 0.0 else dv
    deps2 = abs(controls.deps2) if controls.deps2 != 0.0 else dv

    gamma1 = abs(controls.gamma1) if controls.gamma1 != 0.0 else 0.3
    gamma2 = abs(controls.gamma2) if controls.gamma2 != 0.0 else 0.3

    return fres1, fres2, deps1, deps2, gamma1, gamma2
