"""Parsers for material measurement files (.ROP/.EMU)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RopData:
    """Container for the numeric content of a ROP/EMU material file."""

    path: Path
    sample_id: str
    density_lb_ft3: float
    frequencies_ghz: List[float]
    mu_real: List[float]
    mu_imag: List[float]
    eps_real: List[float]
    eps_imag: List[float]

    def __post_init__(self) -> None:
        length = len(self.frequencies_ghz)
        for field in (self.mu_real, self.mu_imag, self.eps_real, self.eps_imag):
            if len(field) != length:
                raise ValueError("All data arrays must have the same length")

    @property
    def count(self) -> int:
        return len(self.frequencies_ghz)


def read_material_file(path: Path | str) -> RopData:
    """Parse a .ROP/.EMU text file produced by the legacy tools."""

    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    sample_id = ""
    nf: int | None = None
    density: float | None = None
    cursor = 0

    while cursor < len(lines):
        raw = lines[cursor].strip()
        cursor += 1
        if not raw:
            continue
        if raw.startswith("!"):
            upper = raw.upper()
            if upper.startswith("! SAMPLE ID:"):
                sample_id = raw.split(":", 1)[1].strip()
            continue
        numbers = _parse_numeric_tokens(raw)
        if len(numbers) >= 2:
            nf = int(float(numbers[0]))
            density = float(numbers[1])
            break

    if nf is None or density is None:
        raise ValueError(f"{file_path} does not include the NF/DENS line")

    frequencies: List[float] = []
    mu_real: List[float] = []
    mu_imag: List[float] = []
    eps_real: List[float] = []
    eps_imag: List[float] = []

    while len(frequencies) < nf and cursor < len(lines):
        raw = lines[cursor]
        cursor += 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("!"):
            continue
        numbers = _parse_numeric_tokens(raw)
        if len(numbers) < 5:
            raise ValueError(
                f"Line {cursor} of {file_path} does not contain at least five numbers"
            )
        freq, mur, mui, epsr, epsi = numbers[:5]
        frequencies.append(freq)
        mu_real.append(mur)
        mu_imag.append(mui)
        eps_real.append(epsr)
        eps_imag.append(epsi)

    if len(frequencies) != nf:
        raise ValueError(
            f"Expected {nf} rows of data in {file_path}, found {len(frequencies)}"
        )

    if not sample_id:
        sample_id = file_path.stem

    return RopData(
        path=file_path,
        sample_id=sample_id,
        density_lb_ft3=density,
        frequencies_ghz=frequencies,
        mu_real=mu_real,
        mu_imag=mu_imag,
        eps_real=eps_real,
        eps_imag=eps_imag,
    )


def _parse_numeric_tokens(line: str) -> List[float]:
    """Return floats from the portion of a line that precedes an inline comment."""

    data = line.split("!", 1)[0].strip()
    if not data:
        return []
    numbers: List[float] = []
    for token in data.split():
        numbers.append(float(token))
    return numbers
