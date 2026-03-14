"""Manual Figure 1 handoff helper.

Figure 1 is intentionally not auto-generated. The paper requires a cleaner system
architecture diagram than the code can reliably produce, so the final artwork is
expected to be supplied manually by the author.
"""

from __future__ import annotations

from pathlib import Path


def resolve_manual_figure1_path(figure_dir: Path) -> Path:
    """Return the expected manual Figure 1 path without generating any artwork."""

    return figure_dir / "figure1_system_architecture_manual.png"
