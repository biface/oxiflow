#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
# =============================================================================
# abaques.py — dispatch abaques from the oxiflow_parallel benchmark results
# =============================================================================
# Reads the results Criterion saved under target/criterion/ for the
# oxiflow_parallel benchmark and writes, for the wiki, charts and tables
# that help size the sequential/parallel dispatch thresholds (#214, DA-048).
# Criterion's own charts plot one group at a time and never a ratio between
# two measurements; the abaques plot that ratio, on logarithmic axes, with a
# line at 1 between gain and loss.
#
#   diagnostics     one chart and one table per group with sequential and
#                   parallel cases: parallel/sequential time against size
#   per call        every per-call diagnostic on one chart: the ratio
#                   against the sequential time of the call, which reads as
#                   a volume of operations whatever the model
#   orchestrator    per scenario and integrator: the inner dispatch modes
#                   against inner_seq, by nodes per domain, under the
#                   sequential and the parallel orchestrator; and the
#                   orchestrator's parallel/sequential time by inner mode
#
# Times are Criterion's mean estimates. A ratio is marked `~` when the
# confidence intervals of its two times overlap. The figures belong to the
# host and the run that produced them: state both alongside the abaques.
#
# Usage:
#   python3 scripts/abaques.py [--source DIR] [--export DIR] [--venv DIR]
#
#   --source DIR    Criterion results to read (default: target/criterion)
#   --export DIR    where the charts and abaques.md go (default:
#                   target/abaques)
#   --venv DIR      the script's own virtual environment (default:
#                   ~/.cache/oxiflow/abaques-venv, under $XDG_CACHE_HOME
#                   when set)
#
# Environment: the script needs matplotlib and keeps it out of the project
# and of the system Python. When matplotlib is importable (an active
# environment, or `uv run scripts/abaques.py`, which reads the inline
# metadata above), it runs as is. Otherwise it creates its virtual
# environment on first use, installs matplotlib there and runs again with
# that interpreter; later runs reuse it. Creating the environment needs the
# venv module (the python3-venv package on Debian and Ubuntu).
# =============================================================================

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import venv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes

REQUIREMENTS = ["matplotlib>=3.8"]

# Set by load_matplotlib(), once the environment provides it.
plt: Any = None
FixedLocator: Any = None
FuncFormatter: Any = None
LogLocator: Any = None
NullFormatter: Any = None

ORCHESTRATOR_PREFIX = "orchestrator/"
SEQUENTIAL, PARALLEL = "sequential", "parallel"

# Diagnostics whose timed call is a whole Backward Euler step (Jacobian
# assembly plus linear solve), not one call of the dispatched loop: kept out
# of the per-call chart, whose abscissa is the duration of that call.
STEP_GROUPS = {
    "diffusion_1d_implicit_jacobian_dispatch",
    "langmuir_multi_implicit_jacobian_dispatch",
}

INNER_REFERENCE = "inner_seq"
INNER_COMPARED = ("inner_default", "inner_par")
INNER_ORDER = (INNER_REFERENCE,) + INNER_COMPARED
INNER_STYLE = {"inner_seq": ":", "inner_default": "-", "inner_par": "--"}


@dataclass(frozen=True)
class Case:
    """One benchmark case: Criterion's ids and the mean time, in ns."""

    group: str
    function: str
    value: float
    mean: float
    low: float
    high: float


def load_cases(criterion: Path) -> list[Case]:
    """Every case of `criterion` with a numeric parameter."""
    cases = []
    for bench in sorted(criterion.glob("**/new/benchmark.json")):
        meta = json.loads(bench.read_text())
        estimates = json.loads((bench.parent / "estimates.json").read_text())
        try:
            value = float(meta["value_str"])
        except (TypeError, ValueError):
            continue
        mean = estimates["mean"]
        cases.append(
            Case(
                group=meta["group_id"],
                function=meta["function_id"] or "",
                value=value,
                mean=mean["point_estimate"],
                low=mean["confidence_interval"]["lower_bound"],
                high=mean["confidence_interval"]["upper_bound"],
            )
        )
    return cases


def fmt_time(ns: float) -> str:
    """A time in ns with the unit that keeps three significant digits."""
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.3g} {unit}"
    return f"{ns:.3g} ns"


def fmt_ratio(num: Case, den: Case) -> str:
    """`num / den`, marked `~` when the two confidence intervals overlap."""
    overlap = num.low <= den.high and den.low <= num.high
    return f"{'~' if overlap else ''}{num.mean / den.mean:.2f}"


def fmt_size(value: float) -> str:
    return f"{value:,.0f}"


def slug(name: str) -> str:
    return name.replace("/", "_")


# Smallest distance between two labels on the ratio axis, as a fraction of
# the axis span in decades: on the charts' heights, about one and a half
# times the height of a label.
LABEL_GAP = 0.06


def label_priority(value: float) -> int:
    """Rank of a tick in the choice of labels: 5, 2 and 3 times a power of
    ten first, then the other whole multiples, then halves, then the rest."""
    scaled = round(value / 10 ** math.floor(math.log10(value) + 1e-9), 6)
    ranks = {5.0: 0, 2.0: 1, 3.0: 2}
    if scaled in ranks:
        return ranks[scaled]
    if scaled.is_integer():
        return 3
    return 4 if (2 * scaled).is_integer() else 5


def spaced_minor_labels(ax: Axes) -> Any:
    """Formatter for the minor ticks of the ratio axis. Ticks are taken by
    `label_priority`, and each one is labelled only if it lies at least
    `LABEL_GAP` away from the major ticks and from every tick already
    labelled, so that labels never overlap, whatever the span of the axis."""
    formatter = FuncFormatter(lambda value, _: f"{value:g}")

    def format_ticks(values: list[float]) -> list[str]:
        low, high = ax.get_ylim()
        gap = LABEL_GAP * math.log10(high / low)
        taken = [
            math.log10(m) for m in ax.yaxis.get_majorticklocs() if low <= m <= high
        ]
        shown = set()
        visible = [v for v in values if low <= v <= high]
        for value in sorted(visible, key=label_priority):
            position = math.log10(value)
            if all(abs(position - t) >= gap for t in taken):
                taken.append(position)
                shown.add(value)
        return [f"{v:g}" if v in shown else "" for v in values]

    formatter.format_ticks = format_ticks
    return formatter


def ratio_axes(
    ax: Axes, xlabel: str, ylabel: str, xticks: list[float] | None = None
) -> None:
    """Logarithmic axes, the line at 1, labels and grid.

    On the ratio axis, a minor tick gets a label only where it leaves room
    for its neighbours (`spaced_minor_labels`); an axis spanning less than
    a decade also gets ticks at 1.5 times a power of ten. On the size axis,
    the powers of ten carry the labels, or, given `xticks`, the measured
    values themselves, for axes too narrow to show more than one power of
    ten.
    """
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linewidth=0.3)
    low, high = ax.get_ylim()
    if math.log10(high / low) < 1 and all(
        label_priority(v) <= 3 for v in ax.yaxis.get_minorticklocs()
    ):
        # Under a decade, whole multiples alone can leave a ratio such as
        # 1.66 with no tick between 1 and the top of the axis.
        ax.yaxis.set_minor_locator(LogLocator(subs=(1.5, 2, 3, 4, 5, 6, 7, 8, 9)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(spaced_minor_labels(ax))
    if xticks:
        ax.xaxis.set_major_locator(FixedLocator(sorted(set(xticks))))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: fmt_size(value)))
    ax.xaxis.set_minor_formatter(NullFormatter())


# ── Diagnostics ─────────────────────────────────────────────────────────────


def diagnostic_pairs(
    cases: list[Case],
) -> dict[str, dict[str, list[tuple[Case, Case]]]]:
    """group → variant → (sequential, parallel) pairs, sorted by size.

    The variant is the function id without its last component (`dense`,
    `graded`…), empty when the function id is just `sequential` or
    `parallel`. Groups without such pairs are left out.
    """
    by_key: dict[tuple[str, str, float], dict[str, Case]] = defaultdict(dict)
    for case in cases:
        if case.group.startswith(ORCHESTRATOR_PREFIX):
            continue
        prefix, _, mode = case.function.rpartition("/")
        if mode in (SEQUENTIAL, PARALLEL):
            key = (case.group, prefix, case.value)
            by_key[key][mode] = case
    pairs: dict[str, dict[str, list[tuple[Case, Case]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (group, variant, _), modes in sorted(by_key.items()):
        if SEQUENTIAL in modes and PARALLEL in modes:
            pairs[group][variant].append((modes[SEQUENTIAL], modes[PARALLEL]))
    return pairs


def write_diagnostics(
    pairs: dict[str, dict[str, list[tuple[Case, Case]]]], out: Path
) -> list[str]:
    lines = ["## Diagnostics", ""]
    for group, variants in sorted(pairs.items()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for variant, series in variants.items():
            ax.plot(
                [seq.value for seq, _ in series],
                [par.mean / seq.mean for seq, par in series],
                marker="o",
                label=variant or "parallel / sequential",
            )
        ratio_axes(ax, "size", "parallel / sequential time")
        ax.set_title(group)
        ax.legend()
        chart = f"{slug(group)}.svg"
        fig.tight_layout()
        fig.savefig(out / chart)
        plt.close(fig)

        lines += [f"### `{group}`", "", f"![{group}]({chart})", ""]
        lines += [
            "| Variant | Size | Sequential | Parallel | Parallel / sequential |",
            "|---|---:|---:|---:|---:|",
        ]
        for variant, series in variants.items():
            for seq, par in series:
                lines.append(
                    f"| {variant or '—'} | {fmt_size(seq.value)} "
                    f"| {fmt_time(seq.mean)} | {fmt_time(par.mean)} "
                    f"| {fmt_ratio(par, seq)} |"
                )
        lines.append("")
    return lines


def write_per_call(
    pairs: dict[str, dict[str, list[tuple[Case, Case]]]], out: Path
) -> list[str]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, variants in sorted(pairs.items()):
        if group in STEP_GROUPS:
            continue
        for variant, series in variants.items():
            ax.plot(
                [seq.mean / 1e3 for seq, _ in series],
                [par.mean / seq.mean for seq, par in series],
                marker="o",
                label=f"{group}/{variant}" if variant else group,
            )
    ratio_axes(ax, "sequential time of one call (µs)", "parallel / sequential time")
    ax.set_title("Per-call diagnostics: ratio against the work of one call")
    ax.legend(fontsize="x-small")
    fig.tight_layout()
    fig.savefig(out / "per_call.svg")
    plt.close(fig)
    excluded = ", ".join(f"`{g}`" for g in sorted(STEP_GROUPS))
    return [
        "## Per call",
        "",
        "The sequential time of one call measures the work the dispatch",
        "splits, whatever the model: the number of items times the cost of",
        f"one item. Left out: {excluded}, whose timed call is a whole",
        "Backward Euler step.",
        "",
        "![per call](per_call.svg)",
        "",
    ]


# ── Orchestrator ────────────────────────────────────────────────────────────


def orchestrator_cases(
    cases: list[Case],
) -> dict[tuple[str, str], dict[tuple[str, int, str, float], Case]]:
    """(scenario, integrator) → (inner, K, orchestrator mode, volume) → case."""
    found: dict[tuple[str, str], dict[tuple[str, int, str, float], Case]] = defaultdict(
        dict
    )
    for case in cases:
        if not case.group.startswith(ORCHESTRATOR_PREFIX):
            continue
        _, scenario, integrator, inner = case.group.split("/")
        domains, mode = case.function.split("/")
        key = (inner, int(domains.lstrip("K")), mode, case.value)
        found[(scenario, integrator)][key] = case
    return found


def write_orchestrator(
    found: dict[tuple[str, str], dict[tuple[str, int, str, float], Case]],
    out: Path,
) -> list[str]:
    lines = [
        "## Orchestrator",
        "",
        "Inner modes: the dispatch sites inside a domain forced sequential",
        "(`inner_seq`), left at their defaults (`inner_default`) or forced",
        "parallel (`inner_par`). Nodes per domain is the aggregate volume",
        "divided by the number of domains (a mean when the domains differ).",
        "",
    ]
    for (scenario, integrator), table in sorted(found.items()):
        name = f"{scenario}/{integrator}"
        counts = sorted({k for _, k, _, _ in table})
        volumes = sorted({v for _, _, _, v in table})
        colours = {k: f"C{i}" for i, k in enumerate(counts)}

        per_domain = sorted({v / k for _, k, _, v in table})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, mode in zip(axes, (SEQUENTIAL, PARALLEL)):
            for k in counts:
                for inner in INNER_COMPARED:
                    points = [
                        (v / k, table[(inner, k, mode, v)].mean / ref.mean)
                        for v in volumes
                        if (ref := table.get((INNER_REFERENCE, k, mode, v)))
                        and (inner, k, mode, v) in table
                    ]
                    if points:
                        ax.plot(
                            *zip(*points),
                            marker="o",
                            color=colours[k],
                            linestyle=INNER_STYLE[inner],
                            label=f"K{k} {inner}",
                        )
            ratio_axes(ax, "nodes per domain", f"time / {INNER_REFERENCE}", per_domain)
            ax.set_title(f"orchestrator {mode}")
        axes[0].legend(fontsize="x-small")
        fig.suptitle(f"{name}: inner dispatch modes")
        fig.tight_layout()
        inner_chart = f"orchestrator_{slug(name)}_inner.svg"
        fig.savefig(out / inner_chart)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k in counts:
            for inner in INNER_ORDER:
                points = [
                    (v, table[(inner, k, PARALLEL, v)].mean / seq.mean)
                    for v in volumes
                    if (seq := table.get((inner, k, SEQUENTIAL, v)))
                    and (inner, k, PARALLEL, v) in table
                ]
                if points:
                    ax.plot(
                        *zip(*points),
                        marker="o",
                        color=colours[k],
                        linestyle=INNER_STYLE[inner],
                        label=f"K{k} {inner}",
                    )
        ratio_axes(
            ax, "aggregate mesh volume (nodes)", "parallel / sequential time", volumes
        )
        ax.set_title(f"{name}: orchestrator dispatch")
        ax.legend(fontsize="x-small")
        fig.tight_layout()
        orchestrator_chart = f"orchestrator_{slug(name)}_dispatch.svg"
        fig.savefig(out / orchestrator_chart)
        plt.close(fig)

        lines += [
            f"### `{name}`",
            "",
            f"![{name} inner]({inner_chart})",
            "",
            "| K | Volume | Nodes per domain | Orchestrator | `inner_seq` "
            "| `inner_default` / `inner_seq` | `inner_par` / `inner_seq` |",
            "|---:|---:|---:|---|---:|---:|---:|",
        ]
        for k in counts:
            for v in volumes:
                for mode in (SEQUENTIAL, PARALLEL):
                    ref = table.get((INNER_REFERENCE, k, mode, v))
                    if ref is None:
                        continue
                    ratios = [
                        (
                            fmt_ratio(table[(inner, k, mode, v)], ref)
                            if (inner, k, mode, v) in table
                            else "—"
                        )
                        for inner in INNER_COMPARED
                    ]
                    lines.append(
                        f"| {k} | {fmt_size(v)} | {fmt_size(v / k)} | {mode} "
                        f"| {fmt_time(ref.mean)} | {' | '.join(ratios)} |"
                    )
        lines += [
            "",
            f"![{name} orchestrator]({orchestrator_chart})",
            "",
            "| K | Volume | " + " | ".join(f"`{i}`" for i in INNER_ORDER) + " |",
            "|---:|---:|" + "---:|" * len(INNER_ORDER),
        ]
        for k in counts:
            for v in volumes:
                cells = []
                for inner in INNER_ORDER:
                    seq = table.get((inner, k, SEQUENTIAL, v))
                    par = table.get((inner, k, PARALLEL, v))
                    cells.append(fmt_ratio(par, seq) if seq and par else "—")
                if any(cell != "—" for cell in cells):
                    lines.append(f"| {k} | {fmt_size(v)} | {' | '.join(cells)} |")
        lines += [
            "",
            "Orchestrator parallel / sequential time, by inner mode.",
            "",
        ]
    return lines


# ── Environment ─────────────────────────────────────────────────────────────


def default_venv() -> Path:
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "oxiflow" / "abaques-venv"


def venv_python(env: Path) -> Path:
    return env / ("Scripts" if os.name == "nt" else "bin") / "python"


def run_in_own_venv(env: Path) -> int:
    """Run this script again with the interpreter of `env`, creating `env`
    and installing the requirements first when needed."""
    python = venv_python(env)
    if Path(sys.prefix).resolve() == env.resolve():
        print(f"matplotlib is still missing from {env}", file=sys.stderr)
        return 1
    if not python.exists():
        print(f"creating the virtual environment {env}", file=sys.stderr)
        try:
            venv.create(env, with_pip=True, clear=True)
        except Exception as error:  # ensurepip missing, permissions…
            print(f"cannot create {env}: {error}", file=sys.stderr)
            return 1
    has_requirements = subprocess.run(
        [str(python), "-c", "import matplotlib"], capture_output=True
    )
    if has_requirements.returncode != 0:
        print(f"installing {', '.join(REQUIREMENTS)} in {env}", file=sys.stderr)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", *REQUIREMENTS],
            check=True,
        )
    rerun = subprocess.run([str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    return rerun.returncode


def load_matplotlib() -> None:
    global plt, FixedLocator, FuncFormatter, LogLocator, NullFormatter
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot
    import matplotlib.ticker

    plt = matplotlib.pyplot
    FixedLocator = matplotlib.ticker.FixedLocator
    FuncFormatter = matplotlib.ticker.FuncFormatter
    LogLocator = matplotlib.ticker.LogLocator
    NullFormatter = matplotlib.ticker.NullFormatter


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dispatch abaques from the oxiflow_parallel benchmark results."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("target/criterion"),
        help="Criterion results to read (default: target/criterion)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("target/abaques"),
        help="where the charts and abaques.md go (default: target/abaques)",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=default_venv(),
        help="the script's own virtual environment, used when matplotlib "
        "is not importable (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"no Criterion results under {args.source}", file=sys.stderr)
        return 1
    if importlib.util.find_spec("matplotlib") is None:
        return run_in_own_venv(args.venv)
    load_matplotlib()

    cases = load_cases(args.source)
    if not cases:
        print(f"no benchmark case under {args.source}", file=sys.stderr)
        return 1
    args.export.mkdir(parents=True, exist_ok=True)

    pairs = diagnostic_pairs(cases)
    lines = [
        "# Dispatch abaques",
        "",
        f"Generated by `scripts/abaques.py` from `{args.source}`, "
        f"{len(cases)} cases. Times are Criterion's mean estimates; a ratio "
        "is marked `~` when the confidence intervals of its two times "
        "overlap. Below 1, parallel is faster.",
        "",
    ]
    lines += write_per_call(pairs, args.export)
    lines += write_diagnostics(pairs, args.export)
    lines += write_orchestrator(orchestrator_cases(cases), args.export)
    (args.export / "abaques.md").write_text("\n".join(lines))
    print(f"{len(cases)} cases → {args.export / 'abaques.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
