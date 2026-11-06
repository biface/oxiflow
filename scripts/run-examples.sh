#!/usr/bin/env bash
# =============================================================================
# run-examples.sh — run every example and collect the figures they draw
# =============================================================================
# Builds the examples of examples/ once, in release, then runs each of them
# from an output directory: the reference examples check themselves against
# their exact solutions and exit with an error beyond their tolerances, and
# the _plot and _simul examples write their SVG figures to the working
# directory. Useful before a pull request to staging/x.y.z, and to refresh
# the figures of the wiki.
#
# Output, under OUTPUT_DIR (default target/examples-run/<date>):
#   logs/<example>.log   standard output and error of each example
#   logs/_build.log      output of the build
#   logs/_summary.txt    OK, FAILED(code) or TIMEOUT, and duration, per example
#   *.svg                the figures
#   figures.zip          the figures and the summary
#
# Usage:
#   scripts/run-examples.sh [OUTPUT_DIR]
#
# Environment:
#   FEATURES   cargo features (default parallel,serde,vtk,sparse)
#   TIMEOUT    limit per example, in the format of timeout(1) (default 30m)
#
# Exits with 1 if an example fails or times out, 2 on a build error. Runs
# from any directory: it moves to the root of the repository first.
# =============================================================================

set -uo pipefail

FEATURES="${FEATURES:-parallel,serde,vtk,sparse}"
TIMEOUT="${TIMEOUT:-30m}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
OUT="${1:-target/examples-run/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/logs"
OUT="$(cd "$OUT" && pwd)"

echo "== Building the examples (release, features: $FEATURES)"
if ! cargo build --release --examples --features "$FEATURES" > "$OUT/logs/_build.log" 2>&1; then
    tail -20 "$OUT/logs/_build.log" >&2
    echo "Build failed, see $OUT/logs/_build.log" >&2
    exit 2
fi

TARGET_DIR="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
    | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')"
BIN_DIR="${TARGET_DIR:-$ROOT/target}/release/examples"

mapfile -t EXAMPLES < <(find examples -maxdepth 1 -name '*.rs' -printf '%f\n' \
    | sed 's/\.rs$//' | sort)

summary=()
failures=0
cd "$OUT" || exit 2
for name in "${EXAMPLES[@]}"; do
    printf '== %-36s ' "$name"
    bin="$BIN_DIR/$name"
    if [[ ! -x "$bin" ]]; then
        result="MISSING"
        elapsed="-"
        failures=$((failures + 1))
    else
        start=$(date +%s)
        timeout "$TIMEOUT" "$bin" > "logs/$name.log" 2>&1
        status=$?
        elapsed="$(( $(date +%s) - start ))s"
        case $status in
            0)   result="OK" ;;
            124) result="TIMEOUT"; failures=$((failures + 1)) ;;
            *)   result="FAILED($status)"; failures=$((failures + 1)) ;;
        esac
    fi
    echo "$result $elapsed"
    summary+=("$(printf '%-36s %-12s %7s' "$name" "$result" "$elapsed")")
done

echo
echo "== Summary"
printf '%s\n' "${summary[@]}" | tee logs/_summary.txt

shopt -s nullglob
figures=( *.svg *.png )
echo
if (( ${#figures[@]} == 0 )); then
    echo "== No figure written"
elif command -v zip > /dev/null; then
    rm -f figures.zip
    zip -q figures.zip "${figures[@]}" logs/_summary.txt
    echo "== ${#figures[@]} figure(s) in $OUT, archived in figures.zip"
else
    echo "== ${#figures[@]} figure(s) in $OUT (zip not found, no archive)"
fi

exit $(( failures > 0 ))
