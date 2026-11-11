#!/usr/bin/env bash
# Rerun every result and figure of the paper from the distributed inputs, then compare with the published results.
#
#     bash scripts/reproduce.sh            # all stages
#     WORKERS=16 bash scripts/reproduce.sh # parallel processes for sensitivity and Monte Carlo (default 4)
#     N_DRAWS=20 bash scripts/reproduce.sh # fewer Monte Carlo draws, for a quick test (the paper uses 500)
#
# Restartable: each finished stage leaves a marker in work/reproduce/done/ and is skipped next time; Monte Carlo
# draws are checkpointed in work/mc_checkpoints/ as they finish. Delete work/reproduce/ to start over.
# Size WORKERS to memory, not cores (see README). Logs are in work/reproduce/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
WORKERS=${WORKERS:-4}
N_DRAWS=${N_DRAWS:-500}
export RR_SENS_WORKERS=${RR_SENS_WORKERS:-$WORKERS}
export RR_MC_WORKERS=${RR_MC_WORKERS:-$WORKERS}
D=work/reproduce
mkdir -p "$D/done" "$D/logs"

stage() {   # stage NAME COMMAND...: run once, log to work/reproduce/logs/NAME.log
    local name=$1; shift
    if [ -e "$D/done/$name" ]; then echo "[skip]  $name"; return 0; fi
    echo "[start] $name  $(date '+%F %T')"
    if "$@" > "$D/logs/$name.log" 2>&1; then
        touch "$D/done/$name"; echo "[done]  $name  $(date '+%F %T')"
    else
        echo "[FAILED] $name, see $D/logs/$name.log"; exit 1
    fi
}

snapshot() {   # keep the published results to compare with; refuse if results/ was already changed
    if [ -d work/published_results ]; then return 0; fi
    "$PY" - <<'EOF'
import hashlib, json, sys
from pathlib import Path
published = json.loads(Path("results/manifest.json").read_text())["results"]
changed = [p for p, h in published.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
if changed:
    sys.exit("results/ no longer holds the published results (" + ", ".join(changed[:5]) + "); restore them first")
EOF
    mkdir -p work && cp -r results work/published_results
}

figures() {
    for f in fig3_supply_need fig4_maps fig5_no_intervention fig6_food_saved fig7_map_saved fig8_limits \
             fig9_tornado figS1_map_saved_16 figS2_map_saved_150 figS3_limits_16 figS4_limits_150 figS5_countries \
             figS6_tornado_16 figS7_tornado_150; do
        echo "== $f"; "$PY" -m src.figures.$f
    done
    if [ -e work/grid/spam_prod.npz ]; then
        echo "== fig2_reach_inputs"; "$PY" -m src.figures.fig2_reach_inputs
    else
        echo "== fig2_reach_inputs skipped: it needs the 5 arcmin grid (python scripts/download_inputs.py grid,"
        echo "   then python -m src.reach.grid)"
    fi
}

echo "reproduce: $WORKERS workers, $N_DRAWS Monte Carlo draws"
stage snapshot     snapshot
stage core         "$PY" -m src.runs.core
stage sensitivity  "$PY" -m src.runs.sensitivity
stage calibration  "$PY" -m src.runs.calibration
stage urea         "$PY" -m src.runs.urea_budget
stage inventory    "$PY" -m src.runs.residue_inventory
stage montecarlo   "$PY" -m src.runs.sampling "$N_DRAWS" both
stage figures      figures
echo "[start] compare"
"$PY" scripts/compare_results.py | tee "$D/compare.txt"
