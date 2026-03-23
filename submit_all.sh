#!/bin/bash
# =============================================================================
# CLiGNet — Master Job Submission Script
# Submits all pipeline steps to SLURM with dependency chaining.
#
# Usage:
#   bash submit_all.sh              # submit all steps from scratch
#   bash submit_all.sh --from 4     # resume from a specific step
#   bash submit_all.sh --dry-run    # print sbatch commands without submitting
#
# Step map:
#   Step 2   — Build label graph           (GPU,  ~1–2h)
#   Step 3   — Train baselines B1, B2      (CPU,  ~1–2h)  — no dependency
#   Step 4   — Train BERT baselines B3, B4 (GPU,  ~6–12h) — depends on 2
#   Step 4b  — Train Longformer B7         (GPU,  ~6–12h) — depends on 2
#   Step 5   — Train CLiGNet B6, B8        (GPU,  ~8h)    — depends on 2
#   Step 6   — Evaluate + McNemar          (GPU,  ~1h)    — depends on 3,4,4b,5
#   Step 7   — Ablations A1,A2,A4,A5       (GPU,  ~16h)   — depends on 2
#   Step 8   — Interpretability + IG       (GPU,  ~4h)    — depends on 5
# =============================================================================

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")/slurm" && pwd)"
FROM_STEP=2
DRY_RUN=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)    FROM_STEP="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Helper: submit one job; prints job ID to stdout, status to stderr ─────────
submit() {
  local dep_flag="$1"
  local script="$2"
  if $DRY_RUN; then
    echo "[DRY-RUN] sbatch ${dep_flag:+$dep_flag }$script" >&2
    echo "FAKE_JOB_ID"
  else
    sbatch $dep_flag "$script" | awk '{print $4}'
  fi
}

echo "============================================="
echo " CLiGNet Pipeline Submission"
echo " Starting from step: $FROM_STEP"
echo " Dry run: $DRY_RUN"
echo "============================================="

# ── Step 2: Build label graph ─────────────────────────────────────────────────
if [[ $FROM_STEP -le 2 ]]; then
  echo -n "[Step 2 ] Submitting build_graph...          "
  JID2=$(submit "" "$SCRIPTS_DIR/step2_build_graph.sh")
  echo "Job ID: $JID2"
else
  echo "[Step 2 ] Skipped (--from $FROM_STEP)"
  JID2=""
fi

# ── Step 3: Classical baselines B1, B2 (CPU, no graph dependency) ────────────
if [[ $FROM_STEP -le 3 ]]; then
  echo -n "[Step 3 ] Submitting train_baselines B1,B2... "
  JID3=$(submit "" "$SCRIPTS_DIR/step3_train_baselines.sh")
  echo "Job ID: $JID3"
else
  echo "[Step 3 ] Skipped (--from $FROM_STEP)"
  JID3=""
fi

# ── Step 4: BERT baselines B3, B4 (depends on Step 2) ────────────────────────
if [[ $FROM_STEP -le 4 ]]; then
  echo -n "[Step 4 ] Submitting train_bert_baselines B3,B4... "
  DEP4=""
  [[ -n "$JID2" ]] && DEP4="--dependency=afterok:$JID2"
  JID4=$(submit "$DEP4" "$SCRIPTS_DIR/step4_train_bert_baselines.sh")
  echo "Job ID: $JID4"
else
  echo "[Step 4 ] Skipped (--from $FROM_STEP)"
  JID4=""
fi

# ── Step 4b: Longformer B7 (depends on Step 2, separate job for memory control)
if [[ $FROM_STEP -le 4 ]]; then
  echo -n "[Step 4b] Submitting train_b7_longformer...  "
  DEP4B=""
  [[ -n "$JID2" ]] && DEP4B="--dependency=afterok:$JID2"
  JID4B=$(submit "$DEP4B" "$SCRIPTS_DIR/step4b_train_b7.sh")
  echo "Job ID: $JID4B"
else
  echo "[Step 4b] Skipped (--from $FROM_STEP)"
  JID4B=""
fi

# ── Step 5: CLiGNet B6, B8 (depends on Step 2) ───────────────────────────────
if [[ $FROM_STEP -le 5 ]]; then
  echo -n "[Step 5 ] Submitting train_clignet B6,B8...  "
  DEP5=""
  [[ -n "$JID2" ]] && DEP5="--dependency=afterok:$JID2"
  JID5=$(submit "$DEP5" "$SCRIPTS_DIR/step5_train_clignet.sh")
  echo "Job ID: $JID5"
else
  echo "[Step 5 ] Skipped (--from $FROM_STEP)"
  JID5=""
fi

# ── Step 6: Evaluate (depends on Steps 3, 4, 4b, 5) ─────────────────────────
if [[ $FROM_STEP -le 6 ]]; then
  echo -n "[Step 6 ] Submitting evaluate...             "
  # Note: JID3 runs on chip-cpu — cross-cluster deps are unsupported here.
  # B1/B2 (~1-2h) always finish before step 6 unlocks (~8-12h of GPU work).
  DEPS=()
  [[ -n "$JID4"  ]] && DEPS+=("$JID4")
  [[ -n "$JID4B" ]] && DEPS+=("$JID4B")
  [[ -n "$JID5"  ]] && DEPS+=("$JID5")
  DEP6=""
  if [[ ${#DEPS[@]} -gt 0 ]]; then
    DEP6="--dependency=afterok:$(IFS=:; echo "${DEPS[*]}")"
  fi
  JID6=$(submit "$DEP6" "$SCRIPTS_DIR/step6_evaluate.sh")
  echo "Job ID: $JID6"
else
  echo "[Step 6 ] Skipped (--from $FROM_STEP)"
  JID6=""
fi

# ── Step 7: Ablations A1,A2,A4,A5 (depends on Step 2) ───────────────────────
if [[ $FROM_STEP -le 7 ]]; then
  echo -n "[Step 7 ] Submitting ablation A1,A2,A4,A5... "
  DEP7=""
  [[ -n "$JID2" ]] && DEP7="--dependency=afterok:$JID2"
  JID7=$(submit "$DEP7" "$SCRIPTS_DIR/step7_ablation.sh")
  echo "Job ID: $JID7"
else
  echo "[Step 7 ] Skipped (--from $FROM_STEP)"
  JID7=""
fi

# ── Step 8: Interpretability + IG (depends on Step 5) ────────────────────────
if [[ $FROM_STEP -le 8 ]]; then
  echo -n "[Step 8 ] Submitting interpret...            "
  DEP8=""
  [[ -n "$JID5" ]] && DEP8="--dependency=afterok:$JID5"
  JID8=$(submit "$DEP8" "$SCRIPTS_DIR/step8_interpret.sh")
  echo "Job ID: $JID8"
else
  echo "[Step 8 ] Skipped (--from $FROM_STEP)"
  JID8=""
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo " Submission complete. Job summary:"
echo "  Step 2  (build_graph)          : ${JID2:-skipped}"
echo "  Step 3  (baselines B1,B2)      : ${JID3:-skipped}"
echo "  Step 4  (BERT baselines B3,B4) : ${JID4:-skipped}"
echo "  Step 4b (Longformer B7)        : ${JID4B:-skipped}"
echo "  Step 5  (CLiGNet B6,B8)        : ${JID5:-skipped}"
echo "  Step 6  (evaluate)             : ${JID6:-skipped}"
echo "  Step 7  (ablation A1-A5)       : ${JID7:-skipped}"
echo "  Step 8  (interpret)            : ${JID8:-skipped}"
echo "============================================="
echo ""
echo "Monitor all jobs:  squeue --clusters=chip-gpu,chip-cpu -u \$USER"
echo "Watch a log:       tail -f logs/<job-name>-<job-id>.err"
echo "============================================="
