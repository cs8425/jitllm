#!/usr/bin/env bash
# Compare jitllm against Strata (Niko1221/Strata, the engine built for
# Qwen3.8-Flash-Next) on one model and one device, PAIRED and INTERLEAVED, and
# report the median of per-round ratios with its IQR (AGENTS.md RULE 1, 2).
#
#   JITLLM_PATH=/m/q.jlm STRATA=~/strata-src/build-v100/strata \
#     STRATA_PACK=/m/strata/pack STRATA_SHARD1=/m/q-00001-of-00004.gguf \
#     scripts/vs-strata.sh 20
#
# It is vs-llamacpp.sh's method with Strata in the B arm: the other engine is
# another program, so each sample is a fresh process of one engine, the rounds
# alternate ABBA so drift cannot accumulate on one side, and each engine's own
# decode rate is read off its own output (load and prompt excluded on both).
# STRATA_VS_AA=1 puts jitllm in both arms: the self-control that shows the
# harness discriminates, run first and quoted beside the row.
#
# Both engines run on one GPU and one socket: CUDA_VISIBLE_DEVICES=$GPU, and
# numactl binds the CPU and the memory to $NODE, since both run routed experts
# on the host. Speculation is off in both arms: jitllm runs no -spec, and
# Strata (whose native pack requires verify windows, --spec 2) is given no
# drafter -- no --mtp and --suffix-draft 0 -- so no draft is ever accepted; its
# report of accepted drafts is checked to be zero on every sample.
set -euo pipefail

if [ "${JITLLM_NO_LOCK:-0}" != "1" ] && command -v flock >/dev/null 2>&1; then
  lock="${TMPDIR:-/tmp}/jitllm-cap.lock"
  exec 9>>"$lock"
  flock -w "${JITLLM_LOCK_WAIT:-1800}" -x 9 || { echo ">> REFUSING: the box stayed busy" >&2; exit 75; }
fi

ROUNDS="${1:-20}"
JITLLM="${JITLLM:-./jitllm}"
: "${JITLLM_PATH:?the .jlm container}" "${STRATA:?the strata binary}" "${STRATA_PACK:?the Strata pack}"
: "${STRATA_SHARD1:?the first GGUF shard, where Strata reads the experts}"
GPU="${GPU:-0}"
NODE="${NODE:-0}"
N="${STRATA_VS_N:-128}"
PROMPT="${STRATA_VS_PROMPT:-The capital of France is}"
JITLLM_ENV="${JITLLM_ENV:-JITLLM_EXPERTS=host}"
STRATA_ARGS="${STRATA_ARGS:---resident-budget-gib 100 --expert-cache auto --max-context 8192 --prefill auto}"
STRATA_PROFILE="${STRATA_PROFILE:-$(dirname "$STRATA")/../data/expert-profile.bin}"
# NODE=all runs both engines unpinned: both sockets, the 2-socket rows.
PIN=(numactl --cpunodebind="$NODE" --membind="$NODE")
[ "$NODE" = all ] && PIN=()
LOG="${STRATA_VS_LOG:-vs-strata.$(date +%s).log}"

# The prompt as ids, from jitllm's own tokenizer, so both engines read the
# same tokens.
IDS="$("$JITLLM" tokenize "$JITLLM_PATH" "$PROMPT" 2>&1 | sed -n 's/^[0-9]* tokens: \[\(.*\)\]$/\1/p' | tr ' ' ',')"
[ -n "$IDS" ] || { echo "could not tokenize the prompt" >&2; exit 1; }

psi() { awk '/^some/{split($2,a,"="); print a[2]}' /proc/pressure/cpu 2>/dev/null || echo 0; }
# How busy the CPUs are before a sample (corebusy.py): PSI cannot see a
# tenant on a box with CPUs to spare (RULE 2), and the measured GPUs must be
# empty. The CPUs checked are the node's the run is pinned to, or every CPU
# when it is not pinned (NODE=all): a box that runs correctness gates on its
# other cards and socket beside a measurement (gates on GPUs 2-7 under their
# own locks) would otherwise never be quiet.
if [ "$NODE" = all ]; then
  CPUS="$(seq -s, 0 $(($(nproc --all) - 1)))"
else
  CPUS="$(cat /sys/devices/system/node/node$NODE/cpulist)"
fi
BUSYMAX="${STRATA_VS_BUSY:-10}"
busy() { "$(dirname "$0")/corebusy.py" "$CPUS" 2>/dev/null || echo 0; }
# Waits for the node to go quiet (STRATA_VS_WAIT seconds at most), so a
# tenant that started first is outlasted rather than measured.
quiet() {
  local waited=0 b g
  while :; do
    b="$(busy)"
    g="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1 | tr -d ' ')"
    awk -v b="$b" -v m="$BUSYMAX" -v g="${g:-0}" 'BEGIN{exit !(b <= m && g < 600)}' && return 0
    if [ "$waited" -ge "${STRATA_VS_WAIT:-0}" ]; then
      echo "the node is ${b}% busy and a measured GPU holds ${g} MiB before a sample: refusing" >&2
      return 1
    fi
    sleep 30
    waited=$((waited + 30))
  done
}
gpuclk() { nvidia-smi -i "${GPU%%,*}" --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
# Several cards (GPU=0,1): jitllm places over every visible card and Strata
# splits its layers over the same ones; the rows are X cards against X.
JDEV=cuda:0
case "$GPU" in *,*) JDEV=cuda ;; esac
# STRATA_MTP=<dir> runs Strata at its best: its MTP drafter, window
# STRATA_SPEC (4), accepted drafts allowed. jitllm runs no speculation.
STRATA_SPECARGS="--spec 2 --suffix-draft 0"
if [ -n "${STRATA_MTP:-}" ]; then
  STRATA_SPECARGS="--mtp $STRATA_MTP --spec ${STRATA_SPEC:-4}"
fi

jitllm_run() {
  env CUDA_VISIBLE_DEVICES="$GPU" $JITLLM_ENV "${PIN[@]}" "$JITLLM" run -devices "$JDEV" -n "$N" "$JITLLM_PATH" "$PROMPT" \
    2>>"$LOG" >/dev/null || return 1
  tail -n 40 "$LOG" | sed -n 's|.*decode [0-9]* tok in [^(]*(\([0-9.]*\) tok/s).*|\1|p' | tail -1
}
strata_run() {
  local out
  out="$("${PIN[@]}" "$STRATA" --pack "$STRATA_PACK" --native "$STRATA_SHARD1" --expert-profile "$STRATA_PROFILE" \
    $STRATA_ARGS --tokens "$IDS" --max-new "$N" --gpu "$GPU" $STRATA_SPECARGS 2>&1)" || { echo "$out" >>"$LOG"; return 1; }
  echo "$out" >>"$LOG"
  local acc
  acc="$(echo "$out" | sed -n 's|.*drafts accepted \([0-9]*\) of.*|\1|p' | tail -1)"
  if [ -z "${STRATA_MTP:-}" ] && [ "${acc:-1}" != "0" ]; then
    echo "Strata accepted ${acc:-?} drafts: this sample speculated" >&2
    return 1
  fi
  echo "$out" | sed -n 's|^decode .* -> *\([0-9.]*\) tok/s.*|\1|p' | tail -1
}
# The B arm of a jitllm A/B (STRATA_VS_AA with JITLLM_B or JITLLM_ENV_B):
# another build or other settings, under the same gates and interleave.
# JITLLM_B_UNPINNED=1 runs the B arm on every socket (the 2-socket arm).
jitllm_run_b() {
  local pin=("${PIN[@]}")
  [ -n "${JITLLM_B_UNPINNED:-}" ] && PIN=()
  JITLLM="${JITLLM_B:-$JITLLM}" JITLLM_ENV="${JITLLM_ENV_B:-$JITLLM_ENV}" jitllm_run
  local rc=$?
  PIN=("${pin[@]}")
  return $rc
}
A=jitllm_run
B=strata_run
NAME_B=strata
if [ -n "${STRATA_VS_AA:-}" ]; then
  B=jitllm_run_b
  NAME_B="jitllm${JITLLM_B:+ $(basename "$JITLLM_B")}${JITLLM_ENV_B:+ ($JITLLM_ENV_B)}"
fi

echo "jitllm $(basename "$JITLLM_PATH") ($JITLLM_ENV) vs $NAME_B on GPU $GPU, node $NODE: decode $N tokens after the prompt [$IDS], $ROUNDS ABBA rounds; log $LOG"
nvidia-smi -i "$GPU" --query-gpu=name,driver_version --format=csv,noheader
# The soak: one sample of each, unused.
quiet || exit 75
echo ">> soak"
$A >/dev/null || true
$B >/dev/null || true
# A sample that fails (a crash, a refused speculated sample) is retried
# once; a round whose sample still gives no rate is dropped and counted, and
# the row is refused only when more than a quarter of its rounds drop.
retry() { "$1" || { echo "  $1 failed; retrying once (see $LOG)" >&2; "$1"; }; }
dropped=0
ratios=()
for r in $(seq 1 "$ROUNDS"); do
  quiet || exit 75
  p="$(psi)"
  if awk -v p="$p" 'BEGIN{exit !(p > 25)}'; then
    echo "round $r: PSI some avg10 $p > 25: the box is busy, refusing" >&2
    exit 75
  fi
  if [ $((r % 2)) -eq 1 ]; then
    a="$(retry $A)"; ca="$(gpuclk)"; b="$(retry $B)"; cb="$(gpuclk)"
  else
    b="$(retry $B)"; cb="$(gpuclk)"; a="$(retry $A)"; ca="$(gpuclk)"
  fi
  if [ -z "$a" ] || [ -z "$b" ]; then
    dropped=$((dropped + 1))
    echo "round $r: dropped, a sample gave no rate after a retry (see $LOG); $dropped dropped so far" >&2
    if [ $((dropped * 4)) -gt "$ROUNDS" ]; then
      echo "more than a quarter of $ROUNDS rounds dropped: refusing the row" >&2
      exit 1
    fi
    continue
  fi
  ratio="$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.4f", a/b}')"
  ratios+=("$ratio")
  echo "round $r: jitllm $a tok/s (sm,temp $ca)  $NAME_B $b tok/s (sm,temp $cb)  ratio $ratio  psi $p"
done
echo "$dropped of $ROUNDS round(s) dropped"
printf '%s\n' "${ratios[@]}" | sort -n | awk '
  { v[NR]=$1 }
  END {
    n=NR
    m = (n%2) ? v[(n+1)/2] : (v[n/2]+v[n/2+1])/2
    lo = v[int((n+3)/4)]; hi = v[int((3*n+1)/4)]
    iqr = hi-lo
    printf "median ratio %.4f  IQR %.4f  IQR/median %.4f  over %d rounds\n", m, iqr, iqr/m, n
    if (iqr/m > 0.10) { print "REJECT: IQR/median above 0.10"; exit 3 }
  }'
