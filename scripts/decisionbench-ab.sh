#!/bin/bash
# decisionbench-ab.sh: paired, interleaved rounds of dev/decisionbench against two
# servers that speak POST /v1/systemone (AGENTS.md RULE 1-2). Run it ON the host that
# holds both servers, which are already loaded and warm; it drives them, it does not
# start them.
#
#   ARM_A=8081:PID:MODEL ARM_B=8082:PID:MODEL \
#     scripts/decisionbench-ab.sh BENCH WORDS.tsv ROUNDS WINDOW_SECONDS OUT.tsv
#
# An arm is port:pid:model-field. Each sample SIGCONTs its server and SIGSTOPs the
# other, so an idle server's spinning workers share no core with the arm being
# timed; the order alternates by round (AB, BA, ...), so a slot bias cancels. A
# one-second discarded burst follows each wake-up. OUT gets one line per round:
# round, then for A and B the p50 in microseconds and the words in the window.
set -eu
bench=$1 words=$2 rounds=$3 win=$4 out=$5
IFS=: read -r pa pida ma <<<"$ARM_A"
IFS=: read -r pb pidb mb <<<"$ARM_B"

sample() { # sample port pid otherpid model -> "p50us words"
	kill -STOP "$3"
	kill -CONT "$2"
	"$bench" -url "http://127.0.0.1:$1" -model "$4" -file "$words" -cycle -window 1s -raw >/dev/null
	"$bench" -url "http://127.0.0.1:$1" -model "$4" -file "$words" -cycle -window "${win}s" -raw | cut -f1,2 | tr '\t' ' '
	kill -STOP "$2"
}

: >"$out"
trap 'kill -CONT "$pida" "$pidb" 2>/dev/null || true' EXIT
for r in $(seq 1 "$rounds"); do
	if [ $((r % 2)) -eq 1 ]; then
		a=$(sample "$pa" "$pida" "$pidb" "$ma")
		b=$(sample "$pb" "$pidb" "$pida" "$mb")
	else
		b=$(sample "$pb" "$pidb" "$pida" "$mb")
		a=$(sample "$pa" "$pida" "$pidb" "$ma")
	fi
	printf '%d %s %s\n' "$r" "$a" "$b" | tee -a "$out"
done
