#!/usr/bin/env bash
# CPU feedback loop for the "slow to react / CPU at 100%" bug.
# Run ON THE ROBOT while the app (interface_flask.py) is running.
#
#   scripts/cpu_probe.sh [seconds=60] [max_machine_pct=80] [max_proc_pct=90]
#
# During the window it drives a scripted conversation (scripts/talk_driver.py;
# CPU_PROBE_TALK=0 to measure idle instead) and, if py-spy is installed
# (pip install py-spy; needs sudo), profiles every app process.
#
# RED (exit 1) if, averaged over the window:
#   - the app process tree uses more than max_machine_pct of all cores, or
#   - any single app process uses more than max_proc_pct of ONE core
#     (a Python process pinned at one core = GIL saturated = laggy input), or
#   - any conversation turn reacted slower than TALK_MAX_REACTION (default 8s).
set -euo pipefail
PATH="$PATH:$HOME/.local/bin"  # pip install --user puts py-spy here

SECS="${1:-60}"
MAX_MACHINE="${2:-80}"
MAX_PROC="${3:-90}"
PATTERN="${CPU_PROBE_PATTERN:-interface_flask}"
TALK="${CPU_PROBE_TALK:-1}"
HZ="$(getconf CLK_TCK)"
NCPU="$(nproc)"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="/tmp/cpu_probe"
mkdir -p "$OUT"; rm -f "$OUT"/*.txt

root="$(pgrep -f "$PATTERN" | head -n1 || true)"
[ -n "$root" ] || { echo "no process matching '$PATTERN' - start the app first" >&2; exit 2; }

# root + all descendants
pids="$root"
frontier="$root"
while [ -n "$frontier" ]; do
  next=""
  for p in $frontier; do next="$next $(pgrep -P "$p" || true)"; done
  frontier="$(echo $next)"
  pids="$pids $frontier"
done

ticks() { awk '{print $14 + $15}' "/proc/$1/stat" 2>/dev/null || echo 0; }

declare -A before
for p in $pids; do before[$p]="$(ticks "$p")"; done

spy_jobs=()
if command -v py-spy >/dev/null; then
  for p in $pids; do
    sudo "$(command -v py-spy)" record --pid "$p" --duration "$SECS" --format raw --threads --nonblocking \
      -o "$OUT/$p.txt" >/dev/null 2>&1 &   # fails harmlessly on non-Python children
    spy_jobs+=($!)
  done
fi

sleep "$SECS" & timer=$!  # the window starts now, whatever the talk does
talk_rc=0
if [ "$TALK" = 1 ]; then
  # --pause: a new question cancels the answer still being spoken
  timeout "$SECS" python "$HERE/talk_driver.py" --max-reaction "${TALK_MAX_REACTION:-8}" --pause 12 \
    "${CPU_PROBE_SCENARIO:-$HERE/../tests/scenarios/robot_talk.json}" || talk_rc=$?
  [ "$talk_rc" = 124 ] && { echo "(conversation still running at end of window)"; talk_rc=0; }
fi
wait "$timer"
for j in "${spy_jobs[@]}"; do wait "$j" || true; done

red=0
total=0
printf '%-8s %8s  %s\n' PID "CPU%" CMD
for p in $pids; do
  d=$(( $(ticks "$p") - ${before[$p]} ))
  pct=$(( d * 100 / (HZ * SECS) ))   # 100 = one full core
  total=$(( total + d ))
  cmd="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-90)"
  printf '%-8s %7s%%  %s\n' "$p" "$pct" "$cmd"
  if [ "$pct" -gt "$MAX_PROC" ]; then echo "  ^ RED: above ${MAX_PROC}% of one core"; red=1; fi
done
machine=$(( total * 100 / (HZ * SECS * NCPU) ))
echo "app tree: ${machine}% of ${NCPU} cores (limit ${MAX_MACHINE}%), window ${SECS}s"
[ "$machine" -gt "$MAX_MACHINE" ] && { echo "RED: app tree above ${MAX_MACHINE}% of machine"; red=1; }
[ "$talk_rc" -ne 0 ] && { echo "RED: slow reaction (see turns above)"; red=1; }

for f in "$OUT"/*.txt; do
  [ -s "$f" ] || continue
  echo "== py-spy $(basename "$f" .txt): hottest threads (samples)"
  awk '{n=$NF; split($0,f,";"); th[f[1]]+=n} END{for(t in th) print th[t], t}' "$f" | sort -rn | head -8
  echo "== py-spy $(basename "$f" .txt): hottest leaf frames (samples)"
  awk '{n=$NF; $NF=""; k=split($0,f,";"); leaf[f[k]]+=n} END{for(l in leaf) print leaf[l], l}' "$f" | sort -rn | head -15
done
command -v py-spy >/dev/null || echo "(install py-spy for a per-function breakdown: pip install py-spy)"

[ "$red" -eq 0 ] && echo GREEN || echo RED
exit "$red"
