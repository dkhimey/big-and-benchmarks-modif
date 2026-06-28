#!/usr/bin/env bash
set -uo pipefail

# ---- dispatch-only defaults (override via flags) ------------------------
HOSTS_FILE="hosts.txt"
REMOTE_DIR="."                                   # repo root on each host (Python is launched from here)
SCRIPT_PATH="benchmark/streaming/compute_gt_updated.py"  # path to compute_gt.py relative to REMOTE_DIR
PYTHON="python"
SSH_USER="${SSH_USER:-$USER}"
LOG_DIR="./gt_logs"
# The first host in the hosts file (worker 0, the master) has big-disk space
# only on the NFS export itself, so it writes its scratch there. Every other
# host writes scratch to a local SSD mount. Both paths are passed through to
# compute_gt.py as --local_scratch, chosen per host below.
MASTER_SCRATCH="~/extra"
WORKER_SCRATCH="~/extra_local"
# -------------------------------------------------------------------------

usage() {
    cat >&2 <<EOF
Usage: $0 [dispatch opts] -- [compute_gt.py args]

Dispatch options (consumed by this script):
  --hosts_file PATH    File of hosts, one per line (default: hosts.txt)
  --remote_dir PATH    Repo root on each host; Python is launched from here (default: .)
  --script_path PATH   Path to compute_gt.py relative to --remote_dir
                       (default: benchmark/streaming/compute_gt.py)
  --python PATH        Python executable on remote hosts (default: python)
  --ssh_user USER      SSH user (default: \$USER)
  --log_dir PATH       Local dir for per-worker logs (default: ./gt_logs)
  --master_scratch P   Scratch dir for the FIRST host in the hosts file
                       (worker 0); written on the NFS export (default: ~/extra)
  --worker_scratch P   Scratch dir for all other hosts; their local SSD mount
                       (default: ~/extra_local)

Everything after -- is forwarded verbatim to compute_gt.py, with
--num_workers, --worker_id, and --local_scratch appended automatically per host.
Do NOT pass --local_scratch yourself; it is set per host by this script.

Example:
  $0 --hosts_file hosts.txt \\
     --remote_dir /users/dkhimey/big-and-benchmarks-modif \\
     --script_path benchmark/streaming/compute_gt.py -- \\
     --dataset msturing-30M-clustered \\
     --runbook_file neurips23/streaming/runbook.yaml \\
     --gt_cmdline_tool ./compute_groundtruth
EOF
    exit 1
}

# ---- split args at "--" --------------------------------------------------
# Everything before "--" is for us; everything after is for compute_gt.py.
DISPATCH_ARGS=()
PASSTHROUGH_ARGS=()
seen_sep=0
for arg in "$@"; do
    if [[ "$seen_sep" -eq 0 && "$arg" == "--" ]]; then
        seen_sep=1
        continue
    fi
    if [[ "$seen_sep" -eq 0 ]]; then
        DISPATCH_ARGS+=("$arg")
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

if [[ "$seen_sep" -eq 0 ]]; then
    echo "Error: missing '--' separator between dispatch opts and compute_gt.py args." >&2
    usage
fi

# ---- parse dispatch-only flags ------------------------------------------
set -- "${DISPATCH_ARGS[@]+"${DISPATCH_ARGS[@]}"}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts_file)  HOSTS_FILE="$2";  shift 2 ;;
        --remote_dir)  REMOTE_DIR="$2";  shift 2 ;;
        --script_path) SCRIPT_PATH="$2"; shift 2 ;;
        --python)          PYTHON="$2";         shift 2 ;;
        --ssh_user)        SSH_USER="$2";       shift 2 ;;
        --log_dir)         LOG_DIR="$2";        shift 2 ;;
        --master_scratch)  MASTER_SCRATCH="$2"; shift 2 ;;
        --worker_scratch)  WORKER_SCRATCH="$2"; shift 2 ;;
        -h|--help)     usage ;;
        *) echo "Unknown dispatch option: $1" >&2; usage ;;
    esac
done

if [[ "${#PASSTHROUGH_ARGS[@]}" -eq 0 ]]; then
    echo "Error: no compute_gt.py arguments given after '--'." >&2
    usage
fi

# ---- read hosts ----------------------------------------------------------
mapfile -t HOSTS < <(grep -vE '^\s*(#|$)' "$HOSTS_FILE")
NUM_WORKERS="${#HOSTS[@]}"
if [[ "$NUM_WORKERS" -eq 0 ]]; then
    echo "No hosts found in $HOSTS_FILE" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "Forwarding to compute_gt.py: ${PASSTHROUGH_ARGS[*]}"
echo "Dispatching to $NUM_WORKERS workers:"
for i in "${!HOSTS[@]}"; do
    if [[ "$i" -eq 0 ]]; then
        echo "  worker $i -> ${HOSTS[$i]}  (scratch: $MASTER_SCRATCH)"
    else
        echo "  worker $i -> ${HOSTS[$i]}  (scratch: $WORKER_SCRATCH)"
    fi
done
echo

# ---- build a properly-quoted passthrough string -------------------------
# printf %q quotes each arg so spaces/special chars survive the remote shell.
PT_QUOTED=""
for a in "${PASSTHROUGH_ARGS[@]}"; do
    PT_QUOTED+=" $(printf '%q' "$a")"
done

# ---- launch --------------------------------------------------------------
declare -a PIDS PID_HOST PID_WID

for i in "${!HOSTS[@]}"; do
    host="${HOSTS[$i]}"
    log="$LOG_DIR/worker_${i}_${host}.log"

    # First host in the file (worker 0) writes scratch to the NFS export; all
    # others write to their local SSD. The tilde is intentionally left unquoted
    # in the remote command so the REMOTE shell expands it to that host's $HOME.
    if [[ "$i" -eq 0 ]]; then
        scratch="$MASTER_SCRATCH"
    else
        scratch="$WORKER_SCRATCH"
    fi

    # cd into the repo root, then run the script by its path so the top-level
    # `benchmark` package resolves (equivalent to running
    # `python benchmark/streaming/compute_gt.py` from the repo root).
    # NOTE: $scratch is NOT run through printf %q so a leading ~ expands on the
    # remote host. It must therefore be a simple path with no spaces.
    remote_cmd="cd $(printf '%q' "$REMOTE_DIR") && \
$(printf '%q' "$PYTHON") $(printf '%q' "$SCRIPT_PATH")${PT_QUOTED} \
--num_workers $NUM_WORKERS --worker_id $i \
--local_scratch $scratch"

    # -tt forces a pseudo-tty so the remote process is killed if ssh dies;
    # BatchMode avoids hanging on a password prompt.
    ssh -tt -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        "${SSH_USER}@${host}" "$remote_cmd" \
        > >(tee "$log") 2>&1 &

    PIDS+=($!)
    PID_HOST+=("$host")
    PID_WID+=("$i")
done

echo
echo "All $NUM_WORKERS workers launched. Waiting for completion..."
echo

# ---- wait & summarize ----------------------------------------------------
FAILED=0
for idx in "${!PIDS[@]}"; do
    if wait "${PIDS[$idx]}"; then
        echo "worker ${PID_WID[$idx]} (${PID_HOST[$idx]}): OK"
    else
        rc=$?
        echo "worker ${PID_WID[$idx]} (${PID_HOST[$idx]}): FAILED (exit $rc)" >&2
        FAILED=$((FAILED + 1))
    fi
done

echo
if [[ "$FAILED" -eq 0 ]]; then
    echo "All workers finished successfully."
else
    echo "$FAILED worker(s) failed. Check logs in $LOG_DIR." >&2
    exit 1
fi
