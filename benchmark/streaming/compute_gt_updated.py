import argparse
import errno
import os
import re
import socket
import time
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]

from benchmark.datasets import DATASETS
from benchmark.streaming.load_runbook import load_runbook


def lock_path_for(gt_file):
    """Path of the claim/lock file that guards a given GT file."""
    return gt_file + '.lock'


def try_claim(lock_file):
    """
    Atomically claim a step by creating its lock file with O_CREAT|O_EXCL.

    On NFS (v3+) this open is atomic across clients: exactly one worker's
    create succeeds, every other worker gets EEXIST. The winner owns the step.
    Returns True if this worker claimed it, False if someone else already has.

    The lock records who holds it and when, so stale locks (from a worker that
    died mid-step) can be identified and reclaimed.
    """
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as e:
        if e.errno == errno.EEXIST:
            return False
        raise
    try:
        os.write(fd, f"{socket.gethostname()}:{os.getpid()}:{time.time()}\n".encode())
    finally:
        os.close(fd)
    return True


def lock_is_stale(lock_file, stale_after_s):
    """
    A lock is stale if it is older than stale_after_s. Used to reclaim steps
    whose owner died without producing a valid GT. Age is taken from the lock
    file's mtime (set at creation) rather than its contents, so a malformed
    lock is still reclaimable.
    """
    try:
        age = time.time() - os.path.getmtime(lock_file)
    except OSError:
        # Lock vanished between checks: treat as not-stale; the claim attempt
        # that follows will create it fresh.
        return False
    return age > stale_after_s

def get_range_start_end(entry, tag_to_id):
    for i in range(entry['end'] - entry['start']):
        tag_to_id[i+entry['start']] = i+entry['start']
    return tag_to_id

def get_next_set(tag_to_id: np.ndarray, entry):
    match entry['operation']:
        case 'insert':
            for i in range(entry['end'] - entry['start']):
                tag_to_id[i+entry['start']] = i+entry['start']
            return tag_to_id
        case 'delete':
            # delete is by key 
            for i in range(entry['end'] - entry['start']):
                tag_to_id.pop(i + entry['start'])
            return tag_to_id
        case 'replace':
            # replace key with value
            for i in range(entry['tags_end'] - entry['tags_start']):
                tag_to_id[i + entry['tags_start']] = entry['ids_start'] + i
            return tag_to_id
        case 'search':
            return tag_to_id
        case _:       
            raise ValueError('Undefined entry in runbook')
        
def gt_dir(ds, runbook_path):
    runbook_filename = os.path.split(runbook_path)[1]
    return os.path.join(ds.basedir, str(ds.nb), runbook_filename)


def get_num_queries(ds):
    """
    Get the number of queries from the dataset query file.
    Assumes binary format with 4-byte int header for number of queries.
    """
    query_file = os.path.join(ds.basedir, ds.qs_fn)
    if not os.path.exists(query_file):
        return None
    
    try:
        with open(query_file, 'rb') as f:
            nq = int.from_bytes(f.read(4), byteorder='little')
        return nq
    except Exception as e:
        print(f"Warning: Could not read query file to get number of queries: {e}")
        return None

def validate_gt_file(gt_file, num_queries, k=100):
    """
    Validate that a ground truth file exists and has the expected size.
    Expected format:
    - num_queries (uint32_t) - 4 bytes
    - K (uint32) - 4 bytes
    - num_queries * K * sizeof(uint32_t) bytes - IDs of K-nearest neighbors
    - num_queries * K * sizeof(float) bytes - distances to K-nearest neighbors
    
    Returns True if valid, False if missing or corrupted.
    """
    if not os.path.exists(gt_file):
        return False
    
    # Expected size: 4 (num_queries) + 4 (K) + num_queries * K * 4 (IDs) + num_queries * K * 4 (distances)
    expected_size = 8 + num_queries * k * 8
    actual_size = os.path.getsize(gt_file)
    
    if actual_size != expected_size:
        return False
    
    return True

def find_resume_step(ds, runbook_path):
    """
    Find the highest completed runbook step by scanning existing step*.gt100 files.
    Returns 0 when no completed step is found.
    """
    output_dir = gt_dir(ds, runbook_path)
    if not os.path.isdir(output_dir):
        return 0

    step_re = re.compile(r"^step(\d+)\.gt100$")
    max_step = 0

    for filename in os.listdir(output_dir):
        match = step_re.match(filename)
        if not match:
            continue
        step = int(match.group(1))
        if step > max_step:
            max_step = step

    return max_step

def output_gt(ds, base_data, tag_to_id, step, gt_cmdline, runbook_path,
              local_scratch='/tmp'):
    """
    Compute and publish the ground truth for one step. Returns True if a valid
    GT file was published, False otherwise. The caller is responsible for the
    claim lock; this function only reports whether it succeeded so the caller
    can decide whether to release the lock.
    """
    ids_list = []
    tags_list = []
    for tag, id in tag_to_id.items():
        ids_list.append(id)
        tags_list.append(tag)

    ids = np.array(ids_list, dtype = np.uint32)
    tags = np.array(tags_list, dtype = np.uint32)

    data_slice = base_data[ids]

    nfs_dir = gt_dir(ds, runbook_path)            # shared NFS: only final GT lands here
    os.makedirs(nfs_dir, exist_ok=True)
    os.makedirs(local_scratch, exist_ok=True)     # node-local: large intermediates live here

    pid = os.getpid()

    # The large per-step base slice and the GT tool's temp output are written to
    # node-local scratch (NOT the shared filesystem) so workers don't each
    # park a multi-GB .data file on NFS at once. Only the final, comparatively
    # small .gt100 is published to NFS.
    data_file = os.path.join(local_scratch, f'step{step}.data.{pid}')
    gt_tmp    = os.path.join(local_scratch, f'step{step}.gt100.{pid}')

    # Final outputs on NFS. The .tags file is written here because the GT tool
    # consumes it, then removed once the GT is computed.
    gt_file   = os.path.join(nfs_dir, f'step{step}.gt100')
    tags_file = os.path.join(nfs_dir, f'step{step}.tags')

    published = False
    try:
        with open(tags_file, 'wb') as tf:
            one = 1
            tf.write(tags.size.to_bytes(4, byteorder='little'))
            tf.write(one.to_bytes(4, byteorder='little'))
            tags.tofile(tf)
        with open(data_file, 'wb') as f:
            f.write(ids.size.to_bytes(4, byteorder='little')) #npts
            f.write(ds.d.to_bytes(4, byteorder='little'))
            data_slice.tofile(f)

        cmd = gt_cmdline
        cmd += ' --base_file ' + data_file
        cmd += ' --gt_file ' + gt_tmp
        cmd += ' --tags_file ' + tags_file
        print("Executing cmdline: ", cmd)
        rc = os.system(cmd)

        # Publish the GT only on success. gt_tmp is on local disk and gt_file is
        # on NFS (different filesystems), so os.replace would raise EXDEV; use
        # shutil.move, which copies then unlinks across filesystems. Each step
        # has a single claimant, so the brief non-atomic window is harmless.
        if rc == 0 and os.path.exists(gt_tmp):
            import shutil
            shutil.move(gt_tmp, gt_file)
            published = True
        else:
            print(f"GT tool failed (rc={rc}); not publishing step {step}")
    finally:
        # Always remove the local intermediates, even on failure, so a crash
        # mid-write can't leak large files onto local scratch.
        for f in (data_file, gt_tmp):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass
        # Remove the .tags file once the GT has been published. Kept on failure
        # (no gt_file) for debugging; a later recompute rewrites it from scratch.
        if published:
            try:
                os.remove(tags_file)
            except OSError:
                pass

    return published
    

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        help=f'Dataset to benchmark on.',
        required=True)
    parser.add_argument(
        '--runbook_file',
        help='Runbook yaml file path'
    )
    parser.add_argument(
        '--private_query',
        action='store_true'
    )
    parser.add_argument(
        '--gt_cmdline_tool',
        required=True
    )
    parser.add_argument(
        '--download',
        action='store_true'
    )
    parser.add_argument(
        '--local_scratch',
        default='/tmp',
        help='Node-local directory for large intermediate files (kept off the '
             'shared filesystem). Must be local disk with room for one base slice.')
    parser.add_argument(
        '--stale_lock_minutes',
        type=float,
        default=120.0,
        help='A claim lock older than this with no valid GT is considered '
             'abandoned (owner died) and may be reclaimed by another worker. '
             'Set comfortably above the time a single step takes to compute.')
    # Kept for backward compatibility / informational logging only. Work is now
    # claimed dynamically, so these no longer partition the steps.
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Informational only; work is claimed dynamically.')
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Informational only; work is claimed dynamically.')
    args = parser.parse_args()

    ds = DATASETS[args.dataset]()
    max_pts, runbook = load_runbook(args.dataset, ds.nb, args.runbook_file)
    query_file = ds.qs_fn if args.private_query else ds.qs_fn
    
    common_cmd = args.gt_cmdline_tool + ' --dist_fn ' 
    match ds.distance():
        case 'euclidean':
            common_cmd += 'l2'
        case 'ip':
            common_cmd += 'mips'
        case _:
            raise RuntimeError('Invalid metric')
    common_cmd += ' --data_type '
    match ds.dtype:
        case 'float32':
            common_cmd += 'float'
        case 'int8':
            common_cmd += 'int8'
        case 'uint8':
            common_cmd += 'uint8'
        case _:
            raise RuntimeError('Invalid datatype')
    common_cmd += ' --K 100'
    common_cmd += ' --query_file ' + os.path.join(ds.basedir, query_file)

    print(f"Running as worker {args.worker_id} of {args.num_workers} "
          f"(dynamic claiming; ids are informational).")
    me = f"{socket.gethostname()}:{os.getpid()}"

    resume_step = find_resume_step(ds, args.runbook_file)
    if resume_step > 0:
        print(f"Found existing GT outputs through step {resume_step}. Will check each step and fill in any gaps.")
    else:
        print("No prior GT outputs found. Starting from step 1.")

    num_queries = get_num_queries(ds)
    if num_queries is None:
        print("Warning: Could not determine number of queries. Skipping file validation.")

    # Read the base data once per worker. It is constant for the whole run, so
    # loading it here instead of inside output_gt avoids re-reading it on every
    # step. output_gt slices the per-step working set out of this array.
    base_data = ds.get_data_in_range(0, ds.nb)

    stale_after_s = args.stale_lock_minutes * 60.0

    def is_done(gt_file):
        return (validate_gt_file(gt_file, num_queries, k=100)
                if num_queries else os.path.exists(gt_file))

    step = 1
    claimed_count = 0

    for entry in runbook:
        # the first step must be an insertion
        if step == 1:
            tag_to_id = get_range_start_end(entry, {})
        else:
            tag_to_id = get_next_set(tag_to_id, entry)

        if entry['operation'] == 'search':
            # Dynamic work-stealing: every worker replays the full runbook (the
            # cheap part) and reconstructs tag_to_id, then tries to CLAIM each
            # unfinished search step. Claiming is an atomic O_EXCL lock create on
            # NFS, so exactly one worker computes any given step -- no duplicated
            # work -- while fast machines naturally grab more steps than slow
            # ones. No static partition, so nobody sits idle while work remains.
            gt_file = os.path.join(gt_dir(ds, args.runbook_file), f'step{step}.gt100')
            lock_file = lock_path_for(gt_file)

            if is_done(gt_file):
                # Already finished by someone (or a prior run). Nothing to do.
                pass
            else:
                # Reclaim an abandoned lock: present, old, and no valid GT.
                if os.path.exists(lock_file) and lock_is_stale(lock_file, stale_after_s):
                    try:
                        os.remove(lock_file)
                        print(f"[{me}] Reclaimed stale lock for step {step}")
                    except OSError:
                        pass  # someone else reclaimed/recreated it; fine

                if try_claim(lock_file):
                    # We own this step. Re-check in case it completed between the
                    # is_done() test and the claim (another worker finishing).
                    if is_done(gt_file):
                        # Already done; drop our lock and move on.
                        try:
                            os.remove(lock_file)
                        except OSError:
                            pass
                    else:
                        print(f"[{me}] Claimed step {step}")
                        ok = output_gt(ds, base_data, tag_to_id, step, common_cmd,
                                       args.runbook_file, local_scratch=args.local_scratch)
                        claimed_count += 1
                        if ok:
                            # Leave the lock in place as a tombstone; is_done()
                            # gates future workers, and a present lock + valid GT
                            # is unambiguous. (Optional: remove it to keep the
                            # dir tidy -- safe because is_done() now returns True.)
                            try:
                                os.remove(lock_file)
                            except OSError:
                                pass
                        else:
                            # Failed to produce a valid GT: release the lock so
                            # another worker (or a later pass) can retry the step.
                            try:
                                os.remove(lock_file)
                            except OSError:
                                pass
                # else: another worker holds a fresh lock; skip and move on.
        step += 1

    print(f"[{me}] Done. Computed {claimed_count} step(s) this pass.")

if __name__ == '__main__':
    main()
