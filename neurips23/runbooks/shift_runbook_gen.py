"""
Hard-shift runbook generator: a drop-in replacement for the soft-clustered
generator that produces genuine support shift across rounds.

Key behavioral difference from the soft-clustered generator:
    - Soft clustered (final_runbook_gen.py): every cluster receives some
      Dirichlet-weighted inserts in every round, so the *support* of the
      active dataset is constant after round 1 and only the per-cluster
      density varies.
    - Shift (this script): clusters are randomly partitioned into disjoint
      cohorts, one per round. Cluster c in cohort r is inserted only during
      round r and heavily evicted during round r+1. Outside that two-round
      window the cluster contributes no points to the active dataset, so the
      active support visibly changes between rounds.

The CLI is preserved from the original generator: --dataset, -c, -o, -y, and
--random behave identically. New optional flags (--num_rounds,
--num_sub_batches, --delete_fraction_min/max) tune the shift schedule and
default to values that match the operation-count scale of the original
generator. The --random mode produces the same uniform control runbook as
the original, unchanged.
"""

import argparse
import os
import sys
import numpy as np
import random
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scipy.cluster.vq import vq, kmeans2
from typing import Tuple
from benchmark.datasets import DATASETS


# ---------------------------------------------------------------------------
# Clustering and dataset writing: identical to the original generator.
# ---------------------------------------------------------------------------

def cluster_and_permute(
    data, num_clusters, batch_size: int = 25_000_000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster the data and return permutation of row indices that would group
    indices of the same cluster together. Identical to the original generator.
    """
    npts = np.shape(data)[0]
    print(f"[INFO] Clustering step started: {npts} points into {num_clusters} clusters")
    sample_size = min(100000, npts)
    sample_indices = np.random.choice(range(npts), size=sample_size, replace=False)
    sampled_data = data[sample_indices, :].astype(np.float32, copy=False)
    centroids, sample_labels = kmeans2(sampled_data, num_clusters, minit="++", iter=10)
    centroids = centroids.astype(np.float32, copy=False)
    print("[INFO] Initial clustering on sample completed")

    # Pass 1: cluster counts in chunks (avoids a full float32 copy of the data).
    count = np.zeros(num_clusters, dtype=np.int64)
    num_chunks = (npts + batch_size - 1) // batch_size
    for chunk_idx, start in enumerate(range(0, npts, batch_size), start=1):
        end = min(start + batch_size, npts)
        chunk = np.asarray(data[start:end], dtype=np.float32)
        labels_chunk, _ = vq(chunk, centroids)
        count += np.bincount(labels_chunk, minlength=num_clusters)
        if chunk_idx % 10 == 0 or chunk_idx == num_chunks:
            print(f"[INFO] Count pass: chunk {chunk_idx}/{num_chunks} ({end}/{npts})")

    print("Cluster counts")
    print(count)

    offsets = np.zeros(num_clusters + 1, dtype=int)
    for i in range(num_clusters):
        offsets[i + 1] = offsets[i] + count[i]

    # Pass 2: build the permutation in chunks.
    perm_dtype = np.uint32 if npts <= np.iinfo(np.uint32).max else np.int64
    permutation = np.empty(npts, dtype=perm_dtype)
    counters = np.zeros(num_clusters, dtype=np.int64)
    for chunk_idx, start in enumerate(range(0, npts, batch_size), start=1):
        end = min(start + batch_size, npts)
        chunk = np.asarray(data[start:end], dtype=np.float32)
        labels_chunk, _ = vq(chunk, centroids)
        global_indices = np.arange(start, end, dtype=perm_dtype)

        for c in range(num_clusters):
            mask = labels_chunk == c
            k = int(mask.sum())
            if k == 0:
                continue
            dst_start = offsets[c] + counters[c]
            dst_end = dst_start + k
            permutation[dst_start:dst_end] = global_indices[mask]
            counters[c] += k

        if chunk_idx % 10 == 0 or chunk_idx == num_chunks:
            print(f"[INFO] Permutation pass: chunk {chunk_idx}/{num_chunks} ({end}/{npts})")

    print("[INFO] Clustering step completed")
    return offsets, permutation


def write_permuated_data(
    data,
    permutation: np.ndarray,
    output_data_file: str,
    write_batch_size: int = 100_000_000,
):
    """Write the permuted dataset to disk. Identical to the original generator."""
    print(f"[INFO] Writing permuted dataset to {output_data_file}")
    shape = np.shape(data)
    with open(output_data_file, 'wb') as df:
        df.write(shape[0].to_bytes(4, 'little'))
        df.write(shape[1].to_bytes(4, 'little'))

        npts = shape[0]
        num_chunks = (npts + write_batch_size - 1) // write_batch_size
        for chunk_idx, start in enumerate(range(0, npts, write_batch_size), start=1):
            end = min(start + write_batch_size, npts)
            idx = permutation[start:end]
            chunk = np.ascontiguousarray(data[idx, :], dtype=data.dtype)
            chunk.tofile(df)
            # df.write(chunk.tobytes(order='C'))
            if chunk_idx % 10 == 0 or chunk_idx == num_chunks:
                print(f"[INFO] Write pass: chunk {chunk_idx}/{num_chunks} ({end}/{npts})")
    print("[INFO] Permuted dataset write completed")


# ---------------------------------------------------------------------------
# Shift runbook: replaces the soft-clustered create_runbook from the original.
# ---------------------------------------------------------------------------

def create_shift_runbook(
    dataset_str: str,
    offsets: np.ndarray,
    num_clusters: int,
    output_yaml_file: str,
    num_rounds: int = 5,
    num_sub_batches: int = 4,
    delete_fraction_range: Tuple[float, float] = (0.85, 1.0),
):
    """
    Generate a runbook in which the active dataset's *support* shifts across
    rounds.

    Clusters are randomly partitioned into ``num_rounds`` disjoint cohorts.
    For cluster c assigned to cohort r:
      - Inserts happen ONLY during round r, split into ``num_sub_batches``
        sub-batches whose fractional sizes are sampled from a shuffled
        Dirichlet (matching the size distribution used by the original
        soft-clustered generator). A search step is interleaved after every
        insert sub-batch.
      - A single delete batch happens during round r+1, removing a fraction of
        c's points drawn uniformly from ``delete_fraction_range`` (default
        0.85--1.0). A search step follows the delete.

    Outside its insert and evict rounds, a cluster contributes no operations.
    The active set's composition therefore changes from round to round, in
    contrast to the soft-clustered runbook which keeps every cluster
    represented at every search step.

    The last cohort (round ``num_rounds - 1``) is never explicitly evicted;
    its points remain in the active set when the runbook ends.
    """
    print(f"[INFO] Shift runbook generation started for dataset '{dataset_str}'")
    print(f"[INFO] num_clusters={num_clusters}, num_rounds={num_rounds}, "
          f"num_sub_batches={num_sub_batches}, "
          f"delete_fraction_range={delete_fraction_range}")

    rng = np.random.default_rng()

    # 1. Random cohort assignment so cohort identity is decoupled from cluster
    #    index ordering. np.array_split tolerates num_clusters % num_rounds != 0.
    cluster_order = rng.permutation(num_clusters)
    cohorts = [np.asarray(c, dtype=int) for c in np.array_split(cluster_order, num_rounds)]
    cohort_of = np.full(num_clusters, -1, dtype=int)
    for r, members in enumerate(cohorts):
        cohort_of[members] = r
    print(f"[INFO] Cohort sizes: {[len(c) for c in cohorts]}")

    # 2. Per-cluster sub-batch weights, mirroring the Dirichlet pattern used by
    #    the soft-clustered generator (sum to 1, dominated by one large batch).
    base_alpha = (100, 15, 10, 5, 3)
    if num_sub_batches <= len(base_alpha):
        alpha = base_alpha[:num_sub_batches]
    else:
        alpha = base_alpha + (3,) * (num_sub_batches - len(base_alpha))
    sub_batch_weights = rng.dirichlet(alpha, num_clusters)
    for c in range(num_clusters):
        rng.shuffle(sub_batch_weights[c])

    # 3. Cursors. ins_cursor[c] is the next index to insert for cluster c;
    #    del_cursor[c] is the next index to delete; cluster_end[c] is c's
    #    exclusive end in the permuted dataset.
    ins_cursor = offsets[:-1].astype(np.int64).copy()
    del_cursor = offsets[:-1].astype(np.int64).copy()
    cluster_end = offsets[1:].astype(np.int64).copy()

    # 4. Generate operations.
    operation_list = []
    num_operations = 1
    active_points = 0
    max_pts = 0
    active_in_cluster = np.zeros(num_clusters, dtype=np.int64)

    for r in range(num_rounds):
        # ---- Insertion phase: cohort r ----
        print(f"[INFO] Round {r + 1}/{num_rounds}: inserting cohort {r} "
              f"({len(cohorts[r])} clusters)")
        for sb in range(num_sub_batches):
            is_last = (sb == num_sub_batches - 1)
            for c in cohorts[r]:
                cluster_size = int(cluster_end[c] - offsets[c])
                if is_last:
                    # Drain everything remaining for cluster c on the last
                    # sub-batch, so integer-truncation losses don't accumulate.
                    new_end = int(cluster_end[c])
                else:
                    delta = int(cluster_size * sub_batch_weights[c, sb])
                    new_end = min(int(ins_cursor[c]) + delta, int(cluster_end[c]))
                actual_delta = new_end - int(ins_cursor[c])
                if actual_delta <= 0:
                    continue
                active_points += actual_delta
                max_pts = max(max_pts, active_points)
                active_in_cluster[c] += actual_delta
                print(f"  ins c={c} sb={sb}: [{int(ins_cursor[c])}, {new_end}) "
                      f"+{actual_delta} active_in_cluster={int(active_in_cluster[c])} "
                      f"total={active_points}")
                entry = [
                    {'operation': 'insert'},
                    {'start': int(ins_cursor[c])},
                    {'end': int(new_end)},
                ]
                operation_list.append((num_operations, entry))
                num_operations += 1
                operation_list.append((num_operations, [{'operation': 'search'}]))
                num_operations += 1
                ins_cursor[c] = new_end

        # ---- Deletion phase: evict cohort r-1 (skip in round 0) ----
        if r > 0:
            prev = r - 1
            print(f"[INFO] Round {r + 1}/{num_rounds}: evicting cohort {prev} "
                  f"({len(cohorts[prev])} clusters)")
            for c in cohorts[prev]:
                available = int(ins_cursor[c] - del_cursor[c])
                if available <= 0:
                    continue
                fraction = random.uniform(*delete_fraction_range)
                delta = int(fraction * available)
                if delta <= 0:
                    continue
                new_del_end = int(del_cursor[c]) + delta
                active_points -= delta
                active_in_cluster[c] -= delta
                print(f"  del c={c}: [{int(del_cursor[c])}, {new_del_end}) "
                      f"-{delta} active_in_cluster={int(active_in_cluster[c])} "
                      f"total={active_points}")
                entry = [
                    {'operation': 'delete'},
                    {'start': int(del_cursor[c])},
                    {'end': int(new_del_end)},
                ]
                operation_list.append((num_operations, entry))
                num_operations += 1
                operation_list.append((num_operations, [{'operation': 'search'}]))
                num_operations += 1
                del_cursor[c] = new_del_end

    # 5. Write YAML in the same format as the original generator.
    print(f"[INFO] Writing shift runbook YAML to {output_yaml_file}")
    with open(output_yaml_file, 'w') as yf:
        operation_list.sort(key=lambda x: x[0])
        sorted_dict = {'max_pts': int(max_pts)}
        for k, v in operation_list:
            sorted_dict[k] = v
        yaml_object = {dataset_str: sorted_dict}
        yaml.dump(yaml_object, yf)
    print(f"[INFO] Shift runbook generation completed: {num_operations - 1} operations, "
          f"max_pts={max_pts}")


# ---------------------------------------------------------------------------
# Random / uniform runbook: preserved unchanged from the original.
# ---------------------------------------------------------------------------

def create_random_runbook(
    dataset_str: str,
    num_points: int,
    num_clusters: int,
    output_yaml_file: str,
):
    """
    Uniform-control runbook over equal-sized contiguous chunks of an already
    randomly-permuted dataset. Identical to the original generator.
    """
    print(f"[INFO] Random runbook generation started for dataset '{dataset_str}'")

    offsets = np.linspace(0, num_points, num_clusters + 1, dtype=int)
    print(f"[INFO] Equal-sized chunk offsets: min_chunk={int(np.diff(offsets).min())}, "
          f"max_chunk={int(np.diff(offsets).max())}")

    ins_cursor_start = offsets.copy()
    ins_cursor_end = offsets.copy()
    del_cursor_start = offsets.copy()
    del_cursor_end = offsets.copy()

    operation_list = []
    num_operations = 1
    active_points = 0
    max_pts = 0
    active_points_in_chunk = np.zeros(num_clusters)

    num_rounds = 5
    sample = np.random.default_rng().dirichlet((100, 15, 10, 5, 3), num_clusters)
    for c in range(num_clusters):
        np.random.default_rng().shuffle(sample[c])
    print(sample)

    for round_idx in range(num_rounds):
        print(f"[INFO] Starting round {round_idx + 1}/{num_rounds}")
        for c in range(num_clusters):
            delta = int((offsets[c + 1] - offsets[c]) * sample[c, round_idx])
            ins_cursor_end[c] = ins_cursor_start[c] + delta
            active_points += delta
            max_pts = max(max_pts, active_points)
            active_points_in_chunk[c] += delta
            print('ins [', ins_cursor_start[c], ',', ins_cursor_end[c],
                  ') active:', int(active_points_in_chunk[c]),
                  'total:', active_points)
            entry = [{'operation': 'insert'}, {'start': int(ins_cursor_start[c])}, {'end': int(ins_cursor_end[c])}]
            operation_list.append((num_operations, entry))
            num_operations += 1
            operation_list.append((num_operations, [{'operation': 'search'}]))
            num_operations += 1
            ins_cursor_start[c] = ins_cursor_end[c]

        for c in range(num_clusters):
            fraction = random.uniform(0.5, 0.9)
            delta = int(fraction * (ins_cursor_end[c] - del_cursor_start[c]))
            del_cursor_end[c] = del_cursor_start[c] + delta
            active_points -= delta
            active_points_in_chunk[c] -= delta
            print('del [', del_cursor_start[c], ',', del_cursor_end[c],
                  ') active:', int(active_points_in_chunk[c]),
                  'total:', active_points)
            entry = [{'operation': 'delete'}, {'start': int(del_cursor_start[c])}, {'end': int(del_cursor_end[c])}]
            operation_list.append((num_operations, entry))
            num_operations += 1
            operation_list.append((num_operations, [{'operation': 'search'}]))
            num_operations += 1
            del_cursor_start[c] = del_cursor_end[c]

    print(f"[INFO] Writing random runbook YAML to {output_yaml_file}")
    with open(output_yaml_file, 'w') as yf:
        operation_list.sort(key=lambda x: x[0])
        sorted_dict = {'max_pts': int(max_pts)}
        for k, v in operation_list:
            sorted_dict[k] = v
        yaml_object = {dataset_str: sorted_dict}
        yaml.dump(yaml_object, yf)
    print("[INFO] Random runbook generation completed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[INFO] shift_runbook_gen.py started")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate a hard-shift runbook (default) or a uniform-control "
            "runbook (--random). Drop-in replacement for the soft-clustered "
            "generator."
        ),
    )

    # CLI preserved from the original generator.
    parser.add_argument('--dataset', choices=DATASETS.keys(), required=True)
    parser.add_argument('-c', '--num_clusters', type=int, required=True)
    parser.add_argument(
        '-o', '--output_data_file',
        required=False,
        help='Output permuted dataset file. Required for clustered (default) mode; '
             'ignored in --random mode.',
    )
    parser.add_argument('-y', '--output_yaml_file', required=True)
    parser.add_argument(
        '--random',
        action='store_true', default=False,
        help='Generate a uniform-control runbook over equal-sized chunks of an '
             'already randomly-permuted dataset, instead of a hard-shift runbook. '
             'Identical to the --random mode of the original generator.',
    )

    # Shift-mode tuning knobs (used only when --random is NOT set).
    parser.add_argument(
        '--num_rounds', type=int, default=5,
        help='Number of cohorts/rounds for shift mode.',
    )
    parser.add_argument(
        '--num_sub_batches', type=int, default=4,
        help='Number of insert sub-batches per cluster during its active round.',
    )
    parser.add_argument(
        '--delete_fraction_min', type=float, default=0.85,
        help='Lower bound on per-cluster eviction fraction.',
    )
    parser.add_argument(
        '--delete_fraction_max', type=float, default=1.0,
        help='Upper bound on per-cluster eviction fraction.',
    )

    args = parser.parse_args()
    if (not args.random) and (not args.output_data_file):
        parser.error('--output_data_file is required unless --random is set')
    if args.num_rounds < 1:
        parser.error('--num_rounds must be >= 1')
    if args.num_sub_batches < 1:
        parser.error('--num_sub_batches must be >= 1')
    if not (0.0 <= args.delete_fraction_min <= args.delete_fraction_max <= 1.0):
        parser.error('--delete_fraction_min/max must satisfy 0 <= min <= max <= 1')
    if args.num_clusters < args.num_rounds and not args.random:
        print(f"[WARN] num_clusters ({args.num_clusters}) < num_rounds "
              f"({args.num_rounds}); some cohorts will be empty.")

    print(
        f"[INFO] Arguments parsed: dataset={args.dataset}, "
        f"num_clusters={args.num_clusters}, "
        f"output_data_file={args.output_data_file}, "
        f"output_yaml_file={args.output_yaml_file}, "
        f"random={args.random}, num_rounds={args.num_rounds}, "
        f"num_sub_batches={args.num_sub_batches}, "
        f"delete_fraction=[{args.delete_fraction_min}, {args.delete_fraction_max}]"
    )

    ds = DATASETS[args.dataset]()

    if args.random:
        print("[INFO] Random mode: assuming existing dataset file is already randomly permuted")
        npts = ds.nb
        print(f"[INFO] Dataset size from metadata: {npts} points")
        create_random_runbook(
            dataset_str=args.dataset,
            num_points=npts,
            num_clusters=args.num_clusters,
            output_yaml_file=args.output_yaml_file,
        )
    else:
        print(f"[INFO] Loading dataset '{args.dataset}'")
        if hasattr(ds, 'get_data_in_range'):
            data = np.array(ds.get_data_in_range(0, ds.nb))
        elif ds.nb <= 10**7:
            data = ds.get_dataset()
        else:
            data = next(ds.get_dataset_iterator(bs=ds.nb))
        print(f"[INFO] Dataset loaded with shape {np.shape(data)}")

        print("[INFO] Starting cluster + permutation stage")
        offsets, permutation = cluster_and_permute(data, args.num_clusters)
        print(f"[INFO] Permutation generated with {len(permutation)} indices")

        print("[INFO] Starting permuted data write stage")
        write_permuated_data(
            data=data,
            permutation=permutation,
            output_data_file=args.output_data_file,
        )

        print("[INFO] Starting shift runbook creation stage")
        create_shift_runbook(
            dataset_str=args.dataset,
            offsets=offsets,
            num_clusters=args.num_clusters,
            output_yaml_file=args.output_yaml_file,
            num_rounds=args.num_rounds,
            num_sub_batches=args.num_sub_batches,
            delete_fraction_range=(args.delete_fraction_min, args.delete_fraction_max),
        )
    print("[INFO] shift_runbook_gen.py finished successfully")


if __name__ == '__main__':
    main()
