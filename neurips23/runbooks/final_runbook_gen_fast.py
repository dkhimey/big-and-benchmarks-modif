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

def cluster_and_permute(
    data, num_clusters, batch_size: int = 10_000_000
) -> Tuple[np.ndarray[int], np.ndarray[int]]:
    """
    Cluster the data and return permutation of row indices
    that would group indices of the same cluster together
    """
    npts = np.shape(data)[0]
    print(f"[INFO] Clustering step started: {npts} points into {num_clusters} clusters")
    sample_size = min(100000, npts)
    sample_indices = np.random.choice(range(npts), size=sample_size, replace=False)
    sampled_data = data[sample_indices, :].astype(np.float32, copy=False)
    centroids, sample_labels = kmeans2(sampled_data, num_clusters, minit="++", iter=10)
    centroids = centroids.astype(np.float32, copy=False)
    print("[INFO] Initial clustering on sample completed")

    # Pass 1: compute cluster counts in chunks to avoid allocating a full
    # float32 copy of the whole dataset.
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
    for i in range(0, num_clusters, 1):
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
        permutation:np.ndarray[int],
        output_data_file:str,
        write_batch_size: int = 100_000_000
):
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
            df.write(chunk.tobytes(order='C'))
            if chunk_idx % 10 == 0 or chunk_idx == num_chunks:
                print(f"[INFO] Write pass: chunk {chunk_idx}/{num_chunks} ({end}/{npts})")
    print("[INFO] Permuted dataset write completed")


def create_runbook(
    dataset_str:str,
    offsets:np.ndarray[int],
    permutation:np.ndarray[int],
    num_clusters:int, 
    output_yaml_file:str
):
    print(f"[INFO] Runbook generation started for dataset '{dataset_str}'")
    ins_cursor_start = offsets.copy()
    ins_cursor_end = offsets.copy()

    del_cursor_start = offsets.copy()
    del_cursor_end = offsets.copy()

    operation_list = []
    num_operations = 1
    active_points = 0
    max_pts = 0
    active_points_in_cluster = np.zeros(num_clusters)

    num_rounds = 5
    sample = np.random.default_rng().dirichlet((100,15,10,5,3), num_clusters)
    for c in range(num_clusters):
        np.random.default_rng().shuffle(sample[c])
    print(sample)

    for round in range(num_rounds):
        print(f"[INFO] Starting round {round + 1}/{num_rounds}")
        #insertions
        for c in range(num_clusters):
            delta = (int)((offsets[c+1]-offsets[c]) * sample[c,round])
            ins_cursor_end[c] = ins_cursor_start[c] + delta
            active_points += delta
            max_pts = max(max_pts, active_points)
            active_points_in_cluster[c] += delta
            print('ins [', ins_cursor_start[c], ', ', ins_cursor_end[c], 
                  ') active:', int(active_points_in_cluster[c]),
                  'total:', active_points)
            entry = [{'operation': 'insert'}, {'start': int(ins_cursor_start[c])}, {'end': int(ins_cursor_end[c])}]
            operation_list.append((num_operations, entry))
            num_operations += 1
            operation_list.append((num_operations, [{'operation': str('search')}]))
            num_operations += 1
            ins_cursor_start[c] = ins_cursor_end[c]

        #deletions
        for c in range(num_clusters):
            fraction = random.uniform(0.5,0.9)
            delta = (int)(fraction*(ins_cursor_end[c]-del_cursor_start[c]))
            del_cursor_end[c] = del_cursor_start[c] + delta
            active_points -= delta
            active_points_in_cluster[c] -= delta
            print('del [', del_cursor_start[c], ',', del_cursor_end[c],
                  ') active:', int(active_points_in_cluster[c]),
                  'total:', active_points)
            entry = [{'operation': 'delete'}, {'start': int(del_cursor_start[c])}, {'end': int(del_cursor_end[c])}]
            operation_list.append((num_operations, entry))
            num_operations += 1
            operation_list.append((num_operations, [{'operation': 'search'}]))
            num_operations += 1
            del_cursor_start[c] = del_cursor_end[c]


    print(f"[INFO] Writing runbook YAML to {output_yaml_file}")
    with open(output_yaml_file, 'w') as yf:
        operation_list.sort(key = lambda x: x[0])
        sorted_dict = {}
        sorted_dict['max_pts'] = int(max_pts)
        for (k, v) in operation_list:
            sorted_dict[k]=v
        yaml_object = {}
        yaml_object[dataset_str] = sorted_dict
        yaml.dump(yaml_object, yf)
    print("[INFO] Runbook generation completed")


def create_random_runbook(
    dataset_str: str,
    num_points: int,
    num_clusters: int,
    output_yaml_file: str
):
    """
    Creates a runbook with the same structure as create_runbook (same number of
    insert/delete/search steps, similar batch sizes via Dirichlet sampling) but
    operates on equal-sized contiguous chunks of an already randomly-permuted
    dataset rather than on clusters.
    """
    print(f"[INFO] Random runbook generation started for dataset '{dataset_str}'")

    # Divide the (already randomly permuted) data into num_clusters equal chunks.
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
        # insertions
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

        # deletions
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
        sorted_dict = {}
        sorted_dict['max_pts'] = int(max_pts)
        for (k, v) in operation_list:
            sorted_dict[k] = v
        yaml_object = {}
        yaml_object[dataset_str] = sorted_dict
        yaml.dump(yaml_object, yf)
    print("[INFO] Random runbook generation completed")


def main():
    print("[INFO] final_runbook_gen.py started")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    parser.add_argument(
        '-c', '--num_clusters',
        type=int,
        required=True
    )
    parser.add_argument(
        '-o', '--output_data_file',
        required=False,
        help='Output permuted dataset file. Required for clustered mode; ignored in --random mode.'
    )
    parser.add_argument(
        '-y', '--output_yaml_file',
        required=True
    )
    parser.add_argument(
        '--random',
        action='store_true',
        default=False,
        help='Generate a runbook with randomly distributed batches instead of '
             'clustered batches. Assumes dataset file is already randomly '
             'permuted and only generates YAML runbook steps over equal-sized '
             'chunks, while preserving insert/delete/search step counts and '
             'batch size behavior.'
    )
    args = parser.parse_args()
    if (not args.random) and (not args.output_data_file):
        parser.error('--output_data_file is required unless --random is set')
    print(
        f"[INFO] Arguments parsed: dataset={args.dataset}, "
        f"num_clusters={args.num_clusters}, "
        f"output_data_file={args.output_data_file}, "
        f"output_yaml_file={args.output_yaml_file}"
    )

    ds = DATASETS[args.dataset]()

    if args.random:
        print("[INFO] Random mode: assuming existing dataset file is already randomly permuted")
        npts = ds.nb
        print(f"[INFO] Dataset size from metadata: {npts} points")

        print("[INFO] Starting random runbook creation stage")
        create_random_runbook(dataset_str=args.dataset,
                              num_points=npts,
                              num_clusters=args.num_clusters,
                              output_yaml_file=args.output_yaml_file)
    else:
        print(f"[INFO] Loading dataset '{args.dataset}'")
        if hasattr(ds, 'get_data_in_range'):
            # Force load into RAM to avoid memmap random-access overhead.
            data = np.array(ds.get_data_in_range(0, ds.nb))
        elif ds.nb <= 10**7:
            data = ds.get_dataset()
        else:
            # Fallback path for datasets without range API.
            data = next(ds.get_dataset_iterator(bs=ds.nb))
        print(f"[INFO] Dataset loaded with shape {np.shape(data)}")

        print("[INFO] Starting cluster + permutation stage")
        offsets, permutation = cluster_and_permute(data, args.num_clusters)
        print(f"[INFO] Permutation generated with {len(permutation)} indices")

        print("[INFO] Starting permuted data write stage")
        write_permuated_data(data=data,
                             permutation=permutation,
                             output_data_file=args.output_data_file)

        print("[INFO] Starting runbook creation stage")
        create_runbook(dataset_str=args.dataset,
                       offsets=offsets,
                       permutation=permutation,
                       num_clusters=args.num_clusters,
                       output_yaml_file=args.output_yaml_file)
    print("[INFO] final_runbook_gen.py finished successfully")


if __name__ == '__main__':
    main()
