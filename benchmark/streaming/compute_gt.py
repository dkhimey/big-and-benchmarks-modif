import argparse
import os
import re
import numpy as np

import sys
[sys.path.append(i) for i in ['.', '..']]

from benchmark.datasets import DATASETS
from benchmark.streaming.load_runbook import load_runbook

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

def output_gt(ds, tag_to_id, step, gt_cmdline, runbook_path):
    ids_list = []
    tags_list = []
    for tag, id in tag_to_id.items():
        ids_list.append(id)
        tags_list.append(tag)

    ids = np.array(ids_list, dtype = np.uint32)
    tags = np.array(tags_list, dtype = np.uint32)


    data = ds.get_data_in_range(0, ds.nb)
    data_slice = data[np.array(ids)]

    dir = gt_dir(ds, runbook_path)
    prefix = os.path.join(dir, 'step') + str(step) 
    os.makedirs(dir, exist_ok=True)

    tags_file = prefix + '.tags'
    data_file = prefix + '.data'
    gt_file = prefix + '.gt100'

    with open(tags_file, 'wb') as tf:
        one = 1
        tf.write(tags.size.to_bytes(4, byteorder='little'))
        tf.write(one.to_bytes(4, byteorder='little'))
        tags.tofile(tf)    
    with open(data_file, 'wb') as f:
        f.write(ids.size.to_bytes(4, byteorder='little')) #npts
        f.write(ds.d.to_bytes(4, byteorder='little'))
        data_slice.tofile(f)
    
    gt_cmdline += ' --base_file ' + data_file 
    gt_cmdline += ' --gt_file ' + gt_file
    gt_cmdline += ' --tags_file ' + tags_file
    print("Executing cmdline: ", gt_cmdline)
    os.system(gt_cmdline)
    print("Removing data file")
    rm_cmdline = "rm " + data_file
    os.system(rm_cmdline)
    

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

    resume_step = find_resume_step(ds, args.runbook_file)
    if resume_step > 0:
        print(f"Found existing GT outputs through step {resume_step}. Will check each step and fill in any gaps.")
    else:
        print("No prior GT outputs found. Starting from step 1.")

    num_queries = get_num_queries(ds)
    if num_queries is None:
        print("Warning: Could not determine number of queries. Skipping file validation.")

    step = 1
    ids = np.empty(0, dtype=np.uint32)

    for entry in runbook:
        # the first step must be an insertion
        if step == 1:
            tag_to_id = get_range_start_end(entry, {})
        else:
            tag_to_id = get_next_set(tag_to_id, entry)
        if (entry['operation'] == 'search'):
            gt_file = os.path.join(gt_dir(ds, args.runbook_file), f'step{step}.gt100')
            is_valid = validate_gt_file(gt_file, num_queries, k=100) if num_queries else os.path.exists(gt_file)
            if is_valid:
                print(f"Skipping step {step}, found valid existing {gt_file}")
            else:
                if os.path.exists(gt_file):
                    print(f"Ground truth file for step {step} exists but is incomplete/corrupted. Recomputing.")
                output_gt(ds, tag_to_id, step, common_cmd, args.runbook_file)
        step += 1

if __name__ == '__main__':
    main()