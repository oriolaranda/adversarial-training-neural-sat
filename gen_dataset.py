import os
import numpy as np
import random
import pickle
import argparse
import json
from itertools import chain, repeat, cycle, zip_longest
import shutil
from gen_utils import cnf_to_csat
from tqdm import tqdm
import torch_sparse
from data import SATDataset
from pathlib import Path
from sklearn.model_selection import train_test_split



def gen_random_solution(n_vars):
    solution = [x if np.random.random() < 0.5 else -x for x in range(1, n_vars+1)]
    return solution


def gen_random_k_clause(solution, literals, k):
    """ Generates a k-clause in a random manner. A k-clause is defined as (l_1 v ... v l_k)
        with l_1 being sat."""
    true_literal = np.random.choice(solution)
    other_literals = np.random.choice(list(literals - {true_literal}), size=k-1, replace=True)
    clause = np.concatenate([[true_literal], other_literals])
    return clause


def sample_k(n_vars, p_k_2, p_geo):
    while True:
        k_base = 1 if np.random.random() < p_k_2 else 2
        k = min(n_vars, k_base + np.random.geometric(p_geo))
        yield k


def generate_problem(n_vars, k, p_k_2, p_geo, stats_sat_3_10):
    literals = {x for x in range(-n_vars, n_vars+1) if x != 0}
    solution = gen_random_solution(n_vars)

    if not k: # sat-3-10
        mean = stats_sat_3_10['n_clauses_mean'][str(n_vars)]
        std = stats_sat_3_10['n_clauses_std'][str(n_vars)]
        n_clauses = np.max([1, int(np.random.normal(loc=mean, scale=std))])
        gen_ks = sample_k(n_vars, p_k_2, p_geo)

    else: # k-sat-3-10
        assert k > 2, "k-SAT must be k >= 3 to be useful"
        ratio = {
            2: 1.0,
            3: 4.27,
            4: 9.93,
            5: 21.12,
            6: 43.37,
            7: 87.79,
            8: 176.54,
            9: 354.01,
            10: 708.91,
        }
        n_clauses = int(n_vars*ratio[k]) # alpha_k = n_clauses/n_vars
        gen_ks = cycle([k])

    clauses = []
    for _, k in zip(range(n_clauses), gen_ks):
        clause = gen_random_k_clause(solution, literals, k)
        clauses.append(clause)

    adj, ind, mask = cnf_to_csat(clauses, n_vars)
    csat_problem = {
        'clauses': clauses,
        'n_clauses': len(clauses),
        'n_vars': n_vars,
        'label': 1,
        'solution': solution,
        'adj': torch_sparse.SparseTensor.from_dense(adj),
        'ind': ind,
        'mask': mask
    }
    return csat_problem


def generate_dataset(args, out_dir, dataset_name):
    with open('stats-sat-3-10.json', 'r') as f:
        stats_sat_3_10 = json.load(f)

    repeat_n_vars = args.n_problems // (args.max_n_vars - args.min_n_vars + 1)
    assert repeat_n_vars > 0, "repeat_n_vars = 0: Not enough samples"

    if not args.force_write:
        assert not os.path.exists(out_dir), f'WARNING: Overwritting dataset "{dataset_name}"'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    print(f"Data Generator for {dataset_name}:", f"\n{'-'*35}")
    for k, v in vars(args).items():
        if k not in ['force_write']:
            print(f'{k} = {v}') # exclude useless args
    print(f"{'-'*35}\n")

    if not args.k:  # sat-3-10
        list_n_vars = chain.from_iterable(list(repeat(n_vars, repeat_n_vars))
                                          for n_vars in range(args.min_n_vars, args.max_n_vars + 1))
        list_n_vars_k = list(zip_longest(list_n_vars, []))  # default value None -> k=None
    else:
        ks = [3, 4, 5] if args.k == 'small' else [6, 7, 8]
        pairs_n_vars_k = (list(zip(repeat(n_vars, repeat_n_vars), cycle(ks)))
                          for n_vars in range(args.min_n_vars, args.max_n_vars + 1))
        list_n_vars_k = list(chain.from_iterable(pairs_n_vars_k))

    for idx, (n_vars, k) in tqdm(enumerate(list_n_vars_k)):
        p = generate_problem(n_vars, k, args.p_k_2, args.p_geo, stats_sat_3_10)
        problem_name = f"{out_dir}/{dataset_name}--{str(idx)}_true.pkl"
        with open(problem_name, 'wb') as f:
            pickle.dump(p, f)


def get_sat_datasets(dataset_name, seed, split_size=0.16666):
    data_path = Path("../datasets") / dataset_name
    data_names = [d for d in os.listdir(data_path) if not d.startswith('lt_')]
    train_idx, val_idx = train_test_split(np.arange(len(data_names)), test_size=split_size, random_state=seed)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5, random_state=seed)
    train = SATDataset(data_path, train_idx[:32*10])
    val = SATDataset(data_path, val_idx[:32*3])
    test = SATDataset(data_path, test_idx)
    print(test_idx)
    print(test.data_names)
    return train, val, test


def generate_batch(batch_size, n_vars, k, p_k_2, p_geo, stats):
    problems = [generate_problem(n_vars, k, p_k_2, p_geo, stats) for _ in range(batch_size)]
    batch = CircuitSAT.collate_fn(problems)
    return batch


def send_to(batch, device):
    batch['adj'] = batch['adj'].to(device)
    batch['is_sat'] = batch['is_sat'].to(device)
    batch['features'] = batch['features'].to(device)
    return batch


def check_dataset_stats(args, out_dir):
    train_idxs = np.arange(args.n_problems)
    sat = SATDataset(Path(out_dir), train_idxs)
    l_ks, l_n_vars, l_n_clauses = [], [], []
    for i in tqdm(range(len(sat))):
        ks, rs = np.unique([len(c) for c in sat[i]['clauses']], return_counts=True)
        l_ks.append(ks)
        l_n_vars.append(sat[i]['n_vars'])
        l_n_clauses.append(sat[i]['n_clauses'])
    l_ks = list(chain(*l_ks))
    counts_ks = np.unique(l_ks, return_counts=True)
    counts_n_vars = np.unique(l_n_vars, return_counts=True)
    counts_n_clauses = np.unique(l_n_clauses, return_counts=True)
    dic_ks = {k: v for k, v in zip(*counts_ks)}
    dic_n_vars = {k: v for k, v in zip(*counts_n_vars)}
    dic_n_clauses = {k: v for k, v in zip(*counts_n_clauses)}
    print("n_problems ->", len(sat))
    print("k : counts ->", dic_ks)
    print("n_vars : counts ->", dic_n_vars)
    print("n_clauses : counts ->", dic_n_clauses)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    assert args.k in [None, "small"] # Big not implemented
    str_sat = f"K-SAT" if args.k else "SAT"
    dataset_name = f"RND-{str_sat}-{args.min_n_vars}-{args.max_n_vars}"
    out_dir = f"../datasets/{dataset_name}"

    generate_dataset(args, out_dir, dataset_name)
    check_dataset_stats(args, out_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SR(U(start, end)) data')
    parser.add_argument('-n_problems', default=300000, help='How many problem pairs to generate', type=int)
    parser.add_argument('-min_n_vars', default=3, help='Start value for number of variables', type=int)
    parser.add_argument('-max_n_vars', default=10, help='End value for number of variables', type=int)
    parser.add_argument('-k', default=None, type=str)
    parser.add_argument('-p_k_2', default=0.3, type=float)
    parser.add_argument('-p_geo', default=0.4, type=float)
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-f', '-force_write', dest='force_write', default=False, action='store_true')
    parsed_args = parser.parse_args()

    main(parsed_args)
