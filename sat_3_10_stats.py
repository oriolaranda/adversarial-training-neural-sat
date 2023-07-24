import pickle
import os
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

DATA_PATH = Path(".")


def main():
    batch_size = 32
    path = DATA_PATH / "SAT-3-10"
    data_names = os.listdir(path)
    data_names = [n for n in data_names if not n.startswith('lt_')]

    n_clauses = {k: [] for k in range(3, 10+1)}
    for name in tqdm(data_names):
        item = pickle.load(open(path / name, "rb"))
        n_clauses[item['n_vars']].append(item['n_clauses'])

    res = {
        'n_samples': {k: len(v) for k, v in n_clauses.items()},
        'n_clauses_mean': {k: np.mean(v) for k,v in n_clauses.items()},
        'n_clauses_std': {k: np.std(v) for k,v in n_clauses.items()},
    }
    assert all(res['n_samples'].values())

    with open('stats-sat-3-10.json', 'w') as f:
        json.dump(res, f)



if __name__ == '__main__':
    main()
