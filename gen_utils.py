import random
import torch
import numpy as np
import torch_sparse
from pysat.solvers import Minisat22


def clause_to_nsat_adj(iclauses, n_vars):
    n_cells = sum([len(iclause) for iclause in iclauses])
    # construct adjacency for neurosat model
    nsat_indices = np.zeros([n_cells, 2], dtype=np.int64)
    cell = 0
    for clause_idx, iclause in enumerate(iclauses):
        vlits = [ilit_to_vlit(x, n_vars) for x in iclause]
        for vlit in vlits:
            nsat_indices[cell, :] = [vlit, clause_idx]
            cell += 1
    assert(cell == n_cells)
    adj_nsat = torch.sparse.FloatTensor(torch.Tensor(nsat_indices).T.long(), torch.ones(n_cells), 
                                       torch.Size([n_vars*2, len(iclauses)]))
    return adj_nsat

def ilit_to_var_sign(x):
    assert(abs(x) > 0)
    var = abs(x) - 1
    sign = x < 0
    return var, sign

def ilit_to_vlit(x, n_vars):
    assert(x != 0)
    var, sign = ilit_to_var_sign(x)
    if sign: return var + n_vars
    else: return var

def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]

def solve_sat(n_vars, iclauses):
    solver = Minisat22()
    for iclause in iclauses: solver.add_clause(iclause)
    return solver.solve(), solver.get_model()

def gen_iclause_pair(n, p_k_2, p_geo):
    solver = Minisat22()
    iclauses = []
    while True:
        k_base = 1 if random.random() < p_k_2 else 2
        k = k_base + np.random.geometric(p_geo)
        iclause = generate_k_iclause(n, k)  # Generate clause with k variables
        solver.add_clause(iclause)
        is_sat = solver.solve()
        if is_sat:
            iclauses.append(iclause)  # Keep adding clauses until is unsat
        else:
            break
    iclause_unsat = iclause  # Save the unsat clause
    iclause_sat = [- iclause_unsat[0] ] + iclause_unsat[1:]  # Negate one literal of the unsat clause to make it sat
    return iclauses, iclause_unsat, iclause_sat  # Return the sat cnf, with the two additional unsat and sat clause

def make_csat_batch(iclauses, n_vars):
    out_dict = dict()
    out_dict['clauses'] = iclauses
    out_dict['n_clauses'] = len(iclauses)
    out_dict['n_vars'] = n_vars
    out_dict['label'], out_dict['solution'] = solve_sat(n_vars, iclauses)  # Inefficient: recompute solution
    assert out_dict['label']
    
    # construct adjacency for csat model
    out_dict['c_adj'], out_dict['csat_ind'] = cnf_to_csat(iclauses, n_vars)
    out_dict['c_adj'] = torch_sparse.SparseTensor.from_dense(out_dict['c_adj'])
    return out_dict

def cnf_to_csat(cnf, nv):
    adj = torch.zeros(nv*2, nv*2)  # Adjacency matrix

    # Indices for the literals (atoms and their negations). The negation of atom i is i+nv.
    idx = torch.arange(nv*2).view(2, -1).T
    for (i, j) in idx: # Add direct edge between postive literal and its negation
        adj[i, j] = 1

    # Convert negation literals to idices
    t_cnf = [[abs(l)+nv if l < 0 else l for l in c] for c in cnf]

    # Make indicator for each node (literal) of the graph
    indicator = torch.zeros(1, 2*nv)

    # Make mask for adversarial training shape = (nv, nc)
    mask = torch.zeros(nv*2, len(t_cnf))

    # For each clause add a new node
    for i, c in enumerate(t_cnf):
        # Add row and column to adjacency matrix
        adj = torch.nn.functional.pad(adj, (0, 1, 0, 1))
        # Add direct edge for each literal index in the clause to the new node (clause)
        for l in c:
            adj[l-1, -1] = 1
        # Add a new position with 1 for the new added node
        indicator = torch.nn.functional.pad(indicator, (0, 1), value=1)

        # Save the index of the clause node (is the last element), for connecting with the root node later
        t_cnf[i] = adj.size(0)

        # Add the true literal (first) of the clause to the mask
        l1, *_ = c
        mask[l1-1, i] = 1

    # Add a final node (root)
    adj = torch.nn.functional.pad(adj, (0, 1, 0, 1))

    # Add a new postion for the root with value -1
    indicator = torch.nn.functional.pad(indicator, (0, 1), value=-1)

    # For each clause index add a direct edge to the last (root) node
    for l in t_cnf:
        adj[l-1, -1] = 1
    return adj, indicator, mask
    