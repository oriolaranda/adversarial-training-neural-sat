import torch
import copy
import torch_sparse
import torch.nn as nn
from torch_sparse import SparseTensor
from utils import sparse_blockdiag
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.r(self.l1(x))
        x = self.l2(x)
        return x

class CircuitSAT(nn.Module):
    def __init__(self, dim=100, dim_agg=50, dim_class=30, n_rounds=20, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.name = "CSAT"
        self.dim = dim
        self.n_rounds = n_rounds
        
        self.init = nn.Linear(4, dim)
        self.forward_msg = MLP(dim, dim_agg, dim)
        self.backward_msg = MLP(dim, dim_agg, dim)
        self.forward_update = nn.GRU(dim, dim)
        self.backward_update = nn.GRU(dim, dim)

        self.classif = MLP(dim, dim_class, 1) # Classification network C_a
        
        # F_o(u_G) = C_a(P(E_b(u_G)))
        # u_G -> DAG function, in this case, we encode the type as one-hot d-dimensional encoding {And, Or, Not, Variable}
        # C_a -> Classification network from q dimensions to [0, 1] range (soft classification)
        # E_b -> Embedding function that maps from d-dimensional function to q-dimensional function.
        # P -> Non-parametric Pooling function that aggregates the q-dim functions depends on the application

        # x_v = u_G(v) node feature vector
        # h_v = d_G(v) node state vector
        # h_v = GRU(x_v, h'_v)
        # h'_v = A({h_u | u \in \pi(v)}) 
        # A -> aggregator function (deep set function)

        # S(u_G) = R(F_o(u_G))
        # F_o -> Policy network
        # R -> Evaluator network (kind of reward)
        # L = (1-S(u_G))**k/(((1-S(u_G))**k) + S(u_G)**k)
        # Minimizing L loss, we push S(u_G) -> 1 (i.e., sat result, and thus a valid assignment) 

    def forward(self, sample):

        # Allocate tensors contiguous in GPU
        self.forward_update.flatten_parameters()
        self.backward_update.flatten_parameters()
        adj = sample['adj']
        h_state = self.init(sample['features'].cuda()).unsqueeze(0)

        for _ in range(self.n_rounds):
            f_pre_msg = self.forward_msg(h_state.squeeze(0)) # Pass through the GRU
            f_msg = torch_sparse.matmul(adj, f_pre_msg)

            _, h_state = self.forward_update(f_msg.unsqueeze(0), h_state)

            b_pre_msg = self.backward_msg(h_state.squeeze(0))
            b_msg = torch_sparse.matmul(adj.t(), b_pre_msg)

            _, h_state = self.backward_update(b_msg.unsqueeze(0), h_state)
            
        return self.classif(h_state.squeeze(0))

    @staticmethod
    def collate_fn(problems):
        # Prepare a batch of samples
        is_sat, solution, n_vars, clauses = [], [], [], []
        single_adjs, inds, masks, fnames, fs = [], [], [], [], []
        n_clauses = []
        var_labels = []
        for p in problems:
            adj = p['adj']
            single_adjs.append(adj)
            masks.append(p['mask'])
            n_vars.append(p['n_vars'])
            clauses.append(p['clauses'])
            n_clauses.append(len(p['clauses']))
            is_sat.append(float(p['label']))
            inds.append(p['ind'])
            f = copy.deepcopy(p['ind'][0])
            f[f == 1] = 2
            f[f == -1] = 3
            f[p['n_vars']:p['n_vars']*2] = 1
            fs.append(f)
            if 'filename' in p.keys(): fnames.append(p['filename'])
            if 'solution' in p.keys():
                c = np.array(p['solution'])
                c = c[c > 0] - 1
                l = torch.zeros(p['n_vars'])
                l[c] = 1
                var_labels.append(l)
            else:
                var_labels.append(torch.Tensor([-1]))
            solution.append(p['solution'])

        adj = sparse_blockdiag(single_adjs)
        indicators = torch.cat(fs).squeeze(0)
        assert indicators.size(0) == adj.size(0) == adj.size(1)
        feats = torch.zeros((indicators.size(0), 4))
        feats = torch.scatter(feats, 1, indicators.long().unsqueeze(1), torch.ones(feats.shape))
        assert torch.all(feats.sum(-1) == 1)

        sample = {
            'batch_size': len(problems),
            'n_vars': torch.Tensor(n_vars).int(),
            'is_sat': torch.Tensor(is_sat).float(),
            'adj': adj,
            'ind': inds,
            'mask': masks,
            'clauses': clauses,
            'solution': solution,
            'fnames': fnames,
            'features': feats,
            'varlabel': var_labels,
            'n_clauses': torch.Tensor(n_clauses).int()
        }
        return sample

    @staticmethod
    def get_representation(batch):
        all_inds = torch.cat(batch['ind'], dim=1).squeeze()
        literals = all_inds == 0
        ors = all_inds == 1
        # We are only interested in the region of the adjacency matrix between literals and clauses
        repr = batch['adj'][literals, ors].to_dense()
        return repr.cuda()

    @staticmethod
    def reconstruct(repr, batch):
        adj = batch['adj'].to_dense().to(repr.device)
        all_inds = torch.cat(batch['ind'], dim=1).squeeze()
        literals = all_inds == 0
        ors = all_inds == 1
        temp = adj[literals]
        temp[:, ors] = repr
        adj[literals] = temp
        batch['adj'] = SparseTensor.from_dense(adj)
        return batch


def evaluate_circuit(sample, emb, epoch, eps=1.2, hard=False):
    # explore exploit with annealing rate
    t = epoch ** (-eps)
    inds = torch.cat(sample['ind'], dim=1).view(-1)

    # set to negative to make sure we don't accidentally use nn preds for or/and
    temporary = emb.clone()
    temporary[inds == 1] = -1
    temporary[inds == -1] = -1

    # NOT gate
    temporary[sample['features'][:, 1] == 1] = 1 - emb[sample['features'][:, 0] == 1].clone()
    emb = temporary.clone()

    # OR gate
    idx = torch.arange(inds.size(0))[inds==1]
    or_gates = torch_sparse.index_select(sample['adj'], 1, idx.to(emb.device))
    e_gated = torch_sparse.mul(or_gates, emb)
    row, col, vals = e_gated.coo()
    assert torch.all(vals >= 0)

    if hard:
        e_gated = e_gated.max(dim=0)
    else:
        eps = e_gated + (-e_gated.max(dim=0).unsqueeze(0))
        _, _, vals = eps.coo()
        e_temp = torch.exp(vals / t).clone()
        # no elementwise multiplication in torch_sparse
        e_temp = SparseTensor(row=row, rowptr=None, col=col, value=e_temp, sparse_sizes=e_gated.sizes())
        e_gated = sparse_elem_mul(e_gated, e_temp).sum(dim=0) / e_temp.sum(dim=0)
        assert torch.all(torch.eq(e_gated, e_gated))

    # AND gate
    idx2 = torch.arange(inds.size(0))[inds==-1]
    and_gates = torch_sparse.index_select(sample['adj'], 0, idx.to(emb.device))
    and_gates = torch_sparse.index_select(and_gates, 1, idx2.to(emb.device))
    e_gated = torch_sparse.mul(and_gates, e_gated.unsqueeze(1))
    row, col, vals = e_gated.coo()
    assert torch.all(vals >= 0)
    
    if hard:
        e_gated = e_gated.min(dim=0)
    else:
        eps = e_gated + (-e_gated.min(dim=0).unsqueeze(0))
        _, _, vals = eps.coo()
        e_temp = torch.exp(-vals / t).clone()
        e_temp = SparseTensor(row=row, rowptr=None, col=col, value=e_temp, sparse_sizes=e_gated.sizes())
        e_gated = sparse_elem_mul(e_gated, e_temp).sum(dim=0) / e_temp.sum(dim=0)
        assert torch.all(torch.eq(e_gated, e_gated))

    assert torch.all(e_gated >= 0)
    assert len(e_gated) == len(sample['n_vars'])
    return e_gated


def sparse_elem_mul(s1, s2):
    # until torch_sparse support elementwise multiplication, we have to do this
    s1 = s1.to_torch_sparse_coo_tensor()
    s2 = s2.to_torch_sparse_coo_tensor()
    return SparseTensor.from_torch_sparse_coo_tensor(s1 * s2)


def custom_csat_loss(outputs, k=10, mean=True):
    # L loss -> smooth step function
    loss = (1 - outputs)**k / (((1 - outputs)**k) + outputs**k)
    return loss.mean() if mean else loss

