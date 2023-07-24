import json
import torch
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from itertools import chain, repeat, cycle
from torch.utils.data.dataloader import DataLoader

from circuitsat import CircuitSAT, evaluate_circuit, custom_csat_loss
from data import SATDataset
from utils import get_sat_mask
from gen_dataset import get_sat_datasets, send_to
from gen_utils import cnf_to_csat


def save_results(results):
    with open(f"outputs/{results['name']}-results.json", 'w') as f:
        json.dump(results, f)


def process_dict(d, num_batches):
    # reduction is necessary or writing to db will fail because of size
    for key in d.keys():
        values = torch.Tensor(d[key])
        values = values.view(-1, num_batches).mean(-1)
        d[key] = values.tolist()
    return d

def load_adv_batch():
    return NotImplemented

def apply_projections(p, sat_mask, adj_mask):
    # these projections (operations) are discrete, so
    # makes no sense to include them into the computational graph
    with torch.no_grad():
        # enforce values between 0 and 1 for p, and also for repr since repr is binary.
        p.data = torch.clamp(p, 0, 1).detach() # TODO: detach needed?

        # enforce block diag structure in p. satisfied by p -> satisfied by repr
        p.data = p*adj_mask

        # enforce sat values repr[sat] + p = repr[sat]
        p[sat_mask.bool()] = 0

    return p


def update_best(best_loss, best_p, loss, p, n_clauses):
    with torch.no_grad(): # required because loss and p have grads
        improved_mask = best_loss > loss.cpu()
        if improved_mask.any():
            best_loss[improved_mask] = loss[improved_mask].cpu()
            prev = 0
            for k, n in enumerate(np.cumsum(n_clauses)):
                if improved_mask[k]:
                    best_p[..., prev:n] = p[..., prev:n].cpu()
                prev = n
    return best_loss, best_p


def sample_best_perturbation(best_p, repr, batch, model, epoch, temp, n_samples=20):
    with torch.no_grad(): # required because we don't want best_p requires grad
        p = best_p.to(repr.device)
        best_loss = torch.full_like(batch['is_sat'], float('inf'), device='cpu')
        best_p = torch.full_like(p, float('inf'), device='cpu')

        for _ in range(n_samples):
            # obtain a discrete version of p
            p_sampled = torch.bernoulli(p)

            # keep only best sample
            new_repr = repr + torch.where(repr.bool(), -p_sampled, p_sampled)
            batch = CircuitSAT.reconstruct(new_repr, batch)
            assig = model(batch)
            assig /= temp  # temperature to avoid zero grads
            outputs = evaluate_circuit(batch, torch.sigmoid(assig), epoch + 1)
            loss = -custom_csat_loss(outputs, mean=False)
            best_loss, best_p = update_best(best_loss, best_p, loss, p_sampled, batch['n_clauses'])
        assert torch.all(torch.ne(best_p, float('inf')))
    return best_p


def adversarial_training(model, batch, epoch, n_steps=10, lr=0.1, temp=5):

    repr = copy.deepcopy(CircuitSAT.get_representation(batch)) # TODO: deepcopy needed?

    # define the perturbation p
    p = torch.zeros(repr.shape, requires_grad=True, device=repr.device)

    # get the satisfiability mask
    # sat_mask = get_sat_mask(repr, batch)
    sat_mask = torch.block_diag(*batch['mask']).int().to(repr.device)

    # get the adjacency mask
    adj_mask = [torch.ones(x * 2, y) for (x, y) in zip(batch['n_vars'], batch['n_clauses'].int().tolist())]
    adj_mask = torch.block_diag(*adj_mask).int().to(repr.device)

    # define the optimizer only for perturbation p
    optimizer = torch.optim.Adam([p], lr=lr)

    # send these tensors to cpu that are just keeping the best results, for not polluting gpu memory
    best_loss = torch.full_like(batch['is_sat'], float('inf'))
    best_p = p.detach().cpu()

    # optimization of perturbation p (continuous)
    for _ in range(n_steps):
        # apply the perturbation p. p represents the probabilities of flipping an edge.
        # removing edges only if there exist, otherwise trying to add one.
        new_repr = repr + torch.where(repr.bool(), -p, p)

        # reconstruct new batch and get predictions
        batch = CircuitSAT.reconstruct(new_repr, batch)
        assig = model(batch)
        assig /= temp  # temperature to avoid zero grads
        # in order to be consistent with what the model is learning at this stage (this epoch) should be needed.
        outputs = evaluate_circuit(batch, torch.sigmoid(assig), epoch + 1)
        loss = -custom_csat_loss(outputs, mean=False)

        # update best_loss and best_p
        best_loss, best_p = update_best(best_loss, best_p, loss, p, batch['n_clauses'])

        # update matrix p
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # apply projection constraints after p is updated
        p = apply_projections(p, sat_mask, adj_mask) # discrete operations
        batch['adj'] = batch['adj'].detach() # TODO: why is it needed here?
        # batch['adj'] requires grad at this point because of adding p

    # sample best perturbation p (discrete)
    best_p = sample_best_perturbation(best_p, repr, batch, model, epoch, temp)

    # apply best perturbation p (discrete)
    best_p = best_p.to(repr.device)
    best_repr = repr + torch.where(repr.bool(), -best_p, best_p)
    batch = CircuitSAT.reconstruct(best_repr, batch)

    assert torch.all(torch.eq(best_p, best_p * adj_mask))
    assert torch.all(torch.eq(best_repr, best_repr*adj_mask))

    return batch, best


def evaluate(model, val, batch_size=32, device="cuda"):

    dl_val = DataLoader(dataset=val, collate_fn=CircuitSAT.collate_fn, pin_memory=False,
                        shuffle=False, batch_size=batch_size, num_workers=0)

    ts, ps, outs = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    val_loss, val_acc = [], -1
    model.eval()

    for _, batch_eval in enumerate(dl_val):
        with torch.no_grad():
            # For memory allocation efficiency
            model.forward_update.flatten_parameters()
            model.backward_update.flatten_parameters()

            # send to device
            batch_eval['adj'] = batch_eval['adj'].to(device)
            batch_eval['features'] = batch_eval['features'].to(device)

            # forward pred
            assig = model(batch_eval)
            outputs = evaluate_circuit(batch_eval, torch.sigmoid(assig), 1, hard=True)
            val_loss.append(custom_csat_loss(outputs).item())

        ts = torch.cat((ts, batch_eval['is_sat']))
        preds = (outputs > 0.5).float().cuda()
        ps = torch.cat((ps, preds.flatten().detach().cpu()))

        val_acc = accuracy_score(ts.numpy(), ps.numpy()).item()
        print(val_acc)
    val_loss = np.mean(val_loss).item()
    return val_loss, val_acc

def save_adv_sample(batch):
    cvars, cclauses = 0, 0
    singles = []
    for i in range(len(batch['is_sat'])):
        # get perturbation for individual sample
        print(batch.keys())
        assert 0 == 1

        p_sample = p[cvars : (cvars+b['n_vars'][i]*2),
                                  cclauses : (cclauses + b['n_clauses'][i])]
        singles.append(M_single)

        # save with sample file name
        obj = {
            'M': M_single.cpu().numpy(),
            'attack': pert
        }
        with open(folder+"/"+b['fnames'][i], 'wb') as f:
            pkl.dump(obj, f)

        cvars += b['n_vars'][i]*2
        cclauses += b['n_clauses'][i]

    assert torch.cat([torch.flatten(p) for p in singles]).sum() == M.sum()


"""def train(model, n_problems, batch_size=32, epochs=60, lr=0.00002, weight_decay=1e-10, grad_clip=0.65,
          n_vars_interval=(), k=None, p_k_2=0, p_geo=0, adv=False, model_path="./trained_models/", device="cuda"):
"""
def training(model, train, val, epochs=60, batch_size=32, lr=0.00002, weight_decay=1e-10, grad_clip=0.65,
             adv=False, model_path="./trained_models/", dataset_name="", device="cuda"):

    dataset_name = dataset_name
    simulation_name =f"{model.name}_{dataset_name}_{datetime.now().strftime('%d:%m-%H:%M')}".lower()
    print("Experiment identifier: ", simulation_name)

    assert isinstance(train, SATDataset) and isinstance(val, SATDataset)
    dl_train = DataLoader(dataset=train, collate_fn=CircuitSAT.collate_fn, pin_memory=True,
                          shuffle=True, batch_size=batch_size, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    val_accs, val_losses, train_losses = [], [], []
    
    for e in range(epochs):
        train_loss = []
        model.train()

        for _, batch in enumerate(dl_train):
            if adv and e % 2 == 0:
                batch, p = adversarial_training(model, batch, e)
                save_adv_sample(batch, p)
            elif adv:
                batch = load_adv_batch()
            batch = send_to(batch, device)

            # For memory allocation efficiency
            model.forward_update.flatten_parameters()
            model.backward_update.flatten_parameters()

            assig = model(batch)
            pred = evaluate_circuit(batch, torch.sigmoid(assig), e+1)
            loss = custom_csat_loss(pred)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, norm_type=2.0)
            optimizer.step()

        train_loss = np.mean(train_loss).item()
        train_losses.append(train_loss)
        print(f"[Epoch {e}] Training Loss:", train_loss)
        val_loss, val_acc = evaluate(model, val, batch_size)
        print(f"[Epoch {e}] Validation Loss:", val_loss, "| Validation Accuracy:", val_acc)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path + simulation_name + '_MAX.pt')
    results = dict()
    results = process_dict(results, len(dl_train))
    results['val_accs'] = val_accs
    results['val_losses'] = val_losses
    results['train_losses'] = train_losses
    results['name'] = simulation_name
    return results


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert args.data in ['sat', 'k-sat']
    dataset_name = "RND-SAT-3-10" if args.data == 'sat' else "RND-K-SAT-3-10"
    train, val, test = get_sat_datasets(dataset_name, args.seed)

    model = CircuitSAT()
    model.cuda()

    results = training(model, train, val, epochs=args.epochs, adv=args.adv, dataset_name=dataset_name)
    # results = train(model, args.n_problems, epochs=args.epochs, n_vars_interval=n_vars_interval, p_k_2=args.p_k_2,
    #                 p_geo=args.p_geo, adv=args.adv)

    save_results(results)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Training script')
    parser.add_argument('-epochs', default=30, type=int)
    parser.add_argument('-data', default="sat", type=str)
    parser.add_argument('-adv', default=False, action='store_true')
    parser.add_argument('-seed', default=0, type=int)
    parsed_args = parser.parse_args()

    if torch.cuda.is_available():
        print("Using GPU: CUDA", torch.version.cuda)
    else:
        print("WARNING: Using CPU!")

    main(parsed_args)