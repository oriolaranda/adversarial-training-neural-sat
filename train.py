import torch
import numpy as np
import argparse
from circuitsat import *
from gen_utils import *
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def get_random_solution(n_vars):
    solution = np.random.choice([-1, 1], size=n_vars, replace=True)
    return solution*np.arange(1, n_vars+1)

def get_random_clause(solution, literals, n_vars, vars_per_clause, p_k_2=0, p_geo=0):
    k_base = 1 if np.random.random() < p_k_2 else 2
    vars_per_clause = min(n_vars, k_base + np.random.geometric(p_geo))
    true_literal = np.random.choice(solution)
    other_literals = np.random.choice(literals, size=vars_per_clause-1)
    clause = np.concatenate([[true_literal], other_literals])
    assert len(clause) == vars_per_clause
    return clause

def generate_problem(n_vars):
    literals = [x for x in range(-n_vars, n_vars+1) if x != 0]
    solution = get_random_solution(n_vars)
    vars_per_clause = min(n_vars, 4)
    clauses = []
    for _ in range(args.n_clauses):
        clause = get_random_clause(solution, literals, n_vars, vars_per_clause, args.p_k_2, args.p_geo)
        clauses.append(clause)

    c_adj, c_ind = cnf_to_csat(clauses, n_vars)
    csat_problem = {
        'clauses': clauses,
        'n_clauses': len(clauses),
        'n_vars': n_vars,
        'label': 1,
        'solution': solution,
        'c_adj': torch_sparse.SparseTensor.from_dense(c_adj),
        'csat_ind': c_ind

    }
    return csat_problem

def generate_batch(model, n_vars, batch_size):
    problems = []
    for _ in range(batch_size):
        p = generate_problem(n_vars)
        problems.append(p)
    batch = model.collate_fn(problems)
    return batch


def send_to(batch, device):
    batch['adj'] = batch['adj'].to(device)
    batch['is_sat'] = batch['is_sat'].to(device)
    batch['features'] = batch['features'].to(device)
    return batch

def evaluate(model, n_vars=5, batch_size=32, device="cuda"):
    ts, ps, outs = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    eval_batches = 20
    model.eval()
    val_loss = []
    for i in range(eval_batches):
        with torch.no_grad():
            # For memory allocation efficiency
            model.forward_update.flatten_parameters()
            model.backward_update.flatten_parameters()

            batch_eval = generate_batch(model, n_vars, batch_size)
            batch_eval['adj'] = batch_eval['adj'].to(device)
            batch_eval['features'] = batch_eval['features'].to(device)
            assig = model(batch_eval)
            outputs = evaluate_circuit(batch_eval, torch.sigmoid(assig), 1, hard=True)
            val_loss.append(custom_csat_loss(outputs).item())

    ts = torch.cat((ts, batch_eval['is_sat']))
    outs = torch.cat((outs, outputs.flatten().detach().cpu()))
    preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())
    ps = torch.cat((ps, preds.flatten().cpu()))
    val_loss = np.mean(val_loss).item()
    val_acc = accuracy_score(ts.numpy(), ps.numpy())
    return val_loss, val_acc.item()



def train(model, n_batches, n_vars=5, epochs=5, batch_size=32, lr=0.00002, weight_decay=1e-10, grad_clip=0.65,
          model_path="./trained_models/", device="cuda"):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    dataset_name = "SAT-Constructive"
    simulation_name =f"{model.name}_{dataset_name}_{datetime.now().strftime('%d:%m-%H:%M:%S.%f')}"
    print("Experiment identifier: ", simulation_name)

    # n_vars = 5 fixed for the moment
    best_val_acc = 0.0
    val_accs = []
    
    for e in range(1, epochs+1):
        
        train_loss = []
        model.train()

        for i in tqdm(range(n_batches)):
            
            batch = generate_batch(model, n_vars, batch_size)
            batch = send_to(batch, device)
            
            optimizer.zero_grad()
            # For memory allocation efficiency
            model.forward_update.flatten_parameters()
            model.backward_update.flatten_parameters()

            assig = model(batch)
            pred = evaluate_circuit(batch, torch.sigmoid(assig), e)
            loss = custom_csat_loss(pred)
            train_loss.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip, norm_type=2.0)
            optimizer.step()

            if i % 100 == 0:
                print(f"[{e}] Training Loss:", np.mean(train_loss).item())
                train_loss = []

        val_loss, val_acc = evaluate(model, n_vars, batch_size)
        print(f"[{e}] Validation Loss:", val_loss, "| Validation Accuracy:", val_acc)
        val_accs.append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path + simulation_name + '_MAX.pt')
    results = dict()
    results['val_accs']: val_accs
    results['name']: simulation_name
    return results



def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # for n_vars in range(args.min_n_vars, args.max_n_vars):
    model = CircuitSAT()
    model.cuda()
    train(model, args.n_batches)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Training script')
    parser.add_argument('-n_batches', default=300, type=int)
    parser.add_argument('-n_clauses', default=4, type=int)
    parser.add_argument('-min_n_vars', default=3, help='start value for number of variables', type=int)
    parser.add_argument('-max_n_vars', default=10, help='end value for number of variables', type=int)
    parser.add_argument('-p_k_2', default=0.3, type=float)
    parser.add_argument('-p_geo', default=0.4, type=float)
    parser.add_argument('-seed', default=0, type=int)
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Using GPU: CUDA", torch.version.cuda)
    else:
        print("WARNING: Using CPU!")

    main(args)