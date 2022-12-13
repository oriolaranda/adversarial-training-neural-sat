import torch
from data import *
from circuitsat import *
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score

def main():
    batch_size = 32
    train, val, test, name = get_SAT_training_data("SAT-3-10")
    model = CircuitSAT()
    model.cuda()

    # model_path = "./trained_models/CSAT_SAT-Constructive_06:12-15:35:11.011767_MAX.pt"
    model_path = "./trained_models/CSAT_SAT-3-10_06:12-15:06:24.741026_MAX.pt"

    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded!")

    dl_test = DataLoader(dataset=test, collate_fn=model.collate_fn, pin_memory=False,
                            shuffle=False, batch_size=batch_size, num_workers=0)
    print("Dataset loaded!")
    val_loss = []
    ts, ps, outs = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    for _, batch in enumerate(dl_test):
        with torch.no_grad():
            model.forward_update.flatten_parameters()
            model.backward_update.flatten_parameters()
            batch['features'] = batch['features'].cuda()
            batch['adj'] = batch['adj'].cuda()
            outputs = model(batch)
            outputs = torch.sigmoid(outputs)
            outputs = evaluate_circuit(batch, outputs, 1, hard=True)
            val_loss.append(custom_csat_loss(outputs).item())

        ts = torch.cat((ts, batch['is_sat']))
        outs = torch.cat((outs, outputs.flatten().detach().cpu()))
        preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())
        ps = torch.cat((ps, preds.flatten().cpu()))
    val_acc = accuracy_score(ts.numpy(), ps.numpy()).item()
    val_loss = np.mean(val_loss)
    print("Test Loss:", val_loss)
    print("Test Accuracy:", val_acc)
    


if __name__ == '__main__':
    main()