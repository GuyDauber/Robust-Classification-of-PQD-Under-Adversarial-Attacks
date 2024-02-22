import torch
import torchattacks
from utils import *
from torch.utils.data import DataLoader
import numpy as np


def attacker(model, attack_type, data_loader, device, epsilon=8/255, steps=10, random_start=True,
             steps_df=50, overshoot=0.02, c=1, kappa=0, steps_cw=50, lr=0.01):
    """
    Makes an adversarial dataset after an attack

    Args:
        model: DL-based PQD classifier
        attack_type: the type of the adversarial attack on the model
        data_loader: PQD signals to be attacked, and their labels
        device: cuda or cpu
        epsilon: maximum perturbation on the signals under attack
        steps: number of iterations
        random_start: boolean flag for random initialization
        steps_df: number of iterations for DeepFool attack
        overshoot: noise enhancement parameter for DeepFool attack
        c: magnitude of the hueristic function for the CW attack
        kappa: confidence for the CW attack
        lr: learning rate for the optimization algorithm of the CW attack

    Returns:
        adv_signal: adversarial signals after the attack implementation
        adv_loader: adversarial dataset loader after the attack implementation
    """
    match attack_type:
        case 'FGSM':
            attack = torchattacks.FGSM(model, epsilon)
        case 'BIM':
            alpha = epsilon / steps
            attack = torchattacks.BIM(model, epsilon, alpha, steps)
        case 'PGD':
            alpha = 0.5 * (epsilon / steps)
            attack = torchattacks.PGD(model, epsilon, alpha, steps, random_start)
        case 'DeepFool':
            attack = torchattacks.DeepFool(model, steps_df, overshoot)
        case 'CW':
            attack = torchattacks.CW(model, c, kappa, steps_cw, lr)

    adv_inputs = torch.tensor([], dtype=torch.long, device=device)
    labels = torch.tensor([], dtype=torch.long, device=device)

    for i, data in enumerate(data_loader):
        batch_inputs, batch_labels = data
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        attacked_inputs = attack(batch_inputs, batch_labels)
        adv_inputs = torch.cat((adv_inputs, attacked_inputs))
        labels = torch.cat((labels, batch_labels))

    adv_set = TensorDataset(adv_inputs, labels)
    adv_loader = DataLoader(adv_set, batch_size=16)

    return adv_inputs, adv_loader
