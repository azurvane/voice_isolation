import torch
from itertools import permutations


def _si_sdr(predicted, target):
    epsilon = 1e-8
    
    predicted = predicted - torch.mean(predicted, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    dot = torch.sum(predicted * target, dim=-1)
    power = torch.sum(target * target, dim=-1) + epsilon
    
    s_target = (dot/power).unsqueeze(-1) * target
    e_noise = predicted - s_target
    
    return 10 * torch.log10(torch.sum(torch.pow(s_target, 2), dim=-1)/(torch.sum(torch.pow(e_noise, 2), dim=-1) + epsilon))

def pit_loss(outputs, targets, train=True):
    num_speaker = outputs.shape[1]
    perm_iterator = permutations(range(num_speaker))
    perm_list = list(perm_iterator)
    
    loss_list = []
    
    for perm in perm_list:
        speaker_losses = []
        for count, j in enumerate(perm):
            speaker_loss = _si_sdr(outputs[:, j, :], targets[:, count, :])
            speaker_losses.append(speaker_loss)
        perm_loss = -torch.stack(speaker_losses, dim=0).mean(dim=0)
        loss_list.append(perm_loss)
    
    loss_stack = torch.stack(loss_list, dim=0)
    min_loss = torch.min(loss_stack, dim=0).values
    
    if train:
        return min_loss.mean()
    else:
        return min_loss
