import torch


def to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, list):
        return [x.to(device) for x in data]
    if isinstance(data, tuple):
        return tuple([x.to(device) for x in data])
