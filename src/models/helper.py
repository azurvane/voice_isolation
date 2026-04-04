
def _reshape(batch):
    if batch.dim() == 2:
        return batch.unsqueeze(1)
    return batch