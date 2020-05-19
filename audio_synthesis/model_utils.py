import torch

# Utils reference:
# Cite jason6963

def get_masked_with_pad_tensor(size, src, trg, pad_token=0):
    """
    :param size: the size of target input
    :param src: source tensor
    :param trg: target tensor
    :param pad_token: pad token
    :return:
    """
    # if src and trg are of shape (b, t, d), ignore the d dimension by taking argmax
    # as the ultimate goal is to get the mask

    if len(src.shape) > 2:
        src = torch.argmax(src.unsqueeze(1), dim=-1)
        trg = torch.argmax(trg.unsqueeze(1), dim=-1)
    
    print(src.shape)
    print(trg.shape)
    
    src = src[:, None, None, :]
    trg = trg[:, None, None, :]
    src_pad_tensor = torch.ones_like(src).to(src.device.type) * pad_token

    src_mask = torch.eq(src, src_pad_tensor)
    trg_mask = torch.eq(src, src_pad_tensor)
    if trg is not None:
        trg_pad_tensor = torch.ones_like(trg).to(trg.device.type) * pad_token
        dec_trg_mask = trg == trg_pad_tensor
        # boolean reversing i.e) True * -1 + 1 = False
        seq_mask = ~sequence_mask(torch.arange(1, size+1).to(trg.device), size)
        # look_ahead_mask = torch.max(dec_trg_mask, seq_mask)
        print(seq_mask.shape)
        look_ahead_mask = dec_trg_mask | seq_mask

    else:
        trg_mask = None
        look_ahead_mask = None

    return src_mask, trg_mask, look_ahead_mask

def sequence_mask(length, max_length=None):
    """Tensorflow의 sequence_mask를 구현"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


if __name__ == '__main__':
    import numpy as np
    s = np.array([np.array([1, 2]*48),np.array([1, 2, 3, 4]*24)])
    s = torch.Tensor(s)

    t = np.array([np.array([2, 3, 4, 5, 6]*20), np.array([1, 2, 3, 4, 5]*20)])
    t = torch.Tensor(t)
    print(t.shape)

    src_mask, tgt_mask, lookahead_mask = get_masked_with_pad_tensor(100, s, t)
    print(src_mask.shape)
    print(tgt_mask.shape)
    print(lookahead_mask.shape)