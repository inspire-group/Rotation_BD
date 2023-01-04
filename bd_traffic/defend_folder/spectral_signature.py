import numpy as np
import torch
from tqdm import tqdm
import config


def get_features(data_loader, model, num_classes):

    model.eval()
    class_indices = [[] for _ in range(num_classes)]
    feats = []

    with torch.no_grad():
        sid = 0
        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            _, x_feats = model(ins_data, True)
            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                feats.append(x_feats[bid].cpu().numpy())
                b_target = ins_target[bid].item()
                class_indices[b_target].append(sid + bid)
            sid += this_batch_size
    return feats, class_indices


def ss_cleanser(opt, inspection_split_loader, model, poison_index, num_classes):
    """
        adapted from : https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py
    """


    # Spectral Signature requires an expected poison ratio (we allow the oracle here as a baseline)
    num_poisons_expected = opt.pr * 39209 * 1.5 # allow removing additional 50% (following the original paper)

    feats, class_indices = get_features(inspection_split_loader, model, num_classes)

    suspicious_indices = []


    for i in range(num_classes):

        if len(class_indices[i]) > 1:

            temp_feats = np.array([feats[temp_idx] for temp_idx in class_indices[i]])
            temp_feats = torch.FloatTensor(temp_feats)

            mean_feat = torch.mean(temp_feats, dim=0)
            temp_feats = temp_feats - mean_feat
            _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)

            vec = V[:, 0]  # the top right singular vector is the first column of V
            vals = []
            for j in range(temp_feats.shape[0]):
                vals.append(torch.dot(temp_feats[j], vec).pow(2))

            k = min(int(num_poisons_expected), len(vals) // 2)
            # default assumption : at least a half of samples in each class is clean

            _, indices = torch.topk(torch.tensor(vals), k)
            for temp_index in indices:
                suspicious_indices.append(class_indices[i][temp_index])

    num_positive   = int(opt.pr * 39209)
    false_positive = 0
    num_negative   = 39209 
    true_positive = 0

    for i in range(num_negative+num_positive):
        if i in suspicious_indices:
            if i in poison_index:
                true_positive += 1
            else:
                false_positive += 1

    print('<Overall Performance Evaluation>')
    print('Elimination Rate = %d/%d = %f' % (true_positive, num_positive, true_positive / num_positive))
    print('Sacrifice Rate = %d/%d = %f' % (false_positive, num_negative, false_positive / num_negative))


    return suspicious_indices