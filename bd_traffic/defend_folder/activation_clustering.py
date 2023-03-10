import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import silhouette_score

def cluster_metrics(cluster_1, cluster_0):

    num = len(cluster_1) + len(cluster_0)
    features = torch.cat([cluster_1, cluster_0], dim=0)

    labels = torch.zeros(num)
    labels[:len(cluster_1)] = 1
    labels[len(cluster_1):] = 0

    ## Raw Silhouette Score
    raw_silhouette_score = silhouette_score(features, labels)
    return raw_silhouette_score



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


def ac_cleanser(opt,inspection_split_loader, model, poison_index, num_classes, clusters=2):
    """
        adapted from : https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py
    """

    from sklearn.decomposition import PCA
    from sklearn.decomposition import FastICA
    from sklearn.cluster import KMeans


    suspicious_indices = []
    feats, class_indices = get_features(inspection_split_loader, model, num_classes)
    
    threshold = 0.25
    print(len(class_indices[0]), len(poison_index))

    for target_class in range(num_classes):

        if len(class_indices[target_class]) <= 1: continue # no need to perform clustering...

        temp_feats = np.array([feats[temp_idx] for temp_idx in class_indices[target_class]])
        temp_feats = temp_feats - temp_feats.mean(axis=0)
        projector = PCA(n_components=10)

        projected_feats = projector.fit_transform(temp_feats)
        kmeans = KMeans(n_clusters=2, max_iter=2000).fit(projected_feats)

        # by default, take the smaller cluster as the poisoned cluster
        if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
            clean_label = 1
        else:
            clean_label = 0


        outliers = []
        for (bool, idx) in zip((kmeans.labels_ != clean_label).tolist(), list(range(len(kmeans.labels_)))):
            if bool:
                outliers.append(class_indices[target_class][idx])

        score = silhouette_score(projected_feats, kmeans.labels_)
        print('[class-%d] silhouette_score = %f' % (target_class, score))
        if score > threshold:# and len(outliers) < len(kmeans.labels_) * 0.35:
            print(f"Outlier Num in Class {target_class}:", len(outliers))
            suspicious_indices += outliers
            

    true_positive  = 0
    num_positive   = int(opt.pr * 39209)
    false_positive = 0
    num_negative   = 39209 

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