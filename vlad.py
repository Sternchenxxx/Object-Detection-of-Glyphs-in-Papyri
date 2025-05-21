import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, Counter
from tqdm import tqdm

with open(r"results\features\hog_features.pkl", "rb") as f:
    data = pickle.load(f)

features = data["features"]
labels = data["labels"]
sample_ids = data["sample_ids"]

# division samples
unique_samples = np.unique(sample_ids)
n_train = int(0.7 * len(unique_samples))
train_ids = unique_samples[:n_train]
test_ids = unique_samples[n_train:]
train_mask = np.isin(sample_ids, train_ids)
test_mask = np.isin(sample_ids, test_ids)
X_train, y_train, sid_train = features[train_mask], labels[train_mask], sample_ids[train_mask]
X_test, y_test, sid_test = features[test_mask], labels[test_mask], sample_ids[test_mask]

#calculate μ_c
class_centers = {}
for c in np.unique(y_train):
    class_features = X_train[y_train == c]
    mu_c = np.mean(class_features, axis=0)
    class_centers[c] = mu_c

def build_feature_dict(X, y, sids):
    feat_dict = defaultdict(lambda: defaultdict(list))
    for feat, cls, sid in zip(X, y, sids):
        feat_dict[sid][cls].append(feat)
    return feat_dict

train_features = build_feature_dict(X_train, y_train, sid_train)
test_features = build_feature_dict(X_test, y_test, sid_test)

#VLAD
def compute_vlad(features, labels, sample_ids, class_centers):
    sample_to_features = defaultdict(list)
    sample_to_labels = defaultdict(list)

    for f,l, sid in zip(features, labels, sample_ids):
        sample_to_features[sid].append(f)
        sample_to_labels[sid].append(l)

    encodings = []
    sample_order = []

    for sid in tqdm(sample_to_features.keys(), desc="Computing VLAD vectors"):
        vlad_parts = []
        class_features = defaultdict(list)
        for f, l in zip(sample_to_features[sid], sample_to_labels[sid]):
            class_features[l].append(f)

        for c in sorted(class_centers.keys()):
            mu_c = class_centers[c]
            if c not in class_features:
                v_c = np.zeros_like(mu_c)
            else:
                diffs = [mu_c - f for f in class_features[c]]
                v_c = np.sum(diffs, axis=0)
            # L2 norm per class vector
            norm = np.linalg.norm(v_c)
            if norm > 0:
                v_c /= norm
            vlad_parts.append(v_c)
        vlad = np.concatenate(vlad_parts)
        # power normalization
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        vlad /= np.linalg.norm(vlad)
        encodings.append(vlad)
        sample_order.append(sid)

    return np.array(encodings), np.array(sample_order)

enc_train, order_train = compute_vlad(X_train, y_train, sid_train, class_centers)
enc_test, order_test = compute_vlad(X_test, y_test, sid_test, class_centers)

def majority_label_per_sample(sids, lbls):
    sid_to_lbls = defaultdict(list)
    for sid, l in zip(sids, lbls):
        sid_to_lbls[sid].append(l)
    return np.array([max(set(v), key=v.count) for v in sid_to_lbls.values()])

labels_test_eval = majority_label_per_sample(sid_test, y_test)
writer_counts = Counter(labels_test_eval)
rare_writers = {w for w, count in writer_counts.items() if count <= 2}
valid_indices = [i for i, label in enumerate(labels_test_eval) if label not in rare_writers]
enc_test_filtered = enc_test[valid_indices]
labels_test_eval_filtered = labels_test_eval[valid_indices]

def distances(encs):
    dists = 1.0 - encs.dot(encs.T)
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def _evaluate(encs, labels):
    dist_matrix = distances(encs)
    indices = dist_matrix.argsort()
    n_encs = len(encs)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    mAP, mAP_at_r, correct = [], [], 0
    for r in range(n_encs):
        precisions, rel = [], 0
        all_rel = np.count_nonzero(labels[indices[r]] == labels[r]) - 1
        prec_at_r = []

        for k in range(n_encs - 1):
            if labels[indices[r, k]] == labels[r]:
                rel += 1
                precisions.append(rel / float(k + 1))
                if k == 0:
                    correct += 1
                if k < all_rel:
                    prec_at_r.append(rel / float(k + 1))

        avg_precision = np.mean(precisions)
        avg_prec_at_r = np.sum(prec_at_r) / all_rel if all_rel > 0 else 0
        mAP.append(avg_precision)
        mAP_at_r.append(avg_prec_at_r)

    return float(correct) / n_encs, np.mean(mAP), np.mean(mAP_at_r)

top1, mAP, mAPr = _evaluate(enc_test_filtered, labels_test_eval_filtered)
print(f"\n[Eval - Exclude Rare Writers] Top-1: {top1:.4f}, mAP: {mAP:.4f}, mAP@R: {mAPr:.4f}")