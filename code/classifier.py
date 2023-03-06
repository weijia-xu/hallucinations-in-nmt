import sys
import pickle
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

language = sys.argv[1]
dir_out = sys.argv[2]

if language == "zh":
    hidden_size = 10
elif language == "de":
    hidden_size = 30
max_iter = 1000
lr = 0.003
n_classifiers = 20

# specify the paths to the training, development, and test data (all must be pickle files)
data_train = pickle.load(open(dir_out + 'lrp_train', 'rb'))
data_dev = pickle.load(open(dir_out + 'lrp_dev', 'rb'))
data_test = pickle.load(open(dir_out + 'lrp_eval', 'rb'))

def deduplicate(data):
    dedup_data = []
    source_set = set()
    for item in data:
        source = item['src']
        if source not in source_set:
            source_set.add(source)
            dedup_data.append(item)
    return dedup_data

def compute_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)

def sliding_window_similarity(inp_lrp, window_size=1):
    L = inp_lrp.shape[1]
    if L <= window_size * 2:
        return 0.9
    prev_lrp = None
    similarity_scores = []
    for t in range(0, L, window_size):
        mean_lrp = np.sum(inp_lrp[:, t : t + window_size], axis=-1) / window_size
        if prev_lrp is not None:
            similarity_scores.append(compute_similarity(prev_lrp, mean_lrp))
        prev_lrp = mean_lrp
    return sum(similarity_scores) / len(similarity_scores)

def extract_features(data, ignore_eos=False):
    X, y = [], []
    for i in range(len(data)):
        inp_lrp = data[i]['inp_lrp']
        tgt_tokens = inp_lrp.shape[0]
        if ignore_eos:
            elem = inp_lrp[:, :-1] / np.sum(inp_lrp[:, :-1], axis=1).reshape([tgt_tokens, 1])
        else:
            elem = inp_lrp / np.sum(inp_lrp, axis=1).reshape([tgt_tokens, 1])
        if ignore_eos:
            elem = np.sum(elem, axis=0) * (data[i]['inp_lrp'].shape[1] - 1)
        else:
            elem = np.sum(elem, axis=0) *  data[i]['inp_lrp'].shape[1]
        elem = elem / tgt_tokens
        features = []
        if language == "zh":
            features += [elem[i] for i in range(1)] + [elem[-i] for i in range(1, 2)]
            # features.append(np.sum(elem > 1) / elem.shape[0])
            for window_size in range(1, 8):
                mean_sim = sliding_window_similarity(inp_lrp.T, window_size=window_size)
                features.append(mean_sim)
        elif language == "de":
            features += [elem[i] for i in range(1)] + [elem[-i] for i in range(1, 2)]
            # features.append(np.sum(elem > 1) / elem.shape[0])
            for window_size in range(1, 8):
                mean_sim = sliding_window_similarity(inp_lrp.T, window_size=window_size)
                features.append(mean_sim)
        X.append(features)
        y.append(data[i]['label'])
    return X, y

def ensemble_predictions(predictions):
    K = len(predictions)
    N = len(predictions[0])
    final_predictions = []
    for i in range(N):
        num_positive = 0
        for k in range(K):
            if predictions[k][i] == 1:
                num_positive += 1
        if num_positive > K // 2:
            final_predictions.append(1)
        else:
            final_predictions.append(0)
    return final_predictions

def tune_threshold(labels, scores):
    best_threshold = 0
    best_f1 = 0
    for i in range(100):
        threshold = i / 100
        preds = (scores > threshold).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

X_train, y_train = extract_features(data_train)
X_dev, y_dev = extract_features(data_dev)

data_test = deduplicate(data_test)
X_test, y_test = extract_features(data_test)

dev_scores = []
predictions, scores = [], []
max_f1 = 0
f1_scores = []
for i in range(n_classifiers):
    classifier = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=(hidden_size,), learning_rate_init=lr).fit(X_train, y_train)
    y_pred = classifier.predict(X_dev)
    y_scores = classifier.predict_proba(X_dev)[:, 1]
    f1 = f1_score(y_dev, y_pred)
    auc = roc_auc_score(y_dev, y_scores)
    f1_scores.append(f1)
    if f1 > max_f1:
        print(f1)
        print(auc)
        dev_scores = y_scores
        max_f1 = f1
        predictions = classifier.predict(X_test)
        scores = classifier.predict_proba(X_test)[:, 1]

print("Mean F1: %.3f" % np.mean(f1_scores))
print("F1 std: %.3f" % np.std(f1_scores))

print("Test AUC: %.3f" % roc_auc_score(y_test, scores))
print("Test F1: %.3f" % f1_score(y_test, predictions))
print("Test Precision: %.3f" % precision_score(y_test, predictions))
print("Test Recall: %.3f" % recall_score(y_test, predictions))