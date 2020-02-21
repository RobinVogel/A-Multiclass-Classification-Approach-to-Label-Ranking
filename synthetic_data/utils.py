import datetime
from itertools import combinations
import numpy as np

from setting import get_etas, DEF_A

DEFAULT_N_POINTS = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]


def get_etas_array(x_arr, a=DEF_A):
    y_values = list()
    for x in x_arr:
        y_values.append(get_etas(x, a))
    return np.array(y_values)


def categorical_sample(p):
    p = np.array(p)
    return np.where(np.random.multinomial(1, p))[0][0]


def generate_dataset(n=1000, a=DEF_A):
    x = np.random.uniform(0, 1, n)
    y = list()
    for x0 in x:
        y.append(categorical_sample(get_etas(x0, a)))
    return x, np.array(y)


def learn_separator(x, y):
    # y \in {0, 1}
    ind_sort = np.argsort(x)
    x_sorted = x[ind_sort]
    y_sorted = y[ind_sort]

    n = len(y_sorted)
    n_pos = y_sorted.sum()
    best_sep = 0
    # Smallest than thre = positif (True), other false
    best_orient = True
    best_error = len(y_sorted)

    def update_best(orient, err, sep):
        nonlocal best_orient, best_error, best_sep
        if err < best_error:
            best_orient = orient
            best_error = err
            best_sep = sep

    cur_sep = 0
    # For the left part of the split
    cur_fn = n_pos
    cur_fp = 0

    for x0, y0 in zip(x_sorted, y_sorted):
        cur_sep = (x0 + cur_sep)/2
        update_best(True, cur_fn + cur_fp, cur_sep)
        update_best(False, n - cur_fn - cur_fp, cur_sep)
        cur_fn -= (y0 == 1)
        cur_fp += (y0 == 0)

    if len(x_sorted) > 0:
        cur_sep = (x_sorted[-1] + 1)/2
        update_best(True, cur_fn + cur_fp, cur_sep)
        update_best(False, n - cur_fn - cur_fp, cur_sep)

    return best_sep, best_error, best_orient


def ovo_label_ranking(x, y, n_classes):
    """Implements Figure 1."""
    combis = list(combinations(range(n_classes), 2))
    predictors = dict()
    for k, l in combis:
        # g_{k,l} yields the same optimizer as g_{l,k}
        x_k = x[y == k]
        x_l = x[y == l]
        x_bin = np.append(x_k, x_l)
        y_bin = np.append(np.ones(x_k.shape[0]), np.zeros(x_l.shape[0]))
        sep, _, orient = learn_separator(x_bin, y_bin)
        # print("sep class {} vs {}: {:.2f} / orient: {}".format(
        #     k, l, sep, orient))
        predictors[(k, l)] = {"sep": sep, "orient": orient}
    return predictors


def predict_from_ovo(predictors, x, n_classes):
    scores = np.zeros((len(x), n_classes))
    for kl, d in predictors.items():
        k, l = kl
        sep, orient = d["sep"], d["orient"]
        pred = ((x < sep) == orient).astype(int)
        scores[:, k] += pred
        scores[:, l] += 1-pred
    return scores


def cycles_from_ovo(predictors, x, n_classes):
    # print("adj creation {}".format(datetime.datetime.now()))
    def check_dir(x0, node, obj, direct):
        if node > obj:
            d = predictors[(obj, node)]
            dir_pred = 0
        else:
            d = predictors[(node, obj)]
            dir_pred = 1
        sep, orient = d["sep"], d["orient"]
        pred = ((x0 < sep) == orient)
        return pred == (direct == dir_pred)
    # print("cycle check {}".format(datetime.datetime.now()))
    has_cycles = list()
    for i, x0 in enumerate(x):
        has_cycle = False
        def explore(node, seen, direction):
            next_nodes = [l for l in range(0, n_classes)
                          if (l != node) and check_dir(x0, node, l, direction)]
            unseen = list()
            for n in next_nodes:
                if n in seen:
                    # print(n, seen)
                    return True
                else:
                    unseen.append(n)
            if not unseen:
                return False
            for n in unseen:
                cur = explore(n, seen + [n], direction)
                if cur:
                    return cur
            return False
        has_cycle = explore(0, [0], 1) or explore(0, [0], 0)
        has_cycles.append(has_cycle)
    # print("done cycle check {}".format(datetime.datetime.now()))
    return has_cycles


def kendall_tau(s_1, s_2):
    n_c = s_1.shape[1]
    n = s_1.shape[0]
    assert s_1.shape == s_2.shape
    inv_1 = np.zeros((n, int((n_c*(n_c-1))/2)))
    inv_2 = np.zeros((n, int((n_c*(n_c-1))/2)))
    kendall_tau = np.zeros(n)
    for kl in combinations(range(n_c), 2):
        k, l = kl
        or_1 = np.where(s_1==k)[1] - np.where(s_1==l)[1]
        or_2 = np.where(s_2==k)[1] - np.where(s_2==l)[1]
        kendall_tau += (or_1*or_2 < 0).astype(int)
    return np.mean(kendall_tau)


def inversions_from_ovo(predictors, x, n_classes):
    model_pair = np.zeros((len(x), (n_classes*(n_classes-1))//2))
    gt_order = np.argsort(get_etas_array(x), axis=1)
    gt_pair = np.zeros((len(x), (n_classes*(n_classes-1))//2))
    for i, kl in enumerate(combinations(range(n_classes), 2)):
        k, l = kl
        gt_pair[:, i] = (np.where(gt_order == k)[1]
                               > np.where(gt_order == l)[1])
        d = predictors[(k, l)]
        sep, orient = d["sep"], d["orient"]
        pred = ((x < sep) == orient)
        model_pair[:, i] = pred
    res = (model_pair != gt_pair)
    print("0-1 loss: " + str(1 - (~res).all(axis=1).mean()))
    print("kendall tau: " + str(res.mean(axis=0).sum()))
    res = (model_pair != gt_pair).astype(int)
    return res.astype(int)


def make_tab_inv_prop(outfile="tmp.txt", a=DEF_A):
    f = open(outfile, "wt")
    n_learn = 10000
    x, y = generate_dataset(n=n_learn, a=a)
    n_classes = len(get_etas(0))
    pred = ovo_label_ranking(x, y, n_classes)
    print("a value: " + str(a))
    x, y = generate_dataset(n=n_learn, a=a)
    invs = inversions_from_ovo(pred, x, n_classes)
    header = ""
    line = ""
    last_k = 0
    for i, kl in enumerate(pred.keys()):
        k, l = kl
        if k != last_k:
            f.write(header)
            f.write(line)
            header = ""
            line = ""
            for i in range(k):
                header += " & "
                line += " & "
            last_k = k
        header += r"{} vs {}".format(k, l)
        line += " ${:.2f}$ ".format(np.mean(invs[:, i]))
        header += "& " if l < n_classes - 1 else (r"\\" + "\n")
        line += "& " if l < n_classes - 1 else (r"\\" + "\n")
    f.write(header)
    f.write(line)
    f.close()


def compute_error(n_learn=1000, n_test=10000, a=DEF_A):
    # print("start {}".format(datetime.datetime.now()))
    x, y = generate_dataset(n=n_learn, a=a)
    # print("generated data {}".format(datetime.datetime.now()))
    n_classes = len(get_etas(0))
    pred = ovo_label_ranking(x, y, n_classes)
    # print("learnt classifiers {}".format(datetime.datetime.now()))
    x, y = generate_dataset(n=n_test, a=a)
    # print("generated test data {}".format(datetime.datetime.now()))
    s = predict_from_ovo(pred, x, n_classes)
    # print("predicted {}".format(datetime.datetime.now()))
    lr_order = np.argsort(s, axis=1)
    gt_order = np.argsort(get_etas_array(x), axis=1)
    kendall = kendall_tau(lr_order, gt_order)
    loss = 1 - np.all(lr_order == gt_order, axis=1).mean()
    # print("computed loss {}".format(datetime.datetime.now()))
    s = cycles_from_ovo(pred, x, n_classes)
    # print("computed cycles {}".format(datetime.datetime.now()))
    return loss, np.mean(s), kendall


def generate_losses(n_points=DEFAULT_N_POINTS, a=DEF_A, n_exp=100, n_test=1000):
    all_losses = list()
    all_cycles = list()
    all_kendall = list()
    n = len(n_points)
    for i, n_learn in enumerate(n_points):
        print("Experiment {} out of {} - time {}".format(
            i, n, datetime.datetime.now()), flush=True)
        inter_cycle = list()
        inter_loss = list()
        inter_kendall = list()
        for _ in range(n_exp):
            loss, cycle, kendall = compute_error(n_learn, n_test, a=a)
            inter_loss.append(loss)
            inter_cycle.append(cycle)
            inter_kendall.append(kendall)
        all_losses.append(inter_loss)
        all_cycles.append(inter_cycle)
        all_kendall.append(inter_kendall)
    return np.array(all_losses), np.array(all_cycles), np.array(all_kendall)
