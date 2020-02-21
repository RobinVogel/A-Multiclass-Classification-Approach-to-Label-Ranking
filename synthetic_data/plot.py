import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from utils import (get_etas_array, get_etas, DEF_A, generate_dataset,
                   DEFAULT_N_POINTS)


def represent_dist(outfile="test.pdf", n_points=1000, a=DEF_A,
                   legend=False):
    x_axis = np.linspace(0, 1, n_points + 1)
    y_values = get_etas_array(x_axis, a=a)

    plt.figure(figsize=(4, 2))
    n_classes = y_values.shape[1]
    colors = cm.rainbow(np.linspace(0, 1, n_classes))
    all_leg = list()
    last_cumul = np.zeros(n_points + 1)
    for i in range(n_classes):
        cumul = y_values[:, :(i+1)].sum(axis=1).ravel()
        plt.plot(x_axis, cumul, color=colors[i])
        to_leg = plt.fill_between(x_axis, cumul, last_cumul,
                                  color=colors[i], alpha=0.5,
                                  label="class {}".format(i))
        all_leg.append(to_leg)
        last_cumul = cumul
    if legend:
        plt.legend()
    plt.ylabel("$\eta_k(x)$'s")
    plt.xlabel("$x$")
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.tight_layout()
    plt.savefig(outfile, format="pdf")
    return all_leg


def plot_legend(outfile="legend.pdf"):
    n_classes = len(get_etas(0, a=0.5))

    colors = cm.rainbow(np.linspace(0, 1, n_classes))
    # fig = plt.figure(figsize=(4.2, 0.5))
    fig = plt.figure(figsize=(7.5, 1))
    all_handles = [plt.Line2D([0], [0], color='none', label='Class'),
                   plt.Line2D([0], [0], color='none', label='')]
    all_handles = all_handles + [plt.fill_between(
        [0], [0], [0], color=colors[i], alpha=0.5, label="{}".format(i))
                                 for i in range(n_classes)]
    fig.legend(handles=all_handles, ncol=5, loc="center", scatterpoints=1)
    plt.gca().axis('off')
    fig.savefig(outfile)


def plot_noise_condition(k, l, outfile="noise.pdf", n=100000, a=DEF_A):
    x, y = generate_dataset(n=n, a=a)
    filt = np.logical_or(y == k, y==l)
    vals = list()
    for x0 in x[filt]:
        etas = get_etas(x0, a=a)
        eta_kl = etas[k]/(etas[k] + etas[l])
        vals.append(np.abs(eta_kl - 1/2))
    n_classes = len(etas)

    plt.figure(figsize=(4, 2))
    plt.title("class {} vs class {} (k={}, l={})".format(k, l, k, l))
    colors = cm.rainbow(np.linspace(0, 1, n_classes))
    plt.hist(vals, bins=100, density=True)
    plt.ylabel(r"$P\{ \| \eta_{k,l}(x) - 1/2 \| > t \}$")
    plt.xlabel("$t$")
    plt.grid()
    plt.savefig(outfile)


def plot_boxplot(mat_name, outfile="boxplot.pdf", n_points=DEFAULT_N_POINTS,
                 ylabel="", ylim=None):
    m = np.load(mat_name)
    plt.figure(figsize=(4, 2))
    w = 0.1

    def width(p, w):
        return 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)

    plt.boxplot(m.transpose(), positions=n_points, widths=width(n_points, w))
    plt.grid()
    plt.xscale("log")
    plt.ylim(ylim)
    plt.tight_layout(rect=[0.05, 0, 1, 1])  # rect=[0.05, 0, 1, 1])
    plt.ylabel(ylabel)
    plt.xlabel("$n$")
    plt.savefig(outfile, format="pdf")
