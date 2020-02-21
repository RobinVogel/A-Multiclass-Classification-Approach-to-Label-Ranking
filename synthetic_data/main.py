import os
import sys
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

from utils import (make_tab_inv_prop, generate_losses, DEFAULT_N_POINTS)

from plot import (represent_dist, plot_legend, plot_noise_condition,
                  plot_boxplot)


def main():
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],
                      'size': 16})
    plt.rc('text', usetex=True)

    if len(sys.argv) >= 2 and (sys.argv[1] in {"noisy_1", "separ_1",
                                               "noisy_2", "separ_2"}):
        # --------- Generates the losses data
        i_split = int(len(DEFAULT_N_POINTS)*2/3)
        ind_to_n_def = {1: DEFAULT_N_POINTS[:i_split],
                        2: DEFAULT_N_POINTS[i_split:]}
        type_run_to_a = {"noisy": 0.2, "moder": 0.5, "separ": 0.8}
        type_run, ind_run = sys.argv[1].split("_")
        m_loss, m_cycle, m_kend = generate_losses(
            n_points=ind_to_n_def[int(ind_run)], a=type_run_to_a[type_run])

        np.save("data_dump/{}_loss_{}.npy".format(type_run, ind_run), m_loss)
        np.save("data_dump/{}_cycl_{}.npy".format(type_run, ind_run), m_cycle)
        np.save("data_dump/{}_kend_{}.npy".format(type_run, ind_run), m_kend)

    elif len(sys.argv) >= 2 and (sys.argv[1] == "boxplots"):
        # --------- Creates the boxplots
        y_label_for_measure = {
            r"loss": r"$P\{ \hat{\sigma}_X \ne \sigma_X^* \}$",
            r"cycl": "\# of cycles",
            r"kend": r"$E[ d_\tau(\hat{\sigma}_X, \sigma_X^*) ]$"}
        for measure, ylabel in y_label_for_measure.items():
            for type_run in ["noisy", "separ"]:
                ylim = ([-0.1, 1.1] if measure in {"cycl", "loss"}
                        else [-0.1, 15.1])
                ms = [np.load("data_dump/{}_{}_{}.npy".format(
                    type_run, measure, ind_run)) for ind_run in [1, 2]]
                m = np.concatenate(ms, axis=0)
                np.save("data_dump/{}_{}.npy".format(measure, type_run), m)

                plot_boxplot("data_dump/{}_{}.npy".format(measure, type_run),
                             "boxplots/{}_{}.pdf".format(measure, type_run),
                             ylabel=ylabel, ylim=ylim)
    elif sys.argv[1] == "dists":
        # --------- Represent the distributions
        represent_dist(outfile="noisy_dist.pdf", n_points=1000, a=0.2)
        represent_dist(outfile="separ_dist.pdf", n_points=1000, a=0.8)
        plot_legend()

    elif sys.argv[1] == "mammen":
        assert len(sys.argv) > 2
        a = float(sys.argv[2])
        # --------- Plot the noise conditions
        for i, j in combinations(range(0, 8), 2):
            dirname = "noise_cond_a={}".format(str(a))
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            outfile = "{}/class_{}_vs_{}.png".format(dirname, i, j)
            plot_noise_condition(i, j, outfile, a=a)

    elif sys.argv[1] == "invs":
        # --------- Get the inversions proportions:
        make_tab_inv_prop(outfile="inv_prop_tables/noisy.txt", a=0.2)
        make_tab_inv_prop(outfile="inv_prop_tables/separ.txt", a=0.8)


if __name__ == "__main__":
    main()
