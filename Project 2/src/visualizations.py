import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def plotstuff(x, y, ylabel, title, save=False, filepath=None):
    keys = y.keys()
    sns.set_theme()
    fig, ax = plt.subplots()
    for key in keys:
        ax.plot(x, y[key], label=key)

    ax.set(xlabel="Epochs", ylabel=ylabel, title=title)
    ax.legend()
    if save:
        plt.savefig(filepath)
    plt.show()


with open("runs/le_12nodes_different_samplesize.pkl", "rb") as f:
    le_samplesize = pickle.load(f)

with open("runs/var_12nodes_different_samplesize.pkl", "rb") as f:
    var_samplesize = pickle.load(f)

with open("runs/stde_12nodes_different_samplesize.pkl", "rb") as f:
    stde_samplesize = pickle.load(f)

with open("runs/le_12nodes_deltaT.pkl", "rb") as f:
    le_deltaT = pickle.load(f)

with open("runs/le_12nodes_deltanv.pkl", "rb") as f:
    le_deltanv = pickle.load(f)

le_longrange = np.load("runs/le_longrange.npy")
le_20 = np.load("runs/le_20hidden_lr0.05.npy")
le_12 = np.load("runs/le_12hidden_lr0.05.npy")
le_4 = np.load("runs/le_4hidden_lr0.05.npy")
le_20node_218 = np.load("runs/le_20node_2^18sample.npy")
le_20node_214 = np.load("runs/le_20node_2^14.npy")
x = np.arange(25)

sns.set_theme()
# fig, ax = plt.subplots()
# ax.plot(x, le_20node_214, label=r"$2^{14}$ samples")
# ax.plot(x, le_20node_218, label=r"$2^{18}$ samples")
# ax.set_title("Ground state estimate for 20 spin-chain system")
# ax.set_xlabel("Epochs")
# ax.set_ylabel("Local Energy")
# ax.legend()
# plt.savefig("../doc/figs/le_20node_samplesize.pdf")


breakpoint()
