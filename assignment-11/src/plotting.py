import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def new_fig(nrows=1, ncols=1, figsize=(10, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    return fig, axes


def finish(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
