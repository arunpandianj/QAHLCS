"""Figures with sample data: one random test image per class, and example outputs of a trained policy head."""
import glob
import json
import textwrap

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from medmnist import INFO  # noqa: E402

from .config import CFG, G, P, PS, RESULTS, SIG  # noqa: E402
from .data import load_medmnist  # noqa: E402

W = 5.17
plt.rcParams.update({"font.size": 6.5, "savefig.bbox": "tight", "pdf.fonttype": 42})
SHORT = {"actinic keratoses and intraepithelial carcinoma": "actinic keratosis", "benign keratosis-like lesions": "benign keratosis",
         "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)": "immature granulocyte", "colorectal adenocarcinoma epithelium": "adenocarcinoma",
         "cancer-associated stroma": "cancer stroma", "normal colon mucosa": "normal mucosa", "vascular lesions": "vascular lesion", "melanocytic nevi": "melanocytic nevus",
         "lymphocytes": "lymphocyte"}
NAMES = {"pathmnist": "PathMNIST", "dermamnist": "DermaMNIST", "bloodmnist": "BloodMNIST"}


def class_names(flag):
    lab = INFO[flag]["label"]
    return [SHORT.get(lab[str(i)], lab[str(i)]) for i in range(len(lab))]


def save(fig, name):
    fig.savefig(RESULTS / "figures" / f"{name}.pdf"); fig.savefig(RESULTS / "figures" / f"{name}.png", dpi=600); plt.close(fig)


def dataset_samples(seed=7):
    """One random test image from every class; each row is one dataset."""
    ds_list = CFG["DATASETS"]; ncol = 9
    fig, axs = plt.subplots(len(ds_list), ncol, figsize=(W, 0.8 * len(ds_list) + 0.3))
    for ax in np.ravel(axs): ax.axis("off")
    for r, ds in enumerate(ds_list):
        data, C, _ = load_medmnist(ds, splits=("test",)); x, y = data["test"]
        xs = x.cpu().numpy().transpose(0, 2, 3, 1); ys = y.cpu().numpy(); rng = np.random.default_rng(seed); names = class_names(ds)
        for c in range(C):
            ax = np.atleast_2d(axs)[r][c]; ax.imshow(xs[rng.choice(np.where(ys == c)[0])], interpolation="nearest")
            ax.set_title("\n".join(textwrap.wrap(names[c], 14, break_long_words=False)), fontsize=4.4, pad=1.5)
        np.atleast_2d(axs)[r][0].text(-0.15, 0.5, NAMES.get(ds, ds), transform=np.atleast_2d(axs)[r][0].transAxes, rotation=90, va="center", ha="right", fontsize=6.5)
    plt.tight_layout(); save(fig, "figA_dataset_samples")


def prediction_examples(head="quantum_4x2", lam=0.015, seed=0, n=6):
    """Output of a trained head on the first n test images of each dataset, read from the stored traces of the main stage."""
    ds_list = CFG["DATASETS"]; fig, axs = plt.subplots(len(ds_list), n, figsize=(W, 1.1 * len(ds_list) + 0.3))
    for r, ds in enumerate(ds_list):
        files = glob.glob(str(RESULTS / "runs" / f"main__{ds}__{head}__B__lam{lam}__lrdef__s{seed}__{SIG}.json"))
        data, C, _ = load_medmnist(ds, splits=("test",)); xs = data["test"][0][:n].cpu().numpy().transpose(0, 2, 3, 1); names = class_names(ds)
        tr = json.load(open(files[0]))["traces"][:n] if files else None
        for i in range(n):
            ax = np.atleast_2d(axs)[r][i]; ax.imshow(xs[i], interpolation="nearest"); ax.axis("off")
            if tr is None: continue
            t = tr[i]; order = [a for a in t["actions"] if 0 <= a < P]
            for p_ in range(P):
                row, col = divmod(p_, G)
                if p_ not in order: ax.add_patch(plt.Rectangle((col * PS - .5, row * PS - .5), PS, PS, color="k", alpha=0.7))
                else: ax.text(col * PS + PS / 2 - .5, row * PS + PS / 2 - .5, str(order.index(p_) + 1), color="w", ha="center", va="center", fontsize=5.5, weight="bold")
            ok = t["label"] == t["pred"]
            ax.set_title(f"true: {names[t['label']]}\npred: {names[t['pred']]}\n{len(order)} " + ("patch" if len(order) == 1 else "patches"), fontsize=4.6, color="#1a7f1a" if ok else "#b00020", pad=1.5)
    plt.tight_layout(); save(fig, "figB_prediction_examples")


if __name__ == "__main__":
    dataset_samples()
    prediction_examples()
    print("sample figures written to", RESULTS / "figures")
