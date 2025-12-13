"""
Part I: Dimensionality Reduction.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

RESULTS_FOLDER = Path("../out")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    _ensure_dir(path.parent)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def vec_to_img(vec: np.ndarray) -> np.ndarray:
    """
    Convert vector into a displayable 2D image.
    """

    v = vec.reshape(-1)
    img = v.reshape(92, 112, order="F").T

    return img


def mae(X: np.ndarray, Xhat: np.ndarray) -> float:
    return float(np.mean(np.abs(X - Xhat)))


def relative_l2_error(X: np.ndarray, Xhat: np.ndarray) -> float:
    num = np.linalg.norm(X - Xhat)
    den = np.linalg.norm(X)

    return float(num / den)


def read_images(data_dir: Path = Path("../att_faces")) -> np.ndarray:
    """
    Output: images matrix, each column is a face image vector.
    """

    files = []

    for s in range(1, 41):
        sub = data_dir / f"s{s}"

        for i in range(1, 11):
            files.append(sub / f"{i}.pgm")

    images = np.zeros((92 * 112, 400), dtype=np.float64)

    for idx, path in enumerate(files):
        arr = np.array(Image.open(path), dtype=np.float64)  # (112, 92)
        vec = arr.T.reshape(-1, order="F")
        images[:, idx] = vec

    return images


def get_fixed_examples() -> list[int]:
    """
    Choose subjects subset.
    """

    examples = [
        (1 - 1) * 10 + (1 - 1),
        (10 - 1) * 10 + (1 - 1),
        (25 - 1) * 10 + (1 - 1),
    ]

    return examples


def calculate_pca(data: np.ndarray):
    """
    PCA calc.
    """

    _, n = data.shape
    mean_face = np.mean(data, axis=1, keepdims=True)
    X = data - mean_face

    cov = (X @ X.T) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    idx = np.argsort(eigvals)[::-1]  # descending
    variances = eigvals[idx]
    PCs = eigvecs[:, idx]

    total = np.sum(variances)
    percents = variances / total if total > 0 else np.zeros_like(variances)

    data_pc = PCs.T @ X  # projections (features x samples)

    return PCs, variances, data_pc, percents, mean_face


def reconstruct_one(PCs: np.ndarray, mean_face: np.ndarray, pc_coords: np.ndarray, k: int) -> np.ndarray:
    """
    Reconstruction.
    """
    return PCs[:, :k] @ pc_coords[:k, :] + mean_face


def figure_A_mean_and_eigenfaces(mean_face: np.ndarray, PCs: np.ndarray, n_show: int = 16) -> None:
    """
    Show leading eigenfaces.
    """

    out = RESULTS_FOLDER / "pca"
    _ensure_dir(out)

    cols = 4
    rows = int(np.ceil((n_show + 1) / cols))
    plt.figure(figsize=(10, 2.6 * rows))

    # mean face
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(vec_to_img(mean_face[:, 0]), cmap="gray")
    ax.set_title("Mean face")
    ax.axis("off")

    for i in range(n_show):
        ax = plt.subplot(rows, cols, i + 2)
        ef = PCs[:, i]

        ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-12)
        ax.imshow(vec_to_img(ef_norm), cmap="gray")
        ax.set_title(f"PC {i + 1}")
        ax.axis("off")

    plt.suptitle("Mean face and leading eigenfaces", fontsize=14)
    savefig(out / "figure_A_mean_and_eigenfaces.png")


def figure_B_importance_curve(percents: np.ndarray) -> None:
    """
    Importance decreasing via cumulative explained variance.
    """

    out = RESULTS_FOLDER / "pca"
    _ensure_dir(out)

    cum = np.cumsum(percents)

    plt.figure(figsize=(8, 4))
    plt.plot(cum, linewidth=2)
    plt.ylim(0, 1.01)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Number of eigenfaces (k)")
    plt.ylabel("Cumulative explained variance")
    plt.title("Eigenface importance: cumulative variance")

    # mark 90% and 95%
    for target in (0.90, 0.95):
        k_star = int(np.searchsorted(cum, target) + 1)

        plt.axhline(target, linestyle="--", linewidth=1)
        plt.axvline(k_star, linestyle="--", linewidth=1)
        plt.text(k_star + 5, target - 0.03, f"{int(target * 100)}% at k={k_star}", fontsize=10)

    savefig(out / "cumulative_variance.png")


def figure_C_reconstruction_panel(
        data: np.ndarray,
        PCs: np.ndarray,
        mean_face: np.ndarray,
        data_pc: np.ndarray,
        example_indices: list[int],
        k_list: list[int],
) -> None:
    """
    Show originals vs reconstructions as k increases.
    """

    out = RESULTS_FOLDER / "reconstruction"
    _ensure_dir(out)

    n_examples = len(example_indices)
    ncols = 1 + len(k_list)  # original + each k
    plt.figure(figsize=(2.2 * ncols, 2.6 * n_examples))

    for r, idx in enumerate(example_indices):
        x = data[:, idx:idx + 1]
        z = data_pc[:, idx:idx + 1]

        # orig
        ax = plt.subplot(n_examples, ncols, r * ncols + 1)
        ax.imshow(vec_to_img(x[:, 0]), cmap="gray")
        ax.axis("off")

        if r == 0:
            ax.set_title("Original")

        # reconstruction
        for c, k in enumerate(k_list):
            xhat = reconstruct_one(PCs, mean_face, z, k)
            ax = plt.subplot(n_examples, ncols, r * ncols + (c + 2))
            ax.imshow(vec_to_img(xhat[:, 0]), cmap="gray")
            ax.axis("off")

            if r == 0:
                ax.set_title(f"k={k}")

    plt.suptitle("Reconstruction improves as k increases", fontsize=14)
    savefig(out / "reconstruction_panel.png")


def figure_D_error_curve(
        data: np.ndarray,
        PCs: np.ndarray,
        mean_face: np.ndarray,
        example_indices: list[int],
        k_list: list[int]
) -> None:
    """
    Plot reconstruction error vs k on the selected subset.
    """

    out = RESULTS_FOLDER / "reconstruction"
    _ensure_dir(out)

    Xsub = data[:, example_indices]  # (features, n_examples)
    Xsub_c = Xsub - mean_face
    Zsub = PCs.T @ Xsub_c  # (features, n_examples)

    maes = []
    rels = []
    for k in k_list:
        Xhat = reconstruct_one(PCs, mean_face, Zsub, k)
        maes.append(mae(Xsub, Xhat))
        rels.append(relative_l2_error(Xsub, Xhat))

    maes = np.array(maes)

    plt.figure(figsize=(8, 4))
    plt.plot(k_list, maes, marker="o", linewidth=2, label="MAE")

    plt.grid(True, alpha=0.3)
    plt.xlabel("Number of eigenfaces (k)")
    plt.ylabel("Error")
    plt.title("Reconstruction error vs number of eigenfaces")
    plt.legend()

    savefig(out / "reconstruction_error_curve.png")


if __name__ == "__main__":
    num_samples = 400
    data = read_images(Path("../att_faces"))[:, :num_samples]

    PCs, variances, data_pc, percents, mean_face = calculate_pca(data)

    figure_A_mean_and_eigenfaces(mean_face, PCs, n_show=16)
    figure_B_importance_curve(percents)

    example_indices = get_fixed_examples()
    k_list = [5, 10, 25, 50, 100, 200, 300, 400]

    figure_C_reconstruction_panel(
        data=data,
        PCs=PCs,
        mean_face=mean_face,
        data_pc=data_pc,
        example_indices=example_indices,
        k_list=k_list
    )

    figure_D_error_curve(
        data=data,
        PCs=PCs,
        mean_face=mean_face,
        example_indices=example_indices,
        k_list=k_list
    )

    print(f"Saved figures under: {RESULTS_FOLDER.resolve()}")
