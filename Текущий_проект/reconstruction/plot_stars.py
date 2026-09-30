"""Scientific diagnostic figures for the frozen experiment (evaluation only)."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from .physical import PLANE_Z, STRIP_CENTERS


def main():
    report = json.loads(Path("results/star_test.json").read_text())
    rows = [json.loads(s) for s in Path("results/star_test_predictions.jsonl").read_text().splitlines()]
    y = np.array([r["truth_star"] for r in rows])
    p = np.array([r["probability"] for r in rows])
    precision, recall, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(recall, precision, color="#276c9b")
    ax.scatter(report["detection"]["recall"], report["detection"]["precision"], color="#df792f", zorder=3)
    ax.set(xlabel="Полнота", ylabel="Точность", xlim=(0, 1.02), ylim=(0, 1.02),
           title=f"Детекция: F1 = {report['detection']['f1']:.3f}, n = {len(y)}")
    matrix = np.array(report["detection"]["confusion_matrix_tn_fp_fn_tp"]).reshape(2, 2)
    ax = axes[0, 1]; ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max()/2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Не звезда", "Звезда"],
           yticklabels=["Не звезда", "Звезда"], xlabel="Предсказание", ylabel="Слабая MC-разметка", title="Матрица ошибок")
    stars = [r for r in rows if r["truth_star"]]
    error = np.sort([np.linalg.norm(np.array(r["vertex_mm"])-r["true_vertex_mm"]) for r in stars])
    ax = axes[1, 0]; ax.plot(error, np.arange(1, len(error)+1)/len(error))
    ax.axvline(np.median(error), color="#df792f", ls="--")
    ax.set(xlabel="Ошибка вершины до геометрического уточнения, мм", ylabel="Доля событий", xlim=(0, 120),
           title=f"Медиана {np.median(error):.1f} мм; P90 {np.quantile(error,.9):.1f} мм")
    counts = np.zeros((13, 13), int)
    for r in stars:
        counts[min(r["true_branch_count"], 12), min(r["branch_count"], 12)] += 1
    ax = axes[1, 1]; ax.imshow(np.log1p(counts), origin="lower", cmap="Blues")
    ax.plot([0, 12], [0, 12], "--", color="#df792f")
    ax.set(xlabel="Предсказанное число длинных ветвей", ylabel="Число по разметке", xticks=[0, 4, 8, 12], yticks=[0, 4, 8, 12],
           xticklabels=["0", "4", "8", "12+"], yticklabels=["0", "4", "8", "12+"],
           title=f"Средняя ошибка {report['count_mae_on_true_stars']:.2f} ветви")
    fig.suptitle("Geant4: отложенная выборка одного набора моделирования\nМетки основаны на совместном рождении треков; это не проверка на PAMELA", fontsize=12)
    fig.savefig("results/star_test_dashboard.png", dpi=160)
    fig.savefig("results/star_test_dashboard.pdf")
    plt.close(fig)


def example_figure(data, event_id, result, out):
    from .star_reconstruction_benchmark import truth44
    from .core import projections_from_hits
    xz, yz = projections_from_hits(data, event_id)
    truth = truth44(data, event_id)
    recon = result["reconstruction"]
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    layout = fig.add_gridspec(2, 3)
    for image, name, col, parity in ((xz, "XZ: измерение", 0, 1), (yz, "YZ: измерение", 1, 0)):
        ax = fig.add_subplot(layout[0, col]); ax.imshow(np.log1p(image), origin="lower", aspect="auto", cmap="magma", extent=(-.5, 95.5, -.5, 21.5))
        ax.set(xlabel="Стрип", ylabel="Номер плоскости этой проекции", title=name)
    ax = fig.add_subplot(layout[0, 2]); ax.axis("off")
    ax.text(0, 1, f"Событие {event_id}\nСчёт детектора: {result['proposal']['probability']:.3f}\n"
            f"Ветвей предсказано: {result['proposal']['branch_count']}\n"
            f"В 3D сопоставлено: {recon['matched_branches']}\n"
            f"Перестановок: {recon['pairing_count']}\n\n"
            "Цвет = log(1 + энергия в МэВ).\n"
            "Эталон Geant4 доступен только\nпри оценке. Достраивание использует\nинтерполяцию ненаблюдаемой проекции.", va="top", fontsize=10)
    energy_max = float(np.log1p(truth.max()))
    for volume, title, col in ((truth, "Эталон Geant4", 0),
                               (recon["hypotheses"][0]["energy"], "Прямые ветви + энергии", 1),
                               (recon["completion"], "Достраивание (гипотеза)", 2)):
        ax = fig.add_subplot(layout[1, col], projection="3d")
        coords = np.argwhere(volume > .01)
        if len(coords):
            ax.scatter(STRIP_CENTERS[coords[:, 1]], STRIP_CENTERS[coords[:, 2]], PLANE_Z[coords[:, 0]],
                       c=np.log1p(volume[tuple(coords.T)]), s=8, cmap="magma", vmin=0, vmax=energy_max)
        ax.set(xlim=(-123, 123), ylim=(-123, 123), zlim=(0, 178), xlabel="X, мм", ylabel="Y, мм", zlabel="Z, мм", title=title)
        ax.view_init(elev=17, azim=-65)
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
