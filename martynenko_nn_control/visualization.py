from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

PALETTE = ["#203864", "#8E3B46", "#3B7A57", "#C77D1A", "#5E548E", "#2A9D8F"]
LABELS = {
    "lr_too_small": "LR 0.005",
    "lr_balanced": "LR 0.08",
    "lr_too_large": "LR 50",
    "small_network": "3 hidden",
    "larger_network": "18 hidden",
    "relu_balanced": "ReLU"
}

def _style_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

def save_dataset_plot(X_train, y_train, X_test, y_test, out):
    fig, ax = plt.subplots(figsize=(8,5.2))
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train.ravel(), cmap="cividis", marker='o', alpha=0.7, s=32, label='train')
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test.ravel(), cmap="cividis", marker='x', alpha=0.95, s=42, label='test')
    ax.set_title('Синтетичний набір даних для бінарної класифікації', fontsize=13, fontweight='bold')
    ax.set_xlabel('Ознака x1')
    ax.set_ylabel('Ознака x2')
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)

def line_chart(history_df, column, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(8.5,5.2))
    for i,(name, group) in enumerate(history_df.groupby('experiment')):
        ax.plot(group['epoch'], group[column], label=LABELS.get(name,name), linewidth=2.2, color=PALETTE[i % len(PALETTE)])
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Епоха')
    ax.set_ylabel(ylabel)
    _style_axes(ax)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)

def bar_chart(final_df, column, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(8.5,5.2))
    df = final_df.copy()
    labels = [LABELS.get(x,x) for x in df['experiment'].tolist()]
    values = df[column].tolist()
    bars = ax.bar(labels, values, color=PALETTE[:len(values)])
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=25)
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.3f}' if abs(v) < 2 else f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)

def decision_boundary(model, X, y, title, out):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict_proba(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8,5.2))
    ax.contourf(xx, yy, zz, levels=20, cmap="cividis", alpha=0.72)
    ax.contour(xx, yy, zz, levels=[0.5], colors=["#111111"], linewidths=1.8)
    ax.scatter(X[:,0], X[:,1], c=y.ravel(), cmap="cividis", s=22, edgecolors='white', linewidths=0.35)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Ознака x1')
    ax.set_ylabel('Ознака x2')
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)

def network_architecture(out):
    fig, ax = plt.subplots(figsize=(10,6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    input_y = [6.5, 3.5]
    hidden_y = np.linspace(8.6, 1.4, 10)
    output_y = [5]
    xs = {"input": 1.5, "hidden": 5.0, "output": 8.5}
    # layer captions
    ax.text(xs["input"], 9.45, "Вхідний шар", ha='center', fontsize=12, fontweight='bold', color="#203864")
    ax.text(xs["input"], 9.05, "2 ознаки: x1, x2", ha='center', fontsize=9)
    ax.text(xs["hidden"], 9.45, "Прихований шар", ha='center', fontsize=12, fontweight='bold', color="#203864")
    ax.text(xs["hidden"], 9.05, "10 нейронів, tanh/ReLU", ha='center', fontsize=9)
    ax.text(xs["output"], 9.45, "Вихідний шар", ha='center', fontsize=12, fontweight='bold', color="#203864")
    ax.text(xs["output"], 9.05, "ŷ: ймовірність класу", ha='center', fontsize=9)
    # arrows connections
    for iy in input_y:
        for hy in hidden_y:
            ax.plot([xs["input"], xs["hidden"]], [iy, hy], color="#9bb7d4", alpha=0.28, linewidth=1.0)
    for hy in hidden_y:
        for oy in output_y:
            ax.plot([xs["hidden"], xs["output"]], [hy, oy], color="#8fbf9f", alpha=0.35, linewidth=1.0)
    # nodes
    for i, y in enumerate(input_y, 1):
        circ = Circle((xs["input"], y), 0.35, color="#203864", alpha=0.88)
        ax.add_patch(circ)
        ax.text(xs["input"], y, f"x{i}", color="white", ha="center", va="center", fontsize=11, fontweight='bold')
    for i, y in enumerate(hidden_y, 1):
        circ = Circle((xs["hidden"], y), 0.28, color="#7E99B8", alpha=0.95)
        ax.add_patch(circ)
        ax.text(xs["hidden"], y, f"h{i}", color="white", ha="center", va="center", fontsize=8)
    circ = Circle((xs["output"], output_y[0]), 0.35, color="#8E3B46", alpha=0.92)
    ax.add_patch(circ)
    ax.text(xs["output"], output_y[0], "ŷ", color="white", ha="center", va="center", fontsize=13, fontweight='bold')
    # annotations
    ax.text(3.2, 0.55, "W1, b1", ha='center', fontsize=10, color="#203864")
    ax.text(6.7, 0.55, "W2, b2", ha='center', fontsize=10, color="#203864")
    ax.text(5, 0.08, "Архітектура 2-10-1: пряме поширення формує прогноз, backpropagation повертає помилку до ваг", ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)

def cybernetic_loop(out):
    fig, ax = plt.subplots(figsize=(10,5.8))
    ax.axis('off')
    boxes = [
        (0.8,3.0, "Вхідні дані\nX"),
        (2.7,3.0, "Нейронна\nмережа"),
        (4.8,3.0, "Прогноз\nŷ"),
        (6.7,3.0, "Функція\nвтрат L"),
        (4.8,0.9, "Backpropagation\nградієнти"),
        (2.7,0.9, "Оновлення\nваг W,b"),
    ]
    for x,y,t in boxes:
        rect = FancyBboxPatch((x,y), 1.4, 0.75, boxstyle="round,pad=0.08,rounding_size=0.08", facecolor="#eaf2fb", edgecolor="#203864", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+0.7, y+0.375, t, ha='center', va='center', fontsize=10)
    arrows = [((2.2,3.38),(2.7,3.38)), ((4.1,3.38),(4.8,3.38)), ((6.2,3.38),(6.7,3.38)),
              ((7.4,3.0),(5.5,1.65)), ((4.8,1.28),(4.1,1.28)), ((2.7,1.28),(3.4,3.0))]
    for a,b in arrows:
        ax.add_patch(FancyArrowPatch(a,b, arrowstyle='-|>', mutation_scale=14, linewidth=1.5, color="#8E3B46"))
    ax.text(4.8,4.4,"Кібернетичний контур управління навчанням", ha='center', fontsize=14, fontweight='bold', color="#203864")
    ax.text(4.8,0.15,"Помилка є сигналом зворотного зв'язку, який керує зміною параметрів мережі", ha='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)

def build_all_charts(X_train, X_test, y_train, y_test, models, history_df, final_df, results_dir: Path):
    save_dataset_plot(X_train, y_train, X_test, y_test, results_dir / 'dataset.png')
    line_chart(history_df, 'train_loss', 'Значення loss', 'Зменшення помилки під час навчання', results_dir / 'loss_curves.png')
    line_chart(history_df, 'test_accuracy', 'Accuracy', 'Точність класифікації на тестовій вибірці', results_dir / 'accuracy_curves.png')
    line_chart(history_df, 'gradient_norm', 'Норма градієнта', 'Сигнал зворотного поширення помилки', results_dir / 'gradient_norms.png')
    line_chart(history_df, 'update_norm', 'Норма оновлення ваг', 'Інтенсивність керуючої дії на параметри', results_dir / 'weight_updates.png')
    bar_chart(final_df, 'test_accuracy', 'Точність', 'Фінальна точність моделей', results_dir / 'final_accuracy.png')
    bar_chart(final_df, 'test_loss', 'Loss', 'Фінальне значення функції втрат', results_dir / 'final_loss.png')
    X_all = np.vstack([X_train, X_test])
    y_all = np.vstack([y_train, y_test])
    if 'lr_balanced' in models:
        decision_boundary(models['lr_balanced'], X_all, y_all, 'Межа класифікації: збалансований крок навчання', results_dir / 'decision_boundary_balanced.png')
    if 'lr_too_small' in models:
        decision_boundary(models['lr_too_small'], X_all, y_all, 'Межа класифікації: занадто малий крок навчання', results_dir / 'decision_boundary_slow.png')
    if 'lr_too_large' in models:
        decision_boundary(models['lr_too_large'], X_all, y_all, 'Межа класифікації: надмірний крок навчання', results_dir / 'decision_boundary_large_lr.png')
    network_architecture(results_dir / 'network_architecture.png')
    cybernetic_loop(results_dir / 'cybernetic_loop.png')
