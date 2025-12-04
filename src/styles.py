import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Global modern LaTeX-like style ---
def set_style():
    # Base Seaborn modern style
    plt.style.use("seaborn-v0_8")

    # Force Computer Modern Roman (LaTeX style)
    # mpl.rcParams["mathtext.fontset"] = "cm"
    # mpl.rcParams["font.family"] = "serif"
    # mpl.rcParams["font.serif"] = ["cmr10"]  # Computer Modern Roman
    
    # Global style settings
    mpl.rcParams.update({
        # Axes
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.1,
        "axes.titlesize": 14,
        "axes.labelsize": 12,

        # Ticks
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Grid
        "grid.alpha": 0.25,
        "grid.linestyle": "--",

        # Lines
        "lines.linewidth": 2.0,

        # Figure background
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

# Apply globally
set_style()
