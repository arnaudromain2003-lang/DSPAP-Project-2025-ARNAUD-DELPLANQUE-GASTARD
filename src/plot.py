# Usefull functions to plot

# Import necessary libraries
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import calendar
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def plot_validations(df_tuple, title="Weekly / hourly validations"):
    """
    Plot grouped daily validations for multiple DataFrames.
    dfs must be a tuple or list of DataFrames.
    """

    n = len(df_tuple)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)

    # Titre global au-dessus de tous les graphs
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)

    # In casse there is only one dataframe, axes n'est pas iterable → on le transforme
    if n == 1:
        axes = [axes]

    for ax, df in zip(axes, df_tuple):
        # groupby comme dans ta fonction originale
        daily = df.groupby("date")["Flow"].sum()

        ax.plot(daily.index, daily.values)
        #ax.set_title(f"Validations per day - {getattr(df, 'name', 'dataset')}")
        ax.set_title(getattr(df, "name", "Dataset"))
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of validations")
        ax.grid(True)

    plt.tight_layout()
    plt.show() 


def plot_week(df_tuple, week_nbr, color=True, superpose=True):
    """
    Version généralisée de plot_week_data :
    - df_tuple = tuple de DataFrames (bus, tram, métro, etc.)
    - Chaque DF doit contenir 'date' (datetime), 'Flow'
    - Paramètres color et superpose comme ta fonction originale
    """

    # ============= 1. Filtrer chaque DataFrame =============
    week_dfs = []
    for df in df_tuple:
        base_name = getattr(df, "name", "Dataset")
        df_w = df[df["date"].dt.isocalendar().week == week_nbr].copy()
        df_w["day"] = df_w["date"].dt.day_name()
        df_w.name = base_name
        week_dfs.append(df_w)

    # ============= 2. Palette par day =============
    couleurs = {
        'Monday': 'red',
        'Tuesday': 'blue',
        'Wednesday': 'green',
        'Thursday': 'orange',
        'Friday': 'purple',
        'Saturday': 'brown',
        'Sunday': 'pink'
    }

    # ============= 3. Cas simple : pas de couleur, pas de superposition =============
    if not color and not superpose:
        return plot_validations(tuple(week_dfs), title=f"Validations — Week {week_nbr}")

    # ============= 4. Un subplot par DataFrame =============
    n = len(week_dfs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Weekly / hourly validations Week {week_nbr}",
        fontsize=18, fontweight="bold", y=1.05
    )

    # =============================================================
    # ================ 5. CAS : color=True & superpose=True =======
    # =============================================================
    if color and superpose:
        for ax, df_w in zip(axes, week_dfs):

            
            if df_w.empty:
                ax.set_title(f"{getattr(df_w, 'name', 'Dataset')} (empty)")
                continue

            # heure décimale
            df_w["hour"] = df_w["date"].dt.hour + df_w["date"].dt.minute / 60
            df_plot = df_w.groupby(["hour", "day"])["Flow"].sum().reset_index()

            # corrections overlap
            df_plot["hour"] = df_plot["hour"].round(2)

            sns.lineplot(
                data=df_plot,
                x="hour",
                y="Flow",
                hue="day",
                palette=couleurs,
                estimator=None,      # 🔥 indispensable pour overlap correct
                ax=ax
            )
            ax.legend(title="Day of the week")
            ax.set_title(getattr(df_w, "name", "Dataset"))
            ax.set_xlabel("Heure")
            ax.set_ylabel("Validations")
            ax.grid(True)
            ax.set_xticks(range(0, 24))

        plt.tight_layout()
        plt.show()
        return

    # =============================================================
    # ========== 6. CAS : color=True & superpose=False ============
    # =============================================================
    if color and not superpose:
        for ax, df_w in zip(axes, week_dfs):

            if df_w.empty:
                ax.set_title(f"{getattr(df_w, 'name', 'Dataset')} (empty)")
                continue

            df_daily = df_w.groupby(["date", "day"])["Flow"].sum().reset_index()

            sns.lineplot(
                data=df_daily,
                x="date",
                y="Flow",
                hue="day",
                palette=couleurs,
                ax=ax
            )
            ax.legend(title="Day of the week")
            ax.set_title(getattr(df_w, "name", "Dataset"))
            ax.set_xlabel("Date")
            ax.set_ylabel("Validations")
            ax.grid(True)

        plt.tight_layout()
        plt.show()
        return

    # =============================================================
    # ======================== 7. ERREUR ==========================
    # =============================================================
    return "Invalid combination: color=False requires superpose=False."




def plot_day(df_tuple, date_str):
    """
    Plot hourly validations for a given date for all DataFrames in df_tuple.
    Each DF must contain:
        - 'date' (datetime)
        - 'Flow'
        - optional df.name for display
    """

    # Convert input to datetime.date
    date = pd.to_datetime(date_str).date()

    # ========= 1. Filter each DF for the selected day =========
    day_dfs = []
    for df in df_tuple:
        base_name = getattr(df, "name", "Dataset")

        # SAFE: always work on a copy before modifying columns
        df_local = df.copy()

        # SAFE: use .loc to avoid SettingWithCopyWarning
        df_local.loc[:, "date_only"] = df_local["date"].dt.date

        df_d = df_local[df_local["date_only"] == date].copy()
        df_d.loc[:, "hour"] = df_d["date"].dt.hour + df_d["date"].dt.minute / 60

        df_d.name = base_name
        day_dfs.append(df_d)

    # ========= 2. Setup figure =========
    n = len(day_dfs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Hourly validations - {date_str}",
        fontsize=18,
        fontweight="bold",
        y=1.05
    )

    # ========= 3. Plot each DF =========
    for ax, df_d in zip(axes, day_dfs):

        if df_d.empty:
            ax.set_title(f"{df_d.name} (empty)")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Validations")
            ax.grid(True)
            continue

        df_plot = df_d.groupby("hour")["Flow"].sum().reset_index()

        ax.plot(df_plot["hour"], df_plot["Flow"], marker="o")
        ax.set_title(df_d.name)
        ax.set_xlabel("Hour of the day")
        ax.set_ylabel("Validations")
        ax.set_xticks(range(0, 24))
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# def plot_day_variation_non_normalized(df_g):
#     """
#     Scatter plot : daily total flow per weekday, colored by Time_Period.
#     """

#     # Always work on a copy to avoid SettingWithCopyWarning
#     df = df_g.copy()

#     # Ensure datetime type
#     df['date'] = pd.to_datetime(df['date'])

#     # Create clean columns
#     df['date_only'] = df['date'].dt.normalize()        # ← évite les "date != date" bugs
#     df['jour_num'] = df['date'].dt.dayofweek
#     df['jour_semaine'] = df['date'].dt.day_name()

#     # Group by day + period
#     df_daily = (
#         df.groupby(['date_only', 'jour_num', 'jour_semaine', 'Time_Period'])['Flow']
#           .sum()
#           .reset_index()
#     )

#     # Color map by period
#     period_colors = {
#         "Regular week": "blue",
#         "Winter holidays": "red",
#         "Christmas holidays": "green",
#         "COVID period": "purple",
#         "Fête des Lumières": "orange"
#     }

#     plt.figure(figsize=(14, 6))

#     # Plot each period separately
#     for period, color in period_colors.items():
#         subset = df_daily[df_daily['Time_Period'] == period]

#         plt.scatter(
#             subset['jour_num'],
#             subset['Flow'],
#             c=color,
#             label=period,
#             alpha=0.7,
#             s=50
#         )

#     plt.xticks(
#         ticks=range(7),
#         labels=["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
#     )
#     plt.xlabel('Jour de la semaine')
#     plt.ylabel('Flux total journalier')
#     plt.title('Flux journalier par jour de la semaine et période')
#     plt.legend(title='Période')
#     plt.grid(True)

#     plt.show()




def plot_day_variation(df_tuple, normalize=False):
    """
    Affiche autant de subplots que de DataFrames.
    Chaque subplot montre la variation journalière du Flow
    selon la Time_Period avec les mêmes couleurs.

    normalize = True  → Flow_normalized = Flow / Flow.max()
    """

    period_colors = {
        "Regular week": "blue",
        "Winter holidays": "red",
        "Christmas holidays": "green",
        "COVID period": "purple",
        "Fête des Lumières": "orange"
    }

    n = len(df_tuple)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle("Flux journalier par jour de semaine — multi-DF", fontsize=18, y=1.03)

    # =====================================================
    #                 BOUCLE SUR LES DATAFRAMES
    # =====================================================
    for i, df in enumerate(df_tuple):

        ax = axes[i]
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["date_only"] = df["date"].dt.date
        df["jour_num"] = df["date"].dt.dayofweek
        df["jour_semaine"] = df["date"].dt.day_name()

        df_daily = (
            df.groupby(["date_only", "jour_num", "jour_semaine", "Time_Period"])["Flow"]
              .sum()
              .reset_index()
        )

        # Normalisation éventuelle
        if normalize:
            df_daily["Flow_val"] = df_daily["Flow"] / df_daily["Flow"].max()
            ylabel = "Flux normalisé (0–1)"
        else:
            df_daily["Flow_val"] = df_daily["Flow"]
            ylabel = "Flux total journalier"

        # Scatter colored by Time_Period
        for period, color in period_colors.items():
            sub = df_daily[df_daily["Time_Period"] == period]
            if not sub.empty:
                ax.scatter(
                    sub["jour_num"],
                    sub["Flow_val"],
                    c=color,
                    alpha=0.7,
                    s=40,
                    label=period
                )

        # Axe x : jours de la semaine
        ax.set_xticks(range(7))
        ax.set_xticklabels(
            ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"],
            rotation=20
        )

        ax.set_title(f"Dataset #{i+1}")
        ax.grid(True)

        if i == 0:
            ax.set_ylabel(ylabel)

    # Légende commune
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          label=per, markerfacecolor=col, markersize=8)
               for per, col in period_colors.items()]
    fig.legend(handles=handles, labels=period_colors.keys(),
               loc="upper center", ncol=len(period_colors))

    plt.tight_layout()
    plt.show()







# End of src/plot.py