"""
clustering module: functions for clustering days based on transport flow.
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import calendar


colors = ['#1f77b4', "#ff7700", '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def cluster_kmeans_days(df, n_clusters=3, date_limit='2020-03-16'):
    """Cluster days based on transport flow using KMeans.

    Args:
        df (pd.DataFrame): DataFrame with 'date' and 'Flow' columns.
        n_clusters (int, optional): Number of clusters. Defaults to 3.
        date_limit (str, optional): Date limit for filtering data. Defaults to '2020-03-16'.

    Returns:
        pd.DataFrame: DataFrame with daily flow and cluster labels.
    """
    # Charger les données
    df_daily = df.copy()
    df_daily["date_only"] = df_daily["date"].dt.date
    if date_limit != None:
        filter = df_daily['date_only'] < pd.to_datetime(date_limit).date()
        df_daily = df_daily[filter]
    df_daily = df_daily.groupby('date_only').agg({'Flow': 'sum'}).reset_index()

    # Normalisation
    scaler = StandardScaler()
    df_daily['flow_normalized'] = scaler.fit_transform(df_daily[['Flow']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_daily['cluster'] = kmeans.fit_predict(df_daily[['flow_normalized']])
    return df_daily

def plot_cluster(df_daily, cluster_column = "cluster", flow_column="Flow", cmap=["#ff7700", '#2ca02c', '#d62728', '#9467bd']):
    """Plot clusters of days based on transport flow.

    Args:
        df_daily (pd.DataFrame): DataFrame with daily flow and cluster labels.
        cluster_column (str, optional): Name of the column containing cluster labels. Defaults to "cluster".
        flow_column (str, optional): Name of the column containing flow values. Defaults to "Flow".
        cmap (list, optional): List of colors for clusters. Defaults to ["#ff7700", '#2ca02c', '#d62728', '#9467bd'].
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(df_daily['date_only'], df_daily[flow_column], c=df_daily[cluster_column], cmap=ListedColormap(cmap))
    plt.xlabel('Date')
    plt.ylabel('Flow')
    plt.title('Clusters de jours par flow de transport')
    plt.show()


def plot_cluster_distribution(df_daily, cluster_color = ["#ff7700", '#2ca02c', '#d62728', '#9467bd']):
    """
    Plot the distribution of clusters over the days of the week.
    Args:
        df_daily (pd.DataFrame): DataFrame with daily flow and cluster labels.
        cluster_color (list, optional): List of colors for clusters. Defaults to ["#ff7700", '#2ca02c', '#d62728', '#9467bd'].
    Returns:
        A bar plot showing the count of days in each cluster for each day of the week.
    """
    df=df_daily.copy()
    df["day_of_week"]=pd.to_datetime(df['date_only']).dt.day_name()

    # Définir l'ordre des jours
    jours_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Convertir la colonne en type Categorical avec l'ordre spécifié
    df['day_of_week'] = pd.Categorical(
        df['day_of_week'],
        categories=jours_order,
        ordered=True
    )
    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=df,
        x='day_of_week',
        hue='cluster',
        palette=cluster_color,
        order=jours_order
    )
    plt.xlabel('Jour de la semaine')
    plt.ylabel('Nombre de jours')
    plt.title('Répartition des clusters par jour de la semaine')
    plt.xticks(rotation=45)
    plt.legend(title='Cluster')
    plt.show()


def plot_cluster_calendar(df_daily, year, month, cluster_col='cluster', cluster_colors = ['#1f77b4', "#ff7700", '#2ca02c', '#d62728', '#9467bd']):
    """Plot a calendar view of clusters for a given month and year.

    Args:
        df_daily (pd.DataFrame): Daily DataFrame with daily flow and cluster labels.
        year (int): Year for the calendar view.
        month (int): Month for the calendar view.
        cluster_col (str, optional): Name of the column containing cluster labels. Defaults to 'cluster'.
        cluster_colors (list, optional): List of colors for clusters. Defaults to ['#1f77b4', "#ff7700", '#2ca02c', '#d62728', '#9467bd'].
    Raises:
        ValueError: If the number of clusters exceeds the number of defined colors.
    Returns:
        A calendar plot showing the clusters for each day of the specified month and year.
    """
    df_daily_index = df_daily.copy()
    df_daily_index["year"] = pd.to_datetime(df_daily_index["date_only"]).dt.year
    df_daily_index["month"] = pd.to_datetime(df_daily_index["date_only"]).dt.month

    df_daily_index.set_index('date_only', inplace=True)

    # Définir une palette de couleurs pour les clusters
    if df_daily_index[cluster_col].nunique() > len(cluster_colors):
        raise ValueError("Le nombre de clusters dépasse le nombre de couleurs définies.")
    if df_daily_index[cluster_col].nunique() < len(cluster_colors):
        val = df_daily_index[cluster_col].nunique() + 1
        cluster_colors = cluster_colors[:val]
    cmap = ListedColormap(cluster_colors)

    # Filtrer les données pour le mois/année souhaité
    filter_month = df_daily_index['month'] == month
    filter_year = df_daily_index['year'] == year
    df_month = df_daily_index[filter_year & filter_month].copy()

    # Créer une matrice vide pour le mois
    cal = calendar.monthcalendar(year, month)
    n_weeks = len(cal)
    days_matrix = np.zeros((n_weeks, 7)) - 1  # -1 = jour hors du mois

    # Remplir la matrice avec les clusters
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        date = pd.to_datetime(pd.Timestamp(year=year, month=month, day=day)).date()
        if date in df_month.index:
            week_idx = next(i for i, week in enumerate(cal) if day in week)  # Trouver la semaine
            day_idx = date.weekday()  # Trouver le jour de la semaine (0=Lundi, 6=Dimanche)
            days_matrix[week_idx, day_idx] = df_month.loc[date, cluster_col]

    
    # Tracer le calendrier
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(days_matrix, cmap=cmap, vmin=-1, vmax=len(cluster_colors)-1)

    # Ajouter les jours du mois
    for i in range(n_weeks):
        for j in range(7):
            day = cal[i][j]
            if day != 0:
                ax.text(j, i, day, ha='center', va='center', color='black')

    # Personnaliser le plot
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    ax.set_yticks([])
    ax.set_title(f'Calendrier des clusters - {calendar.month_name[month]} {year}')
    plt.legend(handles = [
        plt.Line2D([0], [0], marker='o', color='w', label="Hors de la période d'étude", markerfacecolor=cluster_colors[0], markersize=10)
    ] + [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', markerfacecolor=color, markersize=10)
        for i, color in enumerate(cluster_colors[1:])
    ],
        bbox_to_anchor=(1.05, 1), 
        loc='best', 
        title='Clusters')
    plt.show()


def plot_cluster_calendars_as_subplots(df_daily, start_year, start_month, end_year, end_month, cluster_col='cluster', cluster_colors=['white', "#ff7700", '#2ca02c', '#d62728', '#9467bd']):
    """
    Plot multiple months as subplots in a calendar format, showing clusters for each day.
    Args:
        df_daily (pd.DataFrame): DataFrame with daily flow and cluster labels.
        start_year (int): The starting year for the calendar plots.
        start_month (int): The starting month for the calendar plots.
        end_year (int): The ending year for the calendar plots.
        end_month (int): The ending month for the calendar plots.
        cluster_col (str, optional): The column name for cluster labels. Defaults to 'cluster'.
        cluster_colors (list, optional): List of colors for clusters. Defaults to ['white', "#ff7700", '#2ca02c', '#d62728', '#9467bd'].
    Raises:
        ValueError: If the number of clusters exceeds the number of defined colors.
    Returns:
        A series of subplots showing calendars for each month in the specified range, colored by cluster.
    """
    # Calculer le nombre total de mois à afficher
    n_months = (end_year - start_year) * 12 + (end_month - start_month + 1)

    # Créer une figure avec des subplots (2 colonnes pour une meilleure disposition)
    fig, axes = plt.subplots(nrows=n_months // 2 + (1 if n_months % 2 else 0), ncols=2, figsize=(12, 4 * (n_months // 2 + 1)))
    axes = axes.flatten()  # Aplatir pour faciliter l'indexation

    # Préparer les données une fois pour toute
    df_daily_index = df_daily.copy()
    df_daily_index["year"] = pd.to_datetime(df_daily_index["date_only"]).dt.year
    df_daily_index["month"] = pd.to_datetime(df_daily_index["date_only"]).dt.month
    df_daily_index.set_index('date_only', inplace=True, drop=False)

    # Définir la palette de couleurs
    n_clusters = df_daily_index[cluster_col].nunique()
    if n_clusters >= len(cluster_colors):
        raise ValueError("Le nombre de clusters dépasse le nombre de couleurs définies.")
    cluster_colors = cluster_colors[:n_clusters + 1]  # +1 pour la couleur "hors période"
    cmap = ListedColormap(cluster_colors)

    # Créer les handles pour la légende (une seule fois)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label="Hors étude", markerfacecolor=cluster_colors[0], markersize=10)
    ] + [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', markerfacecolor=color, markersize=10)
        for i, color in enumerate(cluster_colors[1:n_clusters + 1])
    ]

    # Parcourir chaque mois et remplir les subplots
    ax_idx = 0
    for year in range(start_year, end_year + 1):
        start_m = start_month if year == start_year else 1
        end_m = end_month if year == end_year else 12
        for month in range(start_m, end_m + 1):
            # Filtrer les données pour le mois/année en cours
            df_month = df_daily_index[(df_daily_index['year'] == year) & (df_daily_index['month'] == month)].copy()
            df_month["date_only"] = pd.to_datetime(df_month["date_only"]).dt.date
            # Créer une matrice pour le calendrier
            cal = calendar.monthcalendar(year, month)
            n_weeks = len(cal)
            days_matrix = np.zeros((n_weeks, 7)) - 1  # -1 = jour hors du mois

            # Remplir la matrice avec les clusters
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                date = pd.to_datetime(pd.Timestamp(year=year, month=month, day=day)).date()
                if date in df_month.date_only.values:
                    week_idx = next(i for i, week in enumerate(cal) if day in week)
                    day_idx = date.weekday()  # 0=Lundi, 6=Dimanche
                    days_matrix[week_idx, day_idx] = df_month.loc[df_month["date_only"] == date, cluster_col].values[0]

            # Tracer le calendrier dans le subplot correspondant
            ax = axes[ax_idx]
            ax.imshow(days_matrix, cmap=cmap, vmin=-1, vmax=n_clusters)

            # Ajouter les jours du mois
            for i in range(n_weeks):
                for j in range(7):
                    day = cal[i][j]
                    if day != 0:
                        ax.text(j, i, day, ha='center', va='center', color='black')

            # Personnaliser le subplot
            ax.set_xticks(range(7))
            ax.set_xticklabels(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
            ax.set_yticks([])
            ax.set_title(f'{calendar.month_name[month]} {year}')

            ax_idx += 1

    # Ajouter une légende commune
    fig.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper center',title='Clusters')
    plt.tight_layout()
    plt.show()


def create_typical_days_per_cluster(df_global, df_cluster, cluster_col='cluster'):
    """
    Create typical days per cluster by merging global and cluster data, aggregating flow, and associating clusters.

    Args:
        df_global (pd.DataFrame): Global dataframe with flow data.
        df_cluster (pd.DataFrame): Dataframe with cluster labels.
        cluster_col (str, optional): The column name for cluster labels. Defaults to 'cluster'.

    Returns:
        pd.DataFrame: Dataframe with typical days per cluster.
    """
    # 1. Fusion des données
    df = pd.merge(df_global, df_cluster[['date_only', 'cluster']], on='date_only', how='left')[["date","Flow","cluster"]]
    df["time"] = pd.to_datetime(df['date']).dt.time
    df["date_only"] = pd.to_datetime(df["date"]).dt.date

    # 2. Agrégation
    df_test = df.groupby("date").agg({'Flow': 'sum'}).reset_index()

    # 3. Conversion des dates
    df_test['date_only'] = pd.to_datetime(df_test['date']).dt.date
    df_cluster['date_only'] = pd.to_datetime(df_cluster['date_only']).dt.date

    # 4. Fusion avec df_cluster
    df_average = pd.merge(
        df_test,
        df_cluster[['date_only', 'cluster']],
        on='date_only',
        how='left'  # ou 'inner' si vous voulez exclure les dates sans cluster
    )

    # 5. Gestion des valeurs manquantes (si how='left')
    df_average = df_average.dropna(subset=['cluster'])  # ou .fillna()
    df_average["time"] = pd.to_datetime(df_average['date']).dt.time
    return df_average


def plot_typical_days_per_cluster(df_typical_days, cluster_colors, cluster_col='cluster', flow_col='Flow'):
    """
    Plot typical days per cluster using time in decimal hours and flow values.

    Args:
        df_typical_days (pd.DataFrame): Dataframe with typical days data.
        cluster_colors (list): List of colors for clusters.
        cluster_col (str, optional): The column name for cluster labels. Defaults to 'cluster'.
        flow_col (str, optional): The column name for flow values. Defaults to 'Flow'.
    """
    # Convertir l'heure en format numérique pour le tracé
    df_typical_days['time_numeric'] = df_typical_days['time'].apply(
    lambda t: t.hour + t.minute/60
    )

    # Tracé
    plt.figure(figsize=(16, 8))
    sns.lineplot(
        data=df_typical_days,
        x='time_numeric',
        y='Flow',
        hue='cluster',
        palette=cluster_colors
    )
    plt.xlabel('Heure de la journée (heures décimales)')
    plt.ylabel('Flux moyen')
    plt.title('Journées types par cluster (granularité minute)')
    plt.xticks(range(0, 24))
    plt.grid()
    plt.tight_layout()
    plt.show()