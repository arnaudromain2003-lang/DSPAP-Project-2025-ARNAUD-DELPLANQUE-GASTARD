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


# Creation of function to plot sum of validations
def plot_validations(df_filtered_tram, df_filtered_bus, df_filtered_subway):
    df_filtered_tram.groupby(['date'])['Flow'].sum().plot(figsize = (15,5), title = 'Number of validations per day for tramway', xlabel = 'Date', ylabel = 'Number of validations')
    plt.show()
    df_filtered_bus.groupby(['date'])['Flow'].sum().plot(figsize = (15,5), title = 'Number of validations per day for bus', xlabel = 'Date', ylabel = 'Number of validations')
    plt.show()
    df_filtered_subway.groupby(['date'])['Flow'].sum().plot(figsize = (15,5), title = 'Number of validations per day for subway', xlabel = 'Date', ylabel = 'Number of validations')
    plt.show()
    
# Creation of function to plot week data
def plot_week_data(df_tram, df_bus, df_subway, week_nbr, color=True, superpose=True):
    # Création des filtres pour obtenir la semaine souhaitée
    filter_week_tram = (df_tram['date'].dt.isocalendar().week == week_nbr)
    filter_week_bus = (df_bus['date'].dt.isocalendar().week == week_nbr)
    filter_week_subway = (df_subway['date'].dt.isocalendar().week == week_nbr)

    # Filtrer les dataframes pour la semaine souhaitée
    df_week_tram = df_tram[filter_week_tram].copy()
    df_week_bus = df_bus[filter_week_bus].copy()
    df_week_subway = df_subway[filter_week_subway].copy()

    df_week_tram['jour'] = df_week_tram['date'].dt.day_name()
    df_week_bus['jour'] = df_week_bus['date'].dt.day_name()
    df_week_subway['jour'] = df_week_subway['date'].dt.day_name()


    # Définir une palette de couleurs (une couleur par jour)
    couleurs = {
        'Monday': 'red',
        'Tuesday': 'blue',
        'Wednesday': 'green',
        'Thursday': 'orange',
        'Friday': 'purple',
        'Saturday': 'brown',
        'Sunday': 'pink'
    }

    if (not color) & (not superpose):  # On affiche chaque jour avec la même couleur
        return plot_validations(df_week_tram, df_week_bus, df_week_subway)

    elif color & superpose: # On affiche chaque jour l'un sur l'autre tout en gardant un plot pour le bus et un pour le tram
        df_week_bus['hour'] = df_week_bus['date'].dt.hour + df_week_bus['date'].dt.minute / 60
        df_plot_bus = df_week_bus.groupby(['hour', 'jour'])['Flow'].sum().reset_index()
        df_week_tram['hour'] = df_week_tram['date'].dt.hour + df_week_tram['date'].dt.minute / 60
        df_plot_tram = df_week_tram.groupby(['hour', 'jour'])['Flow'].sum().reset_index()
        df_week_subway['hour'] = df_week_subway['date'].dt.hour + df_week_subway['date'].dt.minute / 60
        df_plot_subway = df_week_subway.groupby(['hour', 'jour'])['Flow'].sum().reset_index()


        # Tracer avec Seaborn pour les tram
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=df_plot_tram, x='hour', y='Flow', hue='jour', palette=couleurs, legend='full')
        plt.title(f"Nombre de validations par heure pour chaque jour de la semaine n°{week_nbr} pour les tramways")
        plt.xlabel("Heure de la journée")
        plt.ylabel("Nombre de validations")
        plt.legend(title='Jour')
        plt.xticks(range(0, 24))  # Afficher les heures de 0 à 23
        plt.show()

        # Tracer avec Seaborn pour les bus
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=df_plot_bus, x='hour', y='Flow', hue='jour', palette=couleurs, legend='full')

        plt.title(f"Nombre de validations par heure pour chaque jour de la semaine n°{week_nbr} pour les bus")
        plt.xlabel("Heure de la journée")
        plt.ylabel("Nombre de validations")
        plt.legend(title='Jour')
        plt.xticks(range(0, 24))  # Afficher les heures de 0 à 23
        plt.show()
        
        # Tracer avec Seaborn pour le métro
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=df_plot_subway, x='hour', y='Flow', hue='jour', palette=couleurs, legend='full')

        plt.title(f"Nombre de validations par heure pour chaque jour de la semaine n°{week_nbr} pour le métro")
        plt.xlabel("Heure de la journée")
        plt.ylabel("Nombre de validations")
        plt.legend(title='Jour')
        plt.xticks(range(0, 24))  # Afficher les heures de 0 à 23
        plt.show()
        
    elif color & (not superpose):
        df_daily_bus = df_week_bus.groupby(['date', 'jour'])['Flow'].sum().reset_index()
        df_daily_tram = df_week_tram.groupby(['date', 'jour'])['Flow'].sum().reset_index()
        df_daily_subway = df_week_subway.groupby(['date', 'jour'])['Flow'].sum().reset_index()
        

        # Tracer avec Seaborn pour les bus
        plt.figure(figsize=(15, 5))
        sns.lineplot(data=df_daily_bus, x='date', y='Flow', hue='jour', palette=couleurs)
        plt.title(f'Number of validations per day for bus (Week {week_nbr} of 2020)')
        plt.xlabel('Date')
        plt.ylabel('Number of validations')
        plt.show()

        # Tracer avec Seaborn pour les tramways
        plt.figure(figsize=(15, 5))
        sns.lineplot(data=df_daily_tram, x='date', y='Flow', hue='jour', palette=couleurs)
        plt.title(f'Number of validations per day for tramway (Week {week_nbr} of 2020)')
        plt.xlabel('Date')
        plt.ylabel('Number of validations')
        plt.show()
        
        # Tracer avec Seaborn pour le metro
        plt.figure(figsize=(15, 5))
        sns.lineplot(data=df_daily_subway, x='date', y='Flow', hue='jour', palette=couleurs)
        plt.title(f'Number of validations per day for tramway (Week {week_nbr} of 2020)')
        plt.xlabel('Date')
        plt.ylabel('Number of validations')
        plt.show()        

    else:
        return "Invalid combination of parameters. Please set superpose to True if color is True."

# Function to plot a day
def plot_day_data(df_tram, df_bus, df_subway, date_str):
    # Convertir la chaîne de caractères en objet datetime
    date = pd.to_datetime(date_str).date()

    # Filtrer les dataframes pour la date souhaitée
    df_day_tram = df_tram[df_tram['date_only'] == date].copy()
    df_day_bus = df_bus[df_bus['date_only'] == date].copy()
    df_day_subway = df_subway[df_subway['date_only'] == date].copy()

    # Extraire l'heure et les minutes pour un tracé plus précis
    df_day_tram['hour'] = df_day_tram['date'].dt.hour + df_day_tram['date'].dt.minute / 60
    df_day_bus['hour'] = df_day_bus['date'].dt.hour + df_day_bus['date'].dt.minute / 60
    df_day_subway['hour'] = df_day_subway['date'].dt.hour + df_day_subway['date'].dt.minute / 60


    # Grouper par heure pour obtenir le nombre total de validations par heure
    df_plot_tram = df_day_tram.groupby('hour')['Flow'].sum().reset_index()
    df_plot_bus = df_day_bus.groupby('hour')['Flow'].sum().reset_index()
    df_plot_subway = df_day_subway.groupby('hour')['Flow'].sum().reset_index()

    # Tracer les validations pour le tramway
    plt.figure(figsize=(15, 5))
    plt.plot(df_plot_tram['hour'], df_plot_tram['Flow'], marker='o')
    plt.title(f'Number of validations per hour for tramway on {date_str}')
    plt.xlabel('Hour of the day')
    plt.ylabel('Number of validations')
    plt.xticks(range(0, 24))  # Afficher les heures de 0 à 23
    plt.grid()
    plt.show()

    # Tracer les validations pour le bus
    plt.figure(figsize=(15, 5))
    plt.plot(df_plot_bus['hour'], df_plot_bus['Flow'], marker='o', color='orange')
    plt.title(f'Number of validations per hour for bus on {date_str}')
    plt.xlabel('Hour of the day')
    plt.ylabel('Number of validations')
    plt.xticks(range(0, 24))  # Afficher les heures de 0 à 23
    plt.grid()
    plt.show()
    
    # Tracer les validations pour le metro
    plt.figure(figsize=(15, 5))
    plt.plot(df_day_subway['hour'], df_plot_subway['Flow'], marker='o', color='green')
    plt.title(f'Number of validations per hour for metro on {date_str}')
    plt.xlabel('Hour of the day')
    plt.ylabel('Number of validations')
    plt.xticks(range(0, 24))  # Afficher les heures de 0 à 23
    plt.grid()
    plt.show()
    
def merge_dataframes(df_bus, df_tramway, df_subway):
    """Fusionne les dataframes de bus et de tramway en un seul dataframe global.
    Args:
        df_bus (pd.DataFrame): dataframe contenant les données de flow des bus
        df_tramway (pd.DataFrame): dataframe contenant les données de flow des tramways
    Returns:
        pd.DataFrame: dataframe global fusionné
    """
    df_bus = df_bus.copy()
    df_tramway = df_tramway.copy()
    df_subway = df_subway.copy()
    df_bus['Transport_Type'] = 'Bus'
    df_tramway['Transport_Type'] = 'Tram'
    df_subway['Transport_Type'] = 'Subway'


    df_global = pd.concat([df_bus, df_tramway, df_subway], ignore_index=True)
    return df_global

def give_time_period(df):
    """Ajoute une colonne au df pour lui indiquer si on se trouve en semaine classique, en semaine de vacances (précision du type de vacances) et de la période covid.
    Args:
        df (pd.DataFrame): dataframe contenant les données de flow des transports
    Returns:
        pd.DataFrame: dataframe avec la colonne supplémentaire "Time_Period"
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date

    # Définir les périodes de vacances scolaires en France pour l'année scolaire 2019/2020
    period = {
        "Vacances de Noel": (pd.to_datetime('2019-12-21').date(), pd.to_datetime('2020-01-05').date()),
        "Vacances d'hiver": (pd.to_datetime('2020-02-15').date(), pd.to_datetime('2020-03-01').date()),
        "Période COVID": (pd.to_datetime('2020-03-17').date(), pd.to_datetime('2020-05-11').date()),
        "Fête des Lumières": (pd.to_datetime('2019-12-05').date(), pd.to_datetime('2019-12-08').date())
        }
    def classify_date(date):
        for period_name, (start_date, end_date) in period.items():
            if start_date <= date <= end_date:
                return period_name
        else:
            return "Semaine classique"
        
    df['Time_Period'] = df['date_only'].apply(classify_date)
    return df

def plot_day_variation(df_global):
    """
    Crée un scatter plot pour visualiser les valeurs journalières de flow,
    avec une coloration basée sur la colonne Time_Period.
    """
    df = df_global.copy()
    df['date'] = pd.to_datetime(df['date'])
    df["date_only"] = df['date'].dt.date
    df['jour_semaine'] = df['date'].dt.day_name()
    df['jour_num'] = df['date'].dt.dayofweek  # Pour trier les jours de la semaine
    df_daily = df.groupby(['date_only', 'jour_num', 'jour_semaine', 'Time_Period'])['Flow'].sum().reset_index()

    # Dictionnaire pour associer une couleur à chaque Time_Period
    period_colors = {
        "Semaine classique": "blue",
        "Vacances de Noel": "red",
        "Vacances d'hiver": "green",
        "Période COVID": "purple",
        "Fête des Lumières": "orange"
    }

    # Tracer le scatter plot
    plt.figure(figsize=(12, 6))
    for period, color in period_colors.items():
        subset = df_daily[df_daily['Time_Period'] == period]
        #print(subset.head())
        plt.scatter(
            x=subset['jour_num'],
            y=subset['Flow'],
            c=[color] * len(subset),
            label=period,
            alpha=0.7,
            s=50
        )

    # Personnalisation du plot
    plt.xticks(ticks=range(7), labels=['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    plt.xlabel('Jour de la semaine')
    plt.ylabel('Flux total journalier')
    plt.title('Flux journalier par jour de la semaine et période')
    plt.legend(title='Période')
    plt.grid(True)
    plt.show()

def cluster_kmeans_days(df, n_clusters=3, date_limit='2020-03-16'):
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

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd

def plot_cluster(df_daily, cluster_color, cluster_column="cluster", flow_column="Flow"):
    n_clusters = len(df_daily[cluster_column].unique())
    # Crée une ListedColormap à partir de cluster_color
    cmap = ListedColormap(cluster_color)
    plt.figure(figsize=(12, 6))
    plt.scatter(
        df_daily['date_only'],
        df_daily[flow_column],
        c=df_daily[cluster_column],
        cmap=cmap,
        vmin=0,
        vmax=n_clusters-1
    )
    plt.xlabel('Date')
    plt.ylabel('Flow')
    plt.title('Clusters de jours par flow de transport')
    plt.legend(title='Cluster', handles=[
        plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                  markerfacecolor=cluster_color[i], markersize=10)
        for i in range(n_clusters)
    ])
    plt.show()

def plot_cluster_distribution(df_daily, cluster_color=["#ff7700", '#2ca02c', '#d62728', '#9467bd']):
    df = df_daily.copy()
    df["day_of_week"] = pd.to_datetime(df['date_only']).dt.day_name()
    jours_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=jours_order, ordered=True)
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
            print(df_month.head())
            # Créer une matrice pour le calendrier
            cal = calendar.monthcalendar(year, month)
            n_weeks = len(cal)
            days_matrix = np.zeros((n_weeks, 7)) - 1  # -1 = jour hors du mois

            # Remplir la matrice avec les clusters
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                date = pd.to_datetime(pd.Timestamp(year=year, month=month, day=day)).date()
                print(day, date, date in df_month.date_only.values)
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
    # Fusionner les DataFrames
    df = pd.merge(df_global, df_cluster[['date_only', cluster_col]], on='date_only', how='left')
    # Convertir la colonne 'date' en datetime
    df['datetime'] = pd.to_datetime(df['date'])
    # Extraire l'heure et la minute sous forme de chaîne (ex: "08:00")
    df['time'] = df['datetime'].dt.time
    # Calculer la somme du flux pour chaque minute et chaque cluster
    df_typical_days = df.groupby(['time', cluster_col])['Flow'].sum().reset_index()
    # Normaliser par le nombre de jours de cluster ayant une valeur pour cet instant
    normalize_table = df[['time', cluster_col]].value_counts()
    df_typical_days['Flow'] = df_typical_days.apply(
        lambda row: row['Flow'] / normalize_table[(row['time'], row[cluster_col])],
        axis=1
    )
    return df_typical_days

def plot_typical_days_per_cluster(df_typical_days, cluster_colors, cluster_col='cluster', flow_col='Flow'):
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

