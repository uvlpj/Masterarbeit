#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

#%%
from data_preprocessor import *
# %%
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Pfad zu den CSV-Dateien (alle CSV-Dateien im Ordner "res")
csv_files_r = glob.glob("/home/siefert/projects/Masterarbeit/Data/Results/R_Results/res_different_mtry/*.csv")
csv_files_python = glob.glob("/home/siefert/projects/Masterarbeit/Data/Results/Python_Results/python_res_different_mtry/*.csv")   

# Deaktiviere LaTeX für matplotlib
plt.rcParams['text.usetex'] = False

# Setze den Hintergrundstil auf weiß
plt.style.use('seaborn-whitegrid')

#%%
csv_files = csv_files_r + csv_files_python
#%%
#specifications = [
#    'nott_doy_bt', 'nott_doy_rf',
#    'nott_month_bt', 'nott_month_rf',
#    'tt_doy_bt', 'tt_doy_rf',
#    'tt_month_bt', 'tt_month_rf'
#]

#specifications = [
#    'nott_day_lagged_bt', 'nott_day_lagged_rf',
#    'nott_month_lagged_bt', 'nott_month_lagged_rf',
#    'tt_day_lagged_bt', 'tt_day_lagged_rf',
#    'tt_month_lagged_bt', 'tt_month_lagged_rf'
#    'nott_day_notlagged_bt', 'nott_day_notlagged_rf',
#    'nott_month_notlagged_bt', 'nott_month_notlagged_rf',
#    'tt_day_notlagged_bt', 'tt_day_notlagged_rf',
#    'tt_month_notlagged_bt', 'tt_month_notlagged_rf'
#]


specifications = [
    'nott_day_lagged_', 'nott_day_lagged_rf',
    'nott_month_lagged_bt', 'nott_month_lagged_rf',
    'tt_day_lagged_bt', 'tt_day_lagged_rf',
    'tt_month_lagged_bt', 'tt_month_lagged_rf'
    'nott_day_notlagged_bt', 'nott_day_notlagged_rf',
    'nott_month_notlagged_bt', 'nott_month_notlagged_rf',
    'tt_day_notlagged_bt', 'tt_day_notlagged_rf',
    'tt_month_notlagged_bt', 'tt_month_notlagged_rf'
]


#%%


def plot_cumulative_crps(specification):
    """
    Plot the cumulative CRPS for the three packages for a specific specification.
    """
    plt.figure(figsize=(16, 8), dpi=300)

    # Farbschema und Linienstile für die verschiedenen Pakete
    colors = {'ranger': 'b', 'quantregForest': 'g', 'sklearn': 'r'}
    labels = {'sklearn': 'Sklearn', 'ranger': 'Ranger', 'quantregForest': 'QuantregForest'}
    linestyles = {'sklearn': '-', 'ranger': '--', 'quantregForest': ':'}
    
    # Überprüfen, ob die CSV-Dateien die spezifische Spezifikation enthalten
    for package in ['sklearn', 'ranger', 'quantregForest']:
        # Verwende einen Regex, um exakte Übereinstimmung zu erzwingen (Paketname + Spezifikation)
        pattern = re.compile(f"^{package}_{specification}\.csv$")
        filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
        print(f"Filtered files for {package} => {filtered_files}") 
        
        # Überprüfen, ob eine passende Datei gefunden wurde
        if filtered_files:
            # Lese die erste passende Datei ein (wir nehmen an, dass es nur eine pro Paket und Spezifikation gibt)
            df = pd.read_csv(filtered_files[0])
            print(df)
            print(f"Columns in {package} data: {df.columns.tolist()}")

            # Konvertiere 'date_time' zu datetime-Objekten
            df['date_time'] = pd.to_datetime(df['date_time'])

            # Kumulierten CRPS berechnen
            df['cumulative_crps'] = df['crps'].cumsum()
            print(df['cumulative_crps'].head(10))
            print(df['cumulative_crps'].tail(10))

            # Plot der kumulierten CRPS mit verschiedenen Linienstilen
            plt.plot(df['date_time'], df['cumulative_crps'], label=labels[package], 
                     color=colors[package], linestyle=linestyles[package])

    # Plot-Details
    plt.title(f'Cumulative CRPS for {specification}', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylim(df['cumulative_crps'].min() - 100, df['cumulative_crps'].max() + 100)
    plt.ylabel('Cumulative CRPS', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')  # Kleine Legende links oben
    plt.tight_layout()  # Optimiert das Layout, um Überlappungen zu vermeiden

    # Plot anzeigen
    plt.show()


#%%
plot_cumulative_crps('nott_doy_bt') 
#%%
plot_cumulative_crps('nott_doy_rf')
#%%
plot_cumulative_crps('nott_month_bt') 
#%%
plot_cumulative_crps('nott_month_rf') 
#%%
plot_cumulative_crps('tt_doy_bt')
#%%
plot_cumulative_crps('tt_doy_rf')
#%%
plot_cumulative_crps('tt_month_bt')
#%%
plot_cumulative_crps('tt_month_rf')
#%%

def plot_cumulative_se(specification):
    """
    Plot the cumulative Squared Error for the three packages for a specific specification.
    """
    plt.figure(figsize=(16, 8), dpi=300)

    # Farbschema und Linienstil für die verschiedenen Pakete
    colors = {'ranger': 'b', 'quantregForest': 'g', 'sklearn': 'r'}
    labels = {'sklearn': 'Sklearn', 'ranger': 'Ranger', 'quantregForest': 'QuantregForest'}
    linestyles = {'ranger': '-', 'quantregForest': '--', 'sklearn': ':'}  # Verschiedene Linienstile
    
    # Überprüfen, ob die CSV-Dateien die spezifische Spezifikation enthalten
    # Überprüfen, ob die CSV-Dateien die spezifische Spezifikation enthalten
    for package in ['sklearn', 'ranger', 'quantregForest']:
        # Verwende einen Regex, um exakte Übereinstimmung zu erzwingen (Paketname + Spezifikation)
        pattern = re.compile(f"^{package}_{specification}\.csv$")
        filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
        print(f"Filtered files for {package} => {filtered_files}") 
        
        # Überprüfen, ob eine passende Datei gefunden wurde
        if filtered_files:
            # Lese die erste passende Datei ein (wir nehmen an, dass es nur eine pro Paket und Spezifikation gibt)
            df = pd.read_csv(filtered_files[0])
            print(df)
            print(f"Columns in {package} data: {df.columns.tolist()}")

            # Konvertiere 'date_time' zu datetime-Objekten
            df['date_time'] = pd.to_datetime(df['date_time'])

            # Kumulierten SE berechnen
            df['cumulative_se'] = df['se'].cumsum()
            print(df['cumulative_se'].head(10))
            print(df['cumulative_se'].tail(10))

            # Plot der kumulierten SE mit spezifischem Linienstil
            plt.plot(df['date_time'], df['cumulative_se'], 
                     label=labels[package], 
                     color=colors[package], 
                     linestyle=linestyles[package])

    # Plot-Details
    plt.title(f'Cumulative SE for {specification}', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    
    # Skalierung der Y-Achse zur besseren Sichtbarkeit von Unterschieden
    plt.ylim(df['cumulative_se'].min() - 100, df['cumulative_se'].max() + 100)

    plt.ylabel('Cumulative SE', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')  # Kleine Legende links oben
    plt.tight_layout()  # Optimiert das Layout, um Überlappungen zu vermeiden

    # Plot anzeigen
    plt.show()


#%%
plot_cumulative_se('nott_doy_bt')
#%%
plot_cumulative_se('nott_doy_rf')
#%%
plot_cumulative_se('nott_month_bt') 
#%%
plot_cumulative_se('nott_month_rf') 
#%%
plot_cumulative_se('tt_doy_bt')
#%%
plot_cumulative_se('tt_doy_rf')
#%%
plot_cumulative_se('tt_month_bt')
#%%
plot_cumulative_se('tt_month_rf')
#%%
def plot_mean_crps(specifications):
    """
    Plot the mean CRPS for the three packages for a specific specification.
    """
    plt.figure(figsize=(6, 4), dpi=300)

    # Color scheme and line styles for the different packages
    colors = {'ranger': 'b', 'quantregForest': 'g', 'sklearn': 'r'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}
    linestyles = {'sklearn': '-', 'ranger': '--', 'quantregForest': ':'}

    mean_crps_values = []

    # Check if CSV files contain the specified specification
    for package in ['sklearn', 'ranger', 'quantregForest']:
        # Use regex to ensure an exact match (package name + specification)
        pattern = re.compile(f"^{package}_{specifications}\.csv$")
        filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
        print(f"Filtered files for {package} => {filtered_files}")

        # Check if a matching file is found
        if filtered_files:
            # Read the first matching file (assuming only one per package and specification)
            df = pd.read_csv(filtered_files[0])
            print(df)
            print(f"Columns in {package} data: {df.columns.tolist()}")

            mean_crps = df['crps'].mean()
            mean_crps_values.append(mean_crps)

            # Plot the Mean CRPS with different line styles
            plt.scatter(package, mean_crps, label=labels[package],
                     color=colors[package], linestyle=linestyles[package])

    # Plot details
    plt.title(f'Mean CRPS for {specifications}', fontsize=18)
    plt.ylabel('Mean CRPS', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()

    # Show plot
    plt.show()

# %%
plot_mean_crps('tt_month_lagged_bt')
# %%
plot_mean_crps('tt_month_lagged_rf')
# %%
plot_mean_crps('nott_month_notlagged_bt')

#%%

def plot_mean_crps_subplot(specifications_list):
    """
    Plot the mean CRPS for each specification in a subplot grid with a shared legend.

    Durchschnitts CRPS Werte
    - Konstante Hyperparameter für alle drei Pakete => default werte von sklearn
    - Features (holiday, hour, weekday) und (month/day_of_year, timetrend/no_timetrend, y_{t-1}/no y_{t-1})
    - Für das Paket quantregForest wurde die Konstante entfernt
    """
    
    num_specs = len(specifications_list)
    num_cols = 4  # Anzahl der Spalten im Subplot-Raster
    num_rows = (num_specs + num_cols - 1) // num_cols  
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=300)
    axes = axes.flatten()

    # Gesamttitel hinzufügen
    fig.suptitle('Mean CRPS (with Intercept in quantregForest and no constant Hyperparameters)', fontsize=16)

    # Farb- und Linienstile für die verschiedenen Pakete
    colors = {'ranger': 'b', 'quantregForest': 'g', 'sklearn': 'r'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}
    
    # Loop durch jede Spezifikation und zeichne den Plot
    for i, specifications in enumerate(specifications_list):
        ax = axes[i]  # Zugriff auf die jeweilige Achse
        
        # Plot für die jeweilige Spezifikation
        for package in ['sklearn', 'ranger', 'quantregForest']:
            pattern = re.compile(f"^{package}_{specifications}\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]
            
            if filtered_files:
                df = pd.read_csv(filtered_files[0])
                mean_crps = df['crps'].mean()
                
                # Zeichne den Scatterplot
                ax.scatter(package, mean_crps, color=colors[package])
                
                # Zeige den Mean-CRPS-Wert unter dem Punkt an
                ax.text(package, mean_crps - 100, f'{mean_crps:.2f}',  # Formatierung auf 2 Dezimalstellen
                        color=colors[package], fontsize=8, ha='center', va='top')

        # Details für den jeweiligen Subplot
        ax.set_title(f'Mean CRPS for {specifications}', fontsize=10)
        ax.set_ylabel('Mean CRPS')
        ax.grid(True)
        ax.set_ylim(0, 4500)
        ax.set_xlim(-0.5, 2.5)

    # Gemeinsame Legende erstellen
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[package], markersize=8)
               for package in colors]
    legend_labels = [labels[package] for package in colors]
    
    # Positioniere die Legende unter dem Gesamttitel
    fig.legend(handles, legend_labels, loc='upper center', fontsize=10, ncol=len(colors), bbox_to_anchor=(0.5, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.subplots_adjust(wspace=0.4, top=0.85)  
    plt.show()

#%%
plot_mean_crps_subplot(['nott_day_lagged_bt', 'nott_day_lagged_rf',
    'nott_month_lagged_bt', 'nott_month_lagged_rf',
    'tt_day_lagged_bt', 'tt_day_lagged_rf',
    'tt_month_lagged_bt', 'tt_month_lagged_rf',
    'nott_day_notlagged_bt', 'nott_day_notlagged_rf',
    'nott_month_notlagged_bt', 'nott_month_notlagged_rf',
    'tt_day_notlagged_bt', 'tt_day_notlagged_rf',
    'tt_month_notlagged_bt', 'tt_month_notlagged_rf'])
# %%
# CRPS plot for different mtry values ---
# Hyperparameters are constant set to the value of sklearn and quantregForest has no Intercept

def plot_crps_vs_mtry(specifications_list):
    """
    Plot the mean CRPS values for each specification in a subplot grid with m_try values on the x-axis.

    """

    num_specs = len(specifications_list)
    num_cols = 4  # Number of columns in subplot grid
    num_rows = (num_specs + num_cols - 1) // num_cols  

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), dpi=300)
    axes = axes.flatten()

    # Overall title for the figure
    fig.suptitle('Mean CRPS for different m_try values', fontsize=16)

    # Colors and labels for different packages
    colors = {'sklearn': 'r', 'ranger': 'b', 'quantregForest': 'g'}
    labels = {'sklearn': 'sklearn', 'ranger': 'ranger', 'quantregForest': 'quantregForest'}

    for i, specification in enumerate(specifications_list):
        ax = axes[i]
        
        for package in ['sklearn', 'ranger', 'quantregForest']:
            pattern = re.compile(f"^{package}_{specification}_mtry(\\d+)\\.csv$")
            filtered_files = [file for file in csv_files if pattern.search(os.path.basename(file))]

            mtry_values = []
            crps_values = []

            for file in filtered_files:
                mtry_match = re.search(r'_mtry(\d+)', file)
                if mtry_match:
                    mtry = int(mtry_match.group(1))
                    df = pd.read_csv(file)
                    mean_crps = df['crps'].mean()

                    mtry_values.append(mtry)
                    crps_values.append(mean_crps)

            # Sort mtry and CRPS values for plotting
            sorted_indices = sorted(range(len(mtry_values)), key=lambda k: mtry_values[k])
            mtry_values = [mtry_values[idx] for idx in sorted_indices]
            crps_values = [crps_values[idx] for idx in sorted_indices]

            # Plot CRPS values over m_try for each package within a specification
            ax.plot(mtry_values, crps_values, marker='o', color=colors[package], 
                    label=labels[package], alpha=0.4)  # Set alpha for transparency

        # Set X-axis limits and ticks to ensure they are in steps of 1
        ax.set_xticks(range(min(mtry_values), max(mtry_values) + 1))  # +1 to include the max value

        # Add padding to the x-axis limits
        padding = 0.1  # Adjust the padding as needed
        ax.set_xlim(min(mtry_values) - padding, max(mtry_values) + padding)

        # Title and labels for each subplot
        ax.set_title(f'Mean CRPS for {specification}', fontsize=10)
        ax.set_xlabel('m_try')
        ax.set_ylabel('Mean CRPS')
        ax.set_ylim(0, 4500)
        ax.grid(True)

    # Shared legend
    handles = [plt.Line2D([0], [0], marker='o', color=colors[package], markersize=8, label=labels[package]) for package in colors]
    fig.legend(handles, labels.values(), loc='upper center', fontsize=10, ncol=len(colors), bbox_to_anchor=(0.5, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.subplots_adjust(wspace=0.4, top=0.85)  
    plt.show()

#%%
plot_crps_vs_mtry(['nott_day_lagged', 'nott_month_lagged', 'tt_day_lagged', 'tt_month_lagged',
                    'nott_day_notlagged', 'nott_month_notlagged', 'tt_day_notlagged', 'tt_month_notlagged'])
# %%
