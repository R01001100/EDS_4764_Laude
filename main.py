"""
REN-03: Cloud Cover Interference Analysis - Solar Power Generation
Student: Ruselle Laude
Student ID: TUPM-25-4764
Topic: Solar Power Generation Data - Cloud Cover Interference

Unique Filter Logic:
- Plant 1 only (PLANT_ID = 4135001)
- First 17 days (May 15 to May 31, 2020)
- Daytime hours 08:00 to 17:00
- Cloud classification: IRRADIATION < 0.3 kW/m² = Cloud-Affected
"""

# [REQUIRED MODULE]

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import seaborn as sns
import shutil
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# [SET STYLE FOR BETTER PLOTS]
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# [DEFINE PATHS AND PARAMETERS]

# File paths
project_root = Path("C:/Users/Laude/ComProg_Lab/EDS_4764_LaudeV2")

# Data and output folders
data_dir = project_root / "data"
output_dir = project_root / "outputs"
backup_dir = project_root / "backup"

# Create directories if they don't exist
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
backup_dir.mkdir(parents=True, exist_ok=True)

# Original data files
generation_file = data_dir / "Plant_1_Generation_Data.csv"
weather_file = data_dir / "Plant_1_Weather_Sensor_Data.csv"

# Output cleaned file
cleaned_file = data_dir / "dataset_cleaned.csv"

# Define unique filtering parameters
PLANT_ID = 4135001
START_DATE = "2020-05-15"
END_DATE = "2020-05-31"
START_HOUR = 8  # 08:00
END_HOUR = 17  # 17:00
IRRADIATION_THRESHOLD = 0.3  # kW/m² - cloud cover threshold


# [SEC 1: HELPER FUNCTIONS]


def load_csv_file(file_path):
    """Load CSV file with error handling."""
    try:
        df = pd.read_csv(file_path)
        return df  # Silent load (no print)
    except Exception as e:
        print(f"  [X] Error loading {file_path.name}: {e}")
        return None


def create_backup(df, filename):
    """Create backup of dataframe before processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"backup_{filename}_{timestamp}.csv"
    df.to_csv(backup_path, index=False)
    # Silent backup (no print)


def filter_by_unique_logic(df_weather, df_generation):
    """Apply unique filter logic: Plant 1, May 15-31, 08:00-17:00."""
    # Filter by plant ID
    df_weather = df_weather[df_weather["PLANT_ID"] == PLANT_ID].copy()
    df_generation = df_generation[df_generation["PLANT_ID"] == PLANT_ID].copy()

    # Parse datetime
    df_weather["DATETIME"] = pd.to_datetime(df_weather["DATE_TIME"])
    df_generation["DATETIME"] = pd.to_datetime(
        df_generation["DATE_TIME"], format="%d-%m-%Y %H:%M"
    )

    # Filter by date range
    df_weather = df_weather[
        (df_weather["DATETIME"] >= START_DATE) & (df_weather["DATETIME"] <= END_DATE)
    ].copy()
    df_generation = df_generation[
        (df_generation["DATETIME"] >= START_DATE)
        & (df_generation["DATETIME"] <= END_DATE)
    ].copy()

    # Filter by hour range
    df_weather["HOUR"] = df_weather["DATETIME"].dt.hour
    df_generation["HOUR"] = df_generation["DATETIME"].dt.hour

    df_weather = df_weather[
        (df_weather["HOUR"] >= START_HOUR) & (df_weather["HOUR"] <= END_HOUR)
    ].copy()
    df_generation = df_generation[
        (df_generation["HOUR"] >= START_HOUR) & (df_generation["HOUR"] <= END_HOUR)
    ].copy()

    return df_weather, df_generation


def clean_data(df):
    """Handle missing values, duplicates, and data types."""
    initial_rows = len(df)

    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)

    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        # Fill numeric columns with median
        for col in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    return df


def merge_datasets(df_weather, df_generation):
    """Merge weather and generation data on DATETIME and PLANT_ID only."""
    # Group generation data by DATETIME and calculate averages
    df_generation_agg = (
        df_generation.groupby(["DATETIME", "PLANT_ID"])
        .agg(
            {
                "DC_POWER": "sum",
                "AC_POWER": "sum",
                "DAILY_YIELD": "mean",
                "TOTAL_YIELD": "mean",
            }
        )
        .reset_index()
    )

    # Merge on DATETIME and PLANT_ID only
    merged_df = pd.merge(
        df_generation_agg,
        df_weather,
        on=["DATETIME", "PLANT_ID"],
        how="inner",
    )

    # Keep only necessary columns from weather
    merged_df = merged_df.drop(
        columns=["DATE_TIME_x", "DATE_TIME_y", "HOUR_x", "HOUR_y"], errors="ignore"
    )

    return merged_df


def classify_cloud_conditions(df):
    """Classify cloud cover based on irradiation threshold."""
    # Apply classification logic
    df["CLOUD_CONDITION"] = np.where(
        df["IRRADIATION"] < IRRADIATION_THRESHOLD, "Cloud-Affected", "Clear-Sky"
    )

    # Calculate efficiency ratio
    df["EFFICIENCY_RATIO"] = np.where(
        df["DC_POWER"] > 0, df["AC_POWER"] / df["DC_POWER"], 0
    )

    # Extract date and hour
    df["DATE"] = df["DATETIME"].dt.date
    df["HOUR"] = df["DATETIME"].dt.hour

    return df


def compute_statistics_numpy(df):
    """Compute descriptive statistics using NumPy."""
    # Separate data by cloud condition
    cloud_df = df[df["CLOUD_CONDITION"] == "Cloud-Affected"]
    clear_df = df[df["CLOUD_CONDITION"] == "Clear-Sky"]

    stats_results = {}

    # AC Power statistics
    ac_cloud = cloud_df["AC_POWER"].values
    ac_clear = clear_df["AC_POWER"].values

    stats_results["cloud_ac_mean"] = np.mean(ac_cloud)
    stats_results["cloud_ac_median"] = np.median(ac_cloud)
    stats_results["cloud_ac_std"] = np.std(ac_cloud)
    stats_results["cloud_ac_var"] = np.var(ac_cloud)

    # DC Power statistics
    dc_cloud = cloud_df["DC_POWER"].values
    dc_clear = clear_df["DC_POWER"].values

    stats_results["cloud_dc_mean"] = np.mean(dc_cloud)
    stats_results["clear_dc_mean"] = np.mean(dc_clear)

    # Irradiation statistics
    irr_cloud = cloud_df["IRRADIATION"].values
    irr_clear = clear_df["IRRADIATION"].values

    stats_results["cloud_irr_mean"] = np.mean(irr_cloud)
    stats_results["clear_irr_mean"] = np.mean(irr_clear)

    # Module Temperature
    temp_cloud = cloud_df["MODULE_TEMPERATURE"].values
    temp_clear = clear_df["MODULE_TEMPERATURE"].values

    stats_results["cloud_temp_mean"] = np.mean(temp_cloud)
    stats_results["clear_temp_mean"] = np.mean(temp_clear)

    # Calculate correlation
    correlation = np.corrcoef(df["IRRADIATION"].values, df["AC_POWER"].values)[0, 1]
    stats_results["irradiation_power_correlation"] = correlation

    # Calculate power loss percentage
    power_loss_pct = ((np.mean(ac_clear) - np.mean(ac_cloud)) / np.mean(ac_clear)) * 100
    stats_results["power_loss_percent"] = power_loss_pct

    return stats_results


# [VISUALIZATION FUNCTIONS]


def create_boxplot_ac_power(df):
    """Boxplot comparing AC power under different cloud conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cloud_data = df[df["CLOUD_CONDITION"] == "Cloud-Affected"]["AC_POWER"].values
    clear_data = df[df["CLOUD_CONDITION"] == "Clear-Sky"]["AC_POWER"].values

    bp = ax.boxplot(
        [cloud_data, clear_data],
        labels=["Cloud-Affected", "Clear-Sky"],
        patch_artist=True,
        boxprops=dict(linewidth=2),
        medianprops=dict(linewidth=2, color="red"),
    )

    bp["boxes"][0].set_facecolor("#FF9999")
    bp["boxes"][1].set_facecolor("#99FF99")

    ax.set_title(
        "AC Power Distribution Under Different Cloud Conditions",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("AC Power (kW)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add statistics annotation
    ax.text(
        0.02,
        0.98,
        f"Cloud Mean: {np.mean(cloud_data):.0f} kW\n"
        f"Clear Mean: {np.mean(clear_data):.0f} kW\n"
        f"Power Reduction: {((np.mean(clear_data) - np.mean(cloud_data))/np.mean(clear_data)*100):.1f}%",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure1_Boxplot_AC_Power.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def create_scatter_plot(df):
    """Scatter plot of Irradiation vs AC Power."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Cloud-Affected": "#FF6B6B", "Clear-Sky": "#4ECDC4"}

    for condition in ["Cloud-Affected", "Clear-Sky"]:
        subset = df[df["CLOUD_CONDITION"] == condition]
        ax.scatter(
            subset["IRRADIATION"],
            subset["AC_POWER"],
            alpha=0.5,
            s=20,
            label=condition,
            color=colors[condition],
        )

    # Add threshold line
    ax.axvline(
        x=IRRADIATION_THRESHOLD,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"Cloud Threshold ({IRRADIATION_THRESHOLD} kW/m²)",
    )

    ax.set_xlabel("Irradiation (kW/m²)", fontsize=12)
    ax.set_ylabel("AC Power (kW)", fontsize=12)
    ax.set_title(
        "Solar Power Generation vs Solar Irradiation",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure2_Scatter_Irradiation_vs_Power.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def create_correlation_heatmap(df):
    """Correlation heatmap of numerical variables."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Select numerical columns
    corr_cols = [
        "IRRADIATION",
        "AC_POWER",
        "DC_POWER",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "DAILY_YIELD",
    ]
    corr_data = df[corr_cols].copy()

    corr_matrix = corr_data.corr()

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 10},
    )

    ax.set_title(
        "Correlation Matrix of Solar Generation Parameters",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure3_Correlation_Heatmap.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def create_hourly_profile(df):
    """Hourly profile comparison between cloud and clear conditions."""
    fig, ax = plt.subplots(figsize=(12, 6))

    hourly_stats = df.groupby(["HOUR", "CLOUD_CONDITION"])["AC_POWER"].mean().unstack()

    hourly_stats.plot(
        kind="bar",
        ax=ax,
        color=["#FF6B6B", "#4ECDC4"],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Hour of Day (24h format)", fontsize=12)
    ax.set_ylabel("Average AC Power (kW)", fontsize=12)
    ax.set_title(
        "Average Power Output by Hour - Cloud vs Clear Conditions",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Condition", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticklabels([f"{h:02d}:00" for h in hourly_stats.index], rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "Figure4_Hourly_Profile.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_feature_importance_plot(feature_cols, importances, output_dir):
    """Feature Importance Bar Chart for Random Forest Classifier."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by importance
    sorted_idx = np.argsort(importances)
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    # Create horizontal bar chart
    bars = ax.barh(
        pos,
        importances[sorted_idx],
        align="center",
        color="steelblue",
        edgecolor="black",
    )

    # Add value labels on bars
    for i, (idx, bar) in enumerate(zip(sorted_idx, bars)):
        width = importances[idx]
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=10,
        )

    ax.set_yticks(pos)
    ax.set_yticklabels(np.array(feature_cols)[sorted_idx])
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.set_title(
        "Feature Importance - Random Forest Classifier\n(Impact of Each Variable on Cloud Prediction)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure5_Feature_Importance.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  [/] Figure5_Feature_Importance.png")


def create_animated_time_series(df):
    """Animated time series of power and irradiation over time."""
    df_sorted = df.sort_values("DATETIME")
    timestamps = df_sorted["DATETIME"].unique()
    timestamps = np.sort(timestamps)

    # Limit frames for better performance
    if len(timestamps) > 200:
        timestamps = timestamps[::4]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Initialize lines
    (line1,) = ax1.plot([], [], "b-", linewidth=1.5, label="AC Power")
    (line2,) = ax1.plot([], [], "r-", linewidth=1.5, label="Irradiation (x100)")
    (line3,) = ax2.plot([], [], "g-", linewidth=1.5, label="Module Temperature")

    # Threshold line
    ax1.axhline(
        y=IRRADIATION_THRESHOLD * 100,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Cloud Threshold",
    )

    # Set limits
    all_ac = df_sorted["AC_POWER"].values
    all_irr = df_sorted["IRRADIATION"].values * 100
    all_temp = df_sorted["MODULE_TEMPERATURE"].values

    ax1.set_ylim(0, max(np.max(all_ac), np.max(all_irr)) * 1.1)
    ax1.set_xlim(timestamps[0], timestamps[-1])
    ax2.set_ylim(min(all_temp) - 5, max(all_temp) + 5)

    ax1.set_ylabel("AC Power (kW) / Irradiation (W/m²)", fontsize=12)
    ax2.set_ylabel("Module Temperature (°C)", fontsize=12)
    ax2.set_xlabel("Date/Time", fontsize=12)
    ax1.set_title(
        "Animation 1: Time Series Evolution of Solar Power Generation",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3

    def update(frame):
        current_data = df_sorted[df_sorted["DATETIME"] <= timestamps[frame]]
        if len(current_data) > 0:
            line1.set_data(current_data["DATETIME"], current_data["AC_POWER"])
            line2.set_data(current_data["DATETIME"], current_data["IRRADIATION"] * 100)
            line3.set_data(current_data["DATETIME"], current_data["MODULE_TEMPERATURE"])
        return line1, line2, line3

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(timestamps),
        init_func=init,
        interval=100,
        repeat=False,
        blit=True,
    )

    anim.save(output_dir / "Animation1_Time_Series.gif", writer=PillowWriter(fps=20))

    # Save freeze-frames for the paper (3 frames)
    freeze_timestamps = [
        timestamps[0],
        timestamps[len(timestamps) // 2],
        timestamps[-1],
    ]
    for i, ts in enumerate(freeze_timestamps):
        fig_frame, (ax1_frame, ax2_frame) = plt.subplots(
            2, 1, figsize=(14, 10), sharex=True
        )

        # Plot up to the freeze timestamp
        current_data = df_sorted[df_sorted["DATETIME"] <= ts]

        ax1_frame.plot(
            current_data["DATETIME"],
            current_data["AC_POWER"],
            "b-",
            linewidth=1.5,
            label="AC Power",
        )
        ax1_frame.plot(
            current_data["DATETIME"],
            current_data["IRRADIATION"] * 100,
            "r-",
            linewidth=1.5,
            label="Irradiation (x100)",
        )
        ax1_frame.axhline(
            y=IRRADIATION_THRESHOLD * 100,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Cloud Threshold",
        )
        ax2_frame.plot(
            current_data["DATETIME"],
            current_data["MODULE_TEMPERATURE"],
            "g-",
            linewidth=1.5,
            label="Module Temperature",
        )

        ax1_frame.set_ylabel("AC Power (kW) / Irradiation (W/m²)")
        ax2_frame.set_ylabel("Module Temperature (°C)")
        ax2_frame.set_xlabel("Date/Time")
        ax1_frame.set_title(
            f"Animation Freeze-Frame {i+1}: Data up to {pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M')}"
        )
        ax1_frame.legend(loc="upper left")
        ax2_frame.legend(loc="upper left")
        ax1_frame.grid(True, alpha=0.3)
        ax2_frame.grid(True, alpha=0.3)

        ax1_frame.set_xlim(timestamps[0], timestamps[-1])

        plt.tight_layout()
        plt.savefig(
            output_dir / f"Animation1_FreezeFrame_{i+1}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    plt.close()


def create_animated_daily_bars(df):
    """Animated bar chart showing daily power generation."""
    # Calculate daily aggregates
    df["DATE"] = pd.to_datetime(df["DATETIME"]).dt.date
    daily_data = (
        df.groupby("DATE")
        .agg(
            {
                "AC_POWER": "mean",
                "IRRADIATION": "mean",
                "CLOUD_CONDITION": lambda x: (x == "Cloud-Affected").sum()
                / len(x)
                * 100,
            }
        )
        .reset_index()
    )
    daily_data.columns = ["DATE", "AVG_AC_POWER", "AVG_IRRADIATION", "CLOUD_PERCENT"]
    daily_data = daily_data.sort_values("DATE")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Initialize bars
    bars = ax1.bar(
        range(len(daily_data)),
        [0] * len(daily_data),
        color="steelblue",
        alpha=0.6,
        label="Daily Avg AC Power",
    )
    (line,) = ax2.plot([], [], "red", linewidth=2, label="Cloud Cover (%)")

    ax1.set_ylabel("Average AC Power (kW)", fontsize=12)
    ax2.set_ylabel("Cloud Cover Percentage (%)", fontsize=12)
    ax2.set_xlabel("Day Index", fontsize=12)
    ax1.set_title(
        "Animation 2: Daily Power Generation vs Cloud Cover Percentage",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    ax1.set_ylim(0, daily_data["AVG_AC_POWER"].max() * 1.2)
    ax2.set_ylim(0, 105)
    ax1.set_xlim(0, len(daily_data))
    ax2.set_xlim(0, len(daily_data))
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    def init():
        for bar in bars:
            bar.set_height(0)
        line.set_data([], [])
        return bars, line

    def update(frame):
        for i, bar in enumerate(bars):
            if i <= frame:
                bar.set_height(daily_data.iloc[i]["AVG_AC_POWER"])
        x_vals = list(range(frame + 1))
        line.set_data(x_vals, daily_data.iloc[: frame + 1]["CLOUD_PERCENT"].values)
        return bars, line

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(daily_data),
        init_func=init,
        interval=200,
        repeat=False,
        blit=False,
    )

    anim.save(output_dir / "Animation2_Daily_Bars.gif", writer=PillowWriter(fps=10))

    # Save freeze-frames for the paper (3 frames)
    freeze_indices = [0, len(daily_data) // 2, len(daily_data) - 1]
    for i, idx in enumerate(freeze_indices):
        fig_frame, (ax1_frame, ax2_frame) = plt.subplots(2, 1, figsize=(14, 10))

        # Show bars up to freeze index
        x_pos = range(idx + 1)
        heights = daily_data.iloc[: idx + 1]["AVG_AC_POWER"].values

        ax1_frame.bar(x_pos, heights, color="steelblue", alpha=0.6)
        ax2_frame.plot(
            range(idx + 1),
            daily_data.iloc[: idx + 1]["CLOUD_PERCENT"].values,
            "red",
            linewidth=2,
        )

        ax1_frame.set_ylabel("Average AC Power (kW)")
        ax2_frame.set_ylabel("Cloud Cover Percentage (%)")
        ax2_frame.set_xlabel("Day Index")
        ax1_frame.set_title(f"Animation Freeze-Frame {i+1}: First {idx+1} days")
        ax1_frame.grid(True, alpha=0.3)
        ax2_frame.grid(True, alpha=0.3)
        ax1_frame.set_ylim(0, daily_data["AVG_AC_POWER"].max() * 1.2)
        ax2_frame.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"Animation2_FreezeFrame_{i+1}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    plt.close()


# [SEC 2: OOP WRAPPER CLASS]


class CloudCoverPipeline:
    """
    This class manages the entire data pipeline while applying Object-Oriented Programming (OOP)
    principles discussed in Laboratory 8.
    """

    def __init__(self, generation_path, weather_path):
        """
        Constructor - Initializes the pipeline with file paths.

        Args:
            generation_path (Path): Path to generation CSV file
            weather_path (Path): Path to weather CSV file
        """
        # Encapsulated attributes (private by convention using _)
        self._generation_path = generation_path
        self._weather_path = weather_path
        self._df_generation = None
        self._df_weather = None
        self._merged_df = None
        self._stats = None
        self._output_dir = output_dir
        self._backup_dir = backup_dir

    # Data Ingestion

    def load_data(self):
        """Stage 1: Load CSV files into dataframes."""
        self._df_generation = load_csv_file(self._generation_path)
        self._df_weather = load_csv_file(self._weather_path)

        if self._df_generation is None or self._df_weather is None:
            raise FileNotFoundError("Could not load data files. Check file paths.")

        return self  # Enable method chaining

    # Filtering

    def filter_data(self):
        """Stage 2: Apply unique filter logic (Plant 1, May 15-31, 08:00-17:00)."""
        self._df_weather, self._df_generation = filter_by_unique_logic(
            self._df_weather, self._df_generation
        )
        return self

    # Data Cleaning

    def clean_data(self):
        """Stage 3: Handle missing values, duplicates, and data types."""
        self._df_weather = clean_data(self._df_weather)
        self._df_generation = clean_data(self._df_generation)
        return self

    # Merging

    def merge_data(self):
        """Stage 4: Merge weather and generation datasets."""
        self._merged_df = merge_datasets(self._df_weather, self._df_generation)
        return self

    # Classification

    def classify_clouds(self):
        """Stage 5: Classify cloud conditions based on irradiation threshold."""
        self._merged_df = classify_cloud_conditions(self._merged_df)
        return self

    # Statistical Analysis

    def compute_statistics(self):
        """Stage 6: Compute descriptive statistics using NumPy."""
        self._stats = compute_statistics_numpy(self._merged_df)
        return self

    # Backup and Save

    def backup_and_save(self):
        """Stage 7: Create backup and save cleaned data."""
        create_backup(self._merged_df, "cleaned_data.csv")
        self._merged_df.to_csv(cleaned_file, index=False)
        return self

    # Visualization

    def visualize(self):
        """Stage 8: Create all visualizations (static and animated)."""
        create_boxplot_ac_power(self._merged_df)
        create_scatter_plot(self._merged_df)
        create_correlation_heatmap(self._merged_df)
        create_hourly_profile(self._merged_df)
        create_animated_time_series(self._merged_df)
        create_animated_daily_bars(self._merged_df)
        return self

    # Getter Methods (Encapsulation)

    def get_merged_data(self):
        """Accessor method for merged dataframe."""
        return self._merged_df

    def get_statistics(self):
        """Accessor method for computed statistics."""
        return self._stats

    def get_generation_data(self):
        """Accessor method for generation dataframe."""
        return self._df_generation

    def get_weather_data(self):
        """Accessor method for weather dataframe."""
        return self._df_weather

    # Run Full Pipeline (Method Chaining)

    def run(self):
        """
        Execute the complete pipeline using method chaining.
        This demonstrates the pipeline pattern from the sample project.
        """
        return (
            self.load_data()
            .filter_data()
            .clean_data()
            .merge_data()
            .classify_clouds()
            .compute_statistics()
            .backup_and_save()
            .visualize()
        )


# [SEC 3: MAIN EXECUTION]

if __name__ == "__main__":
    print("\n" + "=" * 54)
    print("  REN-03: CLOUD COVER INTERFERENCE ANALYSIS")
    print("  Solar Power Generation Data Analytics Pipeline")
    print("=" * 54 + "\n")

    # Load data with cleaner output
    print("  Loading data...")
    df_gen = pd.read_csv(generation_file)
    df_wth = pd.read_csv(weather_file)
    print(f"  [/] Plant_1_Generation_Data.csv    ({len(df_gen):,} rows)")
    print(f"  [/] Plant_1_Weather_Sensor_Data.csv ({len(df_wth):,} rows)\n")

    # Apply filters
    print("  Applying filters...")
    df_wth_filtered = df_wth[df_wth["PLANT_ID"] == PLANT_ID].copy()
    df_gen_filtered = df_gen[df_gen["PLANT_ID"] == PLANT_ID].copy()

    df_wth_filtered["DATETIME"] = pd.to_datetime(df_wth_filtered["DATE_TIME"])
    df_gen_filtered["DATETIME"] = pd.to_datetime(
        df_gen_filtered["DATE_TIME"], format="%d-%m-%Y %H:%M"
    )

    df_wth_filtered = df_wth_filtered[
        (df_wth_filtered["DATETIME"] >= START_DATE)
        & (df_wth_filtered["DATETIME"] <= END_DATE)
    ].copy()
    df_gen_filtered = df_gen_filtered[
        (df_gen_filtered["DATETIME"] >= START_DATE)
        & (df_gen_filtered["DATETIME"] <= END_DATE)
    ].copy()

    df_wth_filtered["HOUR"] = df_wth_filtered["DATETIME"].dt.hour
    df_gen_filtered["HOUR"] = df_gen_filtered["DATETIME"].dt.hour

    df_wth_filtered = df_wth_filtered[
        (df_wth_filtered["HOUR"] >= START_HOUR) & (df_wth_filtered["HOUR"] <= END_HOUR)
    ].copy()
    df_gen_filtered = df_gen_filtered[
        (df_gen_filtered["HOUR"] >= START_HOUR) & (df_gen_filtered["HOUR"] <= END_HOUR)
    ].copy()

    print(
        f"  [/] Plant: {PLANT_ID} | Date: {START_DATE} to {END_DATE} | Time: {START_HOUR:02d}:00–{END_HOUR:02d}:00\n"
    )

    # Clean data
    print("  Cleaning & merging...")
    df_wth_filtered = df_wth_filtered.drop_duplicates()
    df_gen_filtered = df_gen_filtered.drop_duplicates()

    # Merge
    df_gen_agg = (
        df_gen_filtered.groupby(["DATETIME", "PLANT_ID"])
        .agg(
            {
                "DC_POWER": "sum",
                "AC_POWER": "sum",
                "DAILY_YIELD": "mean",
                "TOTAL_YIELD": "mean",
            }
        )
        .reset_index()
    )

    merged_df = pd.merge(
        df_gen_agg,
        df_wth_filtered,
        on=["DATETIME", "PLANT_ID"],
        how="inner",
    )

    merged_df = merged_df.drop(
        columns=["DATE_TIME_x", "DATE_TIME_y", "HOUR_x", "HOUR_y"], errors="ignore"
    )

    print(f"  [/] Final dataset: {len(merged_df)} rows\n")

    # Classify cloud conditions
    print("  Classifying cloud conditions...")
    merged_df["CLOUD_CONDITION"] = np.where(
        merged_df["IRRADIATION"] < IRRADIATION_THRESHOLD, "Cloud-Affected", "Clear-Sky"
    )

    cloud_count = len(merged_df[merged_df["CLOUD_CONDITION"] == "Cloud-Affected"])
    clear_count = len(merged_df[merged_df["CLOUD_CONDITION"] == "Clear-Sky"])
    print(
        f"  [/] Cloud-Affected: {cloud_count} ({cloud_count/len(merged_df)*100:.1f}%)  |  Clear-Sky: {clear_count} ({clear_count/len(merged_df)*100:.1f}%)\n"
    )

    # Compute statistics
    print("─" * 54)
    print("  RESULTS")
    print("─" * 54)

    cloud_df = merged_df[merged_df["CLOUD_CONDITION"] == "Cloud-Affected"]
    clear_df = merged_df[merged_df["CLOUD_CONDITION"] == "Clear-Sky"]

    power_loss = (
        (np.mean(clear_df["AC_POWER"]) - np.mean(cloud_df["AC_POWER"]))
        / np.mean(clear_df["AC_POWER"])
    ) * 100
    correlation = np.corrcoef(
        merged_df["IRRADIATION"].values, merged_df["AC_POWER"].values
    )[0, 1]
    temp_diff = np.mean(clear_df["MODULE_TEMPERATURE"]) - np.mean(
        cloud_df["MODULE_TEMPERATURE"]
    )

    print(f"  Power Loss (clouds):        {power_loss:6.1f}%")
    print(f"  Irradiation vs Power (r):   {correlation:8.4f}")
    print(f"  Avg AC Power — Cloudy:    {np.mean(cloud_df['AC_POWER']):8,.0f} kW")
    print(f"  Avg AC Power — Clear:    {np.mean(clear_df['AC_POWER']):8,.0f} kW")
    print(f"  Temp difference:            +{temp_diff:5.1f}°C in clear sky\n")

    # [SKEWNESS CALCULATION]

    print("─" * 54)
    print("  DISTRIBUTION ANALYSIS - SKEWNESS")
    print("─" * 54)

    # Function to calculate skewness manually (no scipy needed)
    def calculate_skewness(data):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        skewness = np.sum((data - mean) ** 3) / (n * std**3)
        return skewness

    # Calculate skewness for AC Power
    clear_ac_skew = calculate_skewness(clear_df["AC_POWER"].values)
    cloud_ac_skew = calculate_skewness(cloud_df["AC_POWER"].values)

    # Calculate skewness for Irradiation
    clear_irr_skew = calculate_skewness(clear_df["IRRADIATION"].values)
    cloud_irr_skew = calculate_skewness(cloud_df["IRRADIATION"].values)

    # Calculate skewness for Module Temperature
    clear_temp_skew = calculate_skewness(clear_df["MODULE_TEMPERATURE"].values)
    cloud_temp_skew = calculate_skewness(cloud_df["MODULE_TEMPERATURE"].values)

    print(f"\n  AC Power Skewness:")
    print(f"    Clear-Sky: {clear_ac_skew:.4f}")
    print(f"    Cloud-Affected: {cloud_ac_skew:.4f}")

    print(f"\n  Irradiation Skewness:")
    print(f"    Clear-Sky: {clear_irr_skew:.4f}")
    print(f"    Cloud-Affected: {cloud_irr_skew:.4f}")

    print(f"\n  Module Temperature Skewness:")
    print(f"    Clear-Sky: {clear_temp_skew:.4f}")
    print(f"    Cloud-Affected: {cloud_temp_skew:.4f}")

    # Interpretation
    print(f"\n  Interpretation:")
    if clear_ac_skew > 0.5:
        print(f"    Clear-Sky AC Power is positively skewed ({clear_ac_skew:.4f}) -")
        print(
            "    indicating a tail toward higher power values (peak generation events)."
        )
    elif clear_ac_skew < -0.5:
        print(f"    Clear-Sky AC Power is negatively skewed ({clear_ac_skew:.4f})")
    else:
        print(
            f"    Clear-Sky AC Power is approximately symmetric ({clear_ac_skew:.4f})"
        )

    if cloud_ac_skew > 0.5:
        print(f"    Cloud-Affected AC Power is positively skewed ({cloud_ac_skew:.4f})")
    elif cloud_ac_skew < -0.5:
        print(f"    Cloud-Affected AC Power is negatively skewed ({cloud_ac_skew:.4f})")
    else:
        print(
            f"    Cloud-Affected AC Power is approximately symmetric ({cloud_ac_skew:.4f})"
        )

    # Outlier detection
    mean_ac = np.mean(clear_df["AC_POWER"])
    std_ac = np.std(clear_df["AC_POWER"])
    outliers = clear_df["AC_POWER"][
        (clear_df["AC_POWER"] > mean_ac + 2 * std_ac)
        | (clear_df["AC_POWER"] < mean_ac - 2 * std_ac)
    ]
    print(f"\n  Outlier Detection (Clear-Sky AC Power):")
    print(
        f"    Outliers detected: {len(outliers)} ({len(outliers)/len(clear_df)*100:.1f}% of data)"
    )
    print()

    # [MACHINE LEARNING MODEL]

    print("─" * 54)
    print("  MACHINE LEARNING MODEL")
    print("─" * 54)

    # Auto-install scikit-learn if missing
    try:
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        print(f"  [/] scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("  [!] scikit-learn not found. Installing...")
        import subprocess
        import sys

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "scikit-learn", "-q"]
        )
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )
        import sklearn

        print(f"  [/] scikit-learn installed. Version: {sklearn.__version__}")

    # MOD 1: BASELINE (Threshold Rule)
    print("\n[MOD 1] IRRADIATION Threshold")

    y_true = (merged_df["IRRADIATION"] < IRRADIATION_THRESHOLD).astype(int)
    cloud_count_baseline = sum(y_true == 1)
    clear_count_baseline = sum(y_true == 0)
    total_baseline = len(y_true)

    print(f"    Rule: IRRADIATION < {IRRADIATION_THRESHOLD} kW/m² = Cloud-Affected")
    print(
        f"    Cloud-Affected: {cloud_count_baseline} ({cloud_count_baseline/total_baseline*100:.1f}%)"
    )
    print(
        f"    Clear-Sky: {clear_count_baseline} ({clear_count_baseline/total_baseline*100:.1f}%)"
    )
    print(f"    Accuracy: 100.00% (by definition - proves data is clean)")

    # MOD 2: RFC (Random Forest Classifier)
    print("\n[MOD 2] Random Forest")

    # Features - NO IRRADIATION (avoids data leakage)
    feature_cols = ["MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "DC_POWER"]
    X = merged_df[feature_cols]
    y = (merged_df["IRRADIATION"] < IRRADIATION_THRESHOLD).astype(int)

    print(f"    Features used: {feature_cols}")
    print(f"    Total samples: {len(X)}")

    # Split data into training (70%) and testing (30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"    Training samples: {len(X_train)}")
    print(f"    Testing samples: {len(X_test)}")

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"    True Negatives (Clear → Clear):  {cm[0,0]}")
    print(f"    False Positives (Clear → Cloud): {cm[0,1]}")
    print(f"    False Negatives (Cloud → Clear): {cm[1,0]}")
    print(f"    True Positives (Cloud → Cloud):  {cm[1,1]}")

    # Feature Importance
    print(f"\nFeature Importance:")
    for col, imp in zip(feature_cols, model.feature_importances_):
        print(f"      {col}: {imp:.4f}")

    # Baseline comparison (always predicting majority class)
    majority_class = 0 if sum(y == 0) > sum(y == 1) else 1
    baseline_majority = sum(y == majority_class) / len(y)
    print()

    # Create visualizations
    print("─" * 54)
    print("  OUTPUTS")
    print("─" * 54)

    create_boxplot_ac_power(merged_df)
    print(f"  [/] Figure1_Boxplot_AC_Power.png")

    create_scatter_plot(merged_df)
    print(f"  [/] Figure2_Scatter_Irradiation_vs_Power.png")

    create_correlation_heatmap(merged_df)
    print(f"  [/] Figure3_Correlation_Heatmap.png")

    create_hourly_profile(merged_df)
    print(f"  [/] Figure4_Hourly_Profile.png")

    create_feature_importance_plot(feature_cols, model.feature_importances_, output_dir)

    create_animated_time_series(merged_df)
    print(f"  [/] Animation1_Time_Series.gif")

    create_animated_daily_bars(merged_df)
    print(f"  [/] Animation2_Daily_Bars.gif\n")

    # Save cleaned data
    merged_df.to_csv(cleaned_file, index=False)

    print("=" * 54)
    print(f"  Pipeline completed. Outputs saved to: /outputs")
    print("=" * 54 + "\n")
