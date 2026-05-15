# FINAL PROJECT: Engineering Data Systems Pipeline

> **Topic ID: REN-03: Cloud Cover Interference**  

> **Research Title: Automated Statistical Detection of Cloud Cover Interference in Solar Power Generation Using Python Data Pipelines**  

> Technological University of the Philippines – Manila | Electronics Engineering Department

---

## Branches

| Branch | Description |
|---|---|
| [`main`](../../tree/main) | Full pipeline — complete source code, datasets, and all outputs |
| [`filter-logic`](../../tree/filter-logic) | Unique filter implementation — Plant ID, date range, and daytime hour filtering |
| [`results`](../../tree/results) | Statistical outputs, visualizations, animations, and cleaned dataset |

> ⚠️ **Note:** The `filter-logic` and `results` branches are **not separate implementations**. They are visual walkthroughs created to demonstrate specific stages of the main pipeline — showing how the filter logic is applied and how the results are produced. The actual working code lives entirely in the `main` branch.

---

## Overview

This project is an automated engineering data analytics pipeline designed to study how cloud cover interference affects photovoltaic (PV) solar power generation systems. Using Python, the system collects and processes data, cleans and organizes the information, performs statistical analysis, creates visualizations, and applies machine learning techniques to classify cloud conditions.

The pipeline was developed using Python, NumPy, Pandas, Matplotlib, Seaborn, and Object-Oriented Programming (OOP) principles. It automates the processing of solar telemetry and weather sensor datasets, allowing for efficient detection of cloud-affected operating conditions and assessment of their impact on solar energy production.

All codes used in this project are based on the concepts and techniques learned from our laboratory activities, particularly Lab 6: Numerical and Data Analysis with NumPy and Pandas, Lab 7: Data Visualization, and Lab 8: Object-Oriented Programming (OOP). These laboratory activities served as the main foundation in developing the entire pipeline.

The system was tested using 621 telemetry records collected from Plant 4135001 over a 17-day period (May 15–31, 2020) during daytime hours (08:00–17:00). Key findings include a very strong positive correlation (r = 0.9917) between solar irradiation and AC power output, a 72.3% power loss during cloud-affected periods, and a Random Forest classifier achieving 96.79% accuracy in detecting cloud conditions without directly using irradiation measurements.

---

## Unique Filter Logic

| Parameter | Value |
|---|---|
| Plant ID | `4135001` (Plant 1 only) |
| Date Range | May 15 – May 31, 2020 |
| Time Window | 08:00 – 17:00 (daytime hours) |
| Cloud Classification | `IRRADIATION < 0.3 kW/m²` = Cloud-Affected |

---

## Features

### 1. Data Ingestion Module
- Loads raw CSV files for solar generation and weather sensor data
- Applies the unique programmatic filter (Plant ID, date range, and time window)
- Includes `try-except` error handling to prevent pipeline crashes on missing or corrupted files

### 2. Automated Data Cleaning Module
- Detects and removes duplicate records
- Identifies columns with missing/null values
- Fills missing numeric values using **median imputation**
- Corrects data type inconsistencies (datetime parsing)
- Automatically creates timestamped backups before processing

### 3. Data Merging & Classification Module
- Aggregates generation data by `DATETIME` and `PLANT_ID`
- Merges weather sensor data with generation data using an inner join
- Classifies each record as `Clear-Sky` or `Cloud-Affected` based on the irradiation threshold
- Computes an efficiency ratio (`AC_POWER / DC_POWER`) for each timestamp

### 4. Statistical Analytics Engine (NumPy)
- Computes **Mean, Median, Standard Deviation, and Variance** for AC Power, Irradiation, and Module Temperature under both cloud conditions
- Calculates **Pearson Correlation Coefficient** between irradiation and AC power output
- Computes **Power Loss Percentage** between clear-sky and cloud-affected periods
- Performs **Skewness Analysis** using a custom NumPy-based function (no SciPy dependency)
- Detects outliers using the **2σ threshold method**

### 5. Machine Learning Module (Random Forest)
- **Model 1 (Baseline):** Irradiation threshold classifier — achieves 100% accuracy by definition, confirming dataset cleanliness
- **Model 2 (Random Forest Classifier):** Trained on `MODULE_TEMPERATURE`, `AMBIENT_TEMPERATURE`, and `DC_POWER` only (irradiation intentionally excluded to avoid data leakage)
- Reports Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Feature Importance

### 6. Visualization & Animation Suite
- **5 Static Plots:**
  - Boxplot of AC Power Distribution (Cloud-Affected vs. Clear-Sky)
  - Scatter Plot of Solar Power Generation vs. Solar Irradiation
  - Correlation Heatmap of Solar Generation Parameters
  - Hourly Profile of Average Power Output (Cloud vs. Clear)
  - Feature Importance Bar Chart (Random Forest Classifier)
- **2 Animated Sequences (GIF):**
  - Time Series Evolution of Solar Power Generation
  - Daily Power Generation vs. Cloud Cover Percentage

---

## Repository Structure

```
EDS_4764_Laude/
│
├── main.py                         # Main Python data pipeline
├── requirements.txt                # Required Python libraries
├── README.md                       # Project documentation
│
├── data/
│   ├── Plant_1_Generation_Data.csv     # Original generation dataset (Kaggle)
│   ├── Plant_1_Weather_Sensor_Data.csv # Original weather sensor dataset (Kaggle)
│   └── dataset_cleaned.csv             # Cleaned and merged output dataset
│
└── outputs/
    ├── Figure1_Boxplot_AC_Power.png
    ├── Figure2_Scatter_Irradiation_vs_Power.png
    ├── Figure3_Correlation_Heatmap.png
    ├── Figure4_Hourly_Profile.png
    ├── Figure5_Feature_Importance.png
    ├── Animation1_Time_Series.gif
    └── Animation2_Daily_Bars.gif
```

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/EDS_4764_Laude.git
cd EDS_4764_Laude
```

### 2. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Go to [Kaggle – Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) and download:
- `Plant_1_Generation_Data.csv`
- `Plant_1_Weather_Sensor_Data.csv`

Place both files inside the `data/` folder.

### 4. Update the Project Root Path
Open `main.py` and update the `project_root` variable to match your local directory:
```python
project_root = Path("your/local/path/EDS_4764_Laude")
```

### 5. Run the Pipeline
```bash
python main.py
```

All outputs (cleaned dataset, static plots, and animated GIFs) will be automatically saved to the `outputs/` folder.

---

## Built With

<div align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="60" alt="python logo"/>
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="60" alt="numpy logo"/>
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" height="60" alt="pandas logo"/>
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matplotlib/matplotlib-original.svg" height="60" alt="matplotlib logo"/>
</div>

<br/>

| Library | Purpose |
|---|---|
| `numpy` | Vectorized statistical computations |
| `pandas` | Data ingestion, cleaning, and merging |
| `matplotlib` | Static plots and animated GIF sequences |
| `seaborn` | Correlation heatmap and styled visualizations |
| `scikit-learn` | Random Forest classifier, train-test split, metrics |

---

## Key Results

| Metric | Value |
|---|---|
| Total Records Analyzed | 621 |
| Study Period | May 15–31, 2020 (17 days) |
| Irradiation–AC Power Correlation (r) | **0.9917** |
| Mean AC Power — Clear-Sky | **18,359.59 kW** |
| Mean AC Power — Cloud-Affected | **5,082.08 kW** |
| Power Loss Due to Cloud Cover | **72.3%** |
| Random Forest Accuracy | **96.79%** |
| Top Predictive Feature | DC Power (58.26%) |

---

## Citation

If referencing this work in your research:

> Laude, R. R. D. (2026). Automated Statistical Detection of Cloud Cover Interference in Solar Power Generation Using Python Data Pipelines. Computer Programming 1 Final Project, Technological University of the Philippines — Manila.

Full paper: [Laude_4764_IEEE_Paper.pdf](https://drive.google.com/file/d/1l6MEAVftRMKXM0VDhcMYYM0Q8eiFQJsg/view?usp=sharing)

---

## AI Disclosure

### Artificial Intelligence (AI) Usage Disclosure

This project utilized Artificial Intelligence (AI) tools as supplementary assistants during the development, debugging, documentation, formatting, and improvement stages of the study. The following AI systems were used throughout the project:

- **ChatGPT** — https://chatgpt.com
- **Claude** — https://claude.ai

These tools were primarily used for:

- Code debugging and syntax assistance
- Documentation refinement and proofreading
- Suggestions for visualization formatting and project structure

All final decisions, implementations, data processing, analysis, interpretation of results, and conclusions in this project were independently reviewed, validated, and finalized by the author. AI-generated suggestions were used only as assistive references and were not considered substitutes for critical analysis, programming logic, or academic judgment.

The author remains fully responsible for the accuracy, originality, integrity, and overall content of this project.

---

## Acknowledgment

### Human Contribution Disclosure

Portions of this project also benefited from the technical guidance of the author’s brother, a **Senior Web Developer at IBM**, whose professional expertise in software engineering, systems architecture, and full-stack development contributed to the following areas:

- Code structure review and software engineering best practices
- Guidance on modular design, version control workflows, and GitHub repository organization
- Advice on production-level Python practices and pipeline optimization
- Technical feedback on data pipeline architecture and project documentation standards

His industry experience in enterprise-level software development helped improve the engineering discipline and professionalism reflected in this project’s codebase and documentation.

All programming logic, data analysis, statistical interpretation, and written content remain the original work of the author.

---

## 👤 Author

**Ruselle Rae D. Laude** | `TUPM-25-4764`
<br/>
Electronics Engineering Department
<br/>
Technological University of the Philippines, Manila
<br/>
rusellerae.laude@tup.edu.ph