# Supervised Machine Learning Pipeline

This project implements a modular and extensible pipeline for conducting **supervised machine learning experiments** using a variety of machine learning models. It supports flexible data preprocessing, customizable model configurations, and post-training evaluation. Designed with reproducibility and experimentation in mind, the pipeline can be easily extended to support new models, techniques and datasets.

## Table of Contents

1. [Project Structure](#project-structure)
4. [Installation](#installation)
   - [Local Setup](#local-setup)
5. [Configuration and Schema](#configuration-and-schema)
6. [Running the Pipeline](#running-the-pipeline)
7. [Output and Logs](#output-and-logs)

---

## Project Structure

```text
├── main.py                     # Entry point for executing the full pipeline
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment configuration file
├── data/                       # Main folder for storing the data
│   ├── raw/                    # Location for raw input data
│   │   └── data.csv            # Main dataset
│   └── schema/                 # Data schema definitions
│       └── schema.json         # JSON file defining expected dataset column structure and types
├── output/                     # Main folder for storing all results and artifacts
│   └── 20250912_145622/        # Timestamped execution results
│       ├── eval_scores_*.html  # Evaluation metrics charts (HTML)
│       ├── eval_scores_*.png   # Evaluation metrics charts (PNG)
│       ├── eval_times_*.html   # Execution time charts (HTML)
│       ├── eval_times_*.png    # Execution time charts(PNG)
│       ├── eval_summary.csv    # Tabular summary of evaluation metrics
│       └── experiment.log      # Execution log file
└── src/                        # Main source code
    ├── pipeline.py             # Orchestrator class coordinating all training pipeline steps
    ├── config/                 # System configurations
    │   ├── logging.py          # Logging configuration
    │   └── settings.py         # General settings
    ├── core/                   # Core modules
    │   ├── data/               # Data management
    │   │   ├── data_loader.py  # Functions for loading and managing datasets
    │   │   └── schema.py       # Schema definitions and validation
    │   ├── evaluation/         # Model evaluation
    │   │   ├── decorator.py    # Evaluation decorators
    │   │   ├── evaluator.py    # Metric calculation and evaluation
    │   │   └── plots.py        # Chart and visualization generation
    │   ├── models/             # Model definitions and factory
    │   │   ├── definitions.py  # Base model definitions
    │   │   └── factory.py      # Factory for building model instances
    │   └── processors/         # Data processing
    │       ├── presplit.py     # Preprocessing steps before dataset splitting
    │       └── postsplit.py    # Transformations applied after data splitting
```

After running the pipeline, the following will be created:

- **`output/`**:  
  Main folder for storing all results and artifacts, including logs, evaluation metrics, plots, and fold definitions.

## Installation

### Local Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/weslleygere/Supervised-ML.git
   cd Supervised-ML
   ```

2. **Create and activate a virtual environment** (optional but highly recommended):

   ```bash
   python -m venv venv
   ```

   **On Windows**:

   ```powershell
   .\venv\Scripts\activate
   ```

   **On Linux/Mac**:

   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file according to your setup (dataset paths, model selection, etc.).

## Configuration and Schema

### Environment Configuration

The pipeline is configured using environment variables defined in a `.env` file. Key configuration options include:

- **`DATA_PATH`**: Path to your dataset file (e.g., `data/raw/data.csv`)
- **`SCHEMA_PATH`**: Path to the schema definition file (`data/schema/schema.json`)
- **`OUTPUT_DIR`**: Directory for storing results (`output`)
- **`MODELS`**: Which models to evaluate (see [Available Models](#available-models) below)
- **`RANDOM_STATE`**: Seed for reproducible results (`42`)
- **`N_SPLITS`**: Number of cross-validation folds (`5`)
- **`LOG_LEVEL`**: Logging verbosity (`INFO`, `DEBUG`, `WARNING`, `ERROR`)

### Available Models

You can configure which models to run using the `MODELS` environment variable:

**Options:**

- `all` - Run all available models
- `MODEL1,MODEL2,MODEL3` - Run only specified models (comma-separated)
- `all --exclude MODEL1,MODEL2` - Run all models except the specified ones

**Available Models:**

- **Linear Models**: `LINEAR_REGRESSION`, `RIDGE_REGRESSION`, `LASSO_REGRESSION`, `ELASTIC_NET`, `PLS`
- **Tree-Based**: `DECISION_TREE`, `RANDOM_FOREST`, `EXTRA_TREES`
- **Boosting**: `ADABOOST`, `XGBOOST`, `LIGHTGBM`, `CATBOOST`
- **Other**: `K_NEIGHBORS`, `SVR`, `MLP`

### Data Schema

To ensure the pipeline correctly interprets your dataset, you must define a `data/schema/schema.json` file that specifies the target column and maps all feature columns to their semantic types.

Example `schema.json`:

```json
{
  "target": {
    "names": ["target_column"]
  },
  "features": [
    {
      "name": "feature_1",
      "type": "numerical"
    },
    {
      "name": "feature_2",
      "type": "numerical"
    },
    {
      "name": "categorical_feature",
      "type": "categorical"
    },
    {
      "name": "id_column",
      "type": "ignore"
    }
  ]
}
```

### Supported Column Types

| Type         | Used in Training | Transformed       | Notes                                         |
|--------------|------------------|-------------------|-----------------------------------------------|
| `numerical`  | Yes              | Imputed, scaled   | Robust to outliers with default pipeline      |
| `ignore`     | No               | Dropped           | Explicitly excluded from modeling             |

### Example Configuration

**Complete `.env` file example:**

```bash
# Data paths
DATA_PATH=data/raw/data.csv
SCHEMA_PATH=data/schema/schema.json
OUTPUT_DIR=output

# Model configuration
MODELS=all
RANDOM_STATE=42

# Cross-validation
N_SPLITS=5

# Logging
LOG_LEVEL=INFO
DEBUG=false
```

## Running the Pipeline

### Locally

1. Make sure all dependencies are installed and your data is in the `data/raw/` folder.
2. Configure the schema file at `data/schema/schema.json`.
3. Run the main script:

   ```bash
   python main.py
   ```

4. After execution, check the `output/` directory for results and visualizations.

## Output and Logs

- **`output/YYYYMMDD_HHMMSS/`**  
  Main folder for storing all experiment results.  
  Each execution creates a timestamped subfolder.

  Inside each execution folder, you will find:

  - `eval_summary.csv`: Tabular summary of evaluation metrics (mean and standard deviation).
  - `eval_scores_METRIC.png/html`: Bar charts comparing model performance for each metric.
  - `eval_times_times.png/html`: Bar charts comparing training and prediction times across models.
  - `experiment.log`: Execution log file.

- **Other Temporary Outputs**  
  Depending on the models used, additional folders may be generated:
  - `__pycache__/`: Auto-generated Python bytecode; can be ignored or deleted.