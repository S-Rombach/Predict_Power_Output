"""
This module contains configuration constants for the project.
It defines paths for data directories, model directories, and other project-related constants.
"""
import os
from pathlib import Path


# Use 'pyproject.toml' file in the project root as indicator for the project root.
# It is also possible to hard code this variable to the project path.
BASE_DIR = next(
    (
        str(p)
        for p in [Path.cwd(), *Path.cwd().parents]
        if (p / "pyproject.toml").exists()
    ),
    "",
)
""" Base directory of the project, used to construct paths relative to this script. """

if BASE_DIR is None or BASE_DIR == "":
    raise FileNotFoundError(
        "Could not find 'pyproject.toml' as indicator of the project root."
        " Initialization of directory constants is not possible."
        " Add a file with name 'pyproject.toml' to the project root or change the"
        " evaluation mechanism in 'src/config.py'."
    )


DATA_DIR = os.path.join(BASE_DIR, "data")
""" Directory for all data files, including raw and processed data. """
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
""" Directory for raw data files, typically the orig after some form of preprocessing. """
DATA_ORIG_DIR = os.path.join(DATA_DIR, "orig")
""" Directory for original data files, typically downloaded from Kaggle. """
DATA_SKRIPTS_DIR = os.path.join(DATA_DIR, "scripts")
""" Directory for scripts related to data processing and transformation. """
DATA_SPLIT_DIR = os.path.join(DATA_DIR, "split")
""" Directory for split data files, such as training, test and validation sets. """

DATA_ORIG_FILENAME = ".csv"
""" Name of the original data file downloaded from power installation. """
DATA_RAW_FILENAME = "power_data.csv"
""" Name of the raw data file after transformation. """
TRAIN_RAW_FILENAME = "train_raw_split.csv"
""" Name of the training data file after splitting. """
TEST_RAW_FILENAME = "test_raw_split.csv"
""" Name of the test data file after splitting. """
VALIDATION_RAW_FILENAME = "validation_raw_split.csv"
""" Name of the validation data file after splitting. """


MODELS_DIR = os.path.join(BASE_DIR, "models")
""" Directory for machine learning models and related files. """

STUDY_DIR = os.path.join(BASE_DIR, "studies")
""" Directory for optuna studies and related files. """

MODEL_ALIASES = {
    "HistGradientBoostingClassifier": "hgb",
    "RandomForestClassifier": "rf",
    "Sequential": "nn",
    "GradientBoostingClassifier": "gb",
    "ExtraTreesClassifier": "et",
    "BaggingClassifier": "bag",
    "DecisionTreeClassifier": "dt",
    "AdaBoostClassifier": "ada",
    "RidgeClassifier": "ridge",
    "KNeighborsClassifier": "knn",
    "LogisticRegression": "logreg",
    "MLPClassifier": "mlp",
    "GaussianNB": "gnb",
}
""" Short aliases for common classifiers. """