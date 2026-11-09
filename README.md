# Titanic Survival Prediction (Python + R with Docker)

This repository predicts which passengers survived the Titanic disaster using a **logistic regression** model. The same workflow is implemented twice, once in **Python** and once in **R**, and each runs inside its own **Docker** container, so you do not need Python or R installed on your computer.

Each script:
1. Loads `train.csv` and prints basic exploration (missing values, survival rates)
2. Cleans the data (fills missing Age, Fare, Embarked; encodes Sex and Embarked)
3. Adds new features (`FamilySize`, `IsAlone`)
4. Trains a logistic regression model and prints the training accuracy
5. Loads `test.csv`, predicts survival, and prints the test accuracy
6. Saves the predictions to `src/data/`

Every step prints its output to the terminal.

---

## Repository structure

```
titanic-disaster/
├── .github/
│   └── CODEOWNERS
├── src/
│   ├── data/                  <- put the Kaggle CSV files here (not tracked by git)
│   ├── python/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── titanic.py
│   └── r/
│       ├── Dockerfile
│       ├── install_packages.R
│       └── titanic.R
├── .gitignore
└── README.md
```

---

## Prerequisites

You only need two things installed:

1. **Git**: https://git-scm.com/downloads
2. **Docker Desktop**: https://www.docker.com/products/docker-desktop/
   - After installing, **open Docker Desktop** and wait until it shows **"Engine running"**.
   - Check it works by running `docker --version` in a terminal.

You also need a free **Kaggle** account to download the data: https://www.kaggle.com

---

## Step 1: Clone the repository

Open a terminal (macOS/Linux: Terminal; Windows: PowerShell) and run:

```bash
git clone https://github.com/Hikari7101/titanic-disaster.git
cd titanic-disaster
```

**All commands below must be run from this `titanic-disaster` folder.**

---

## Step 2: Download the data

The dataset is **not included** in this repository.

1. Go to https://www.kaggle.com/competitions/titanic/data and log in.
2. Click **"Download All"** (accept the competition rules if asked).
3. Unzip the downloaded `titanic.zip`.
4. Create the data folder (if it does not exist) and move the three CSV files into it:

```bash
mkdir -p src/data          # macOS/Linux
mkdir src\data             # Windows (skip if the folder already exists)
```

Your folder should look like this:

```
src/data/
├── gender_submission.csv
├── test.csv
└── train.csv
```

> **Optional (Kaggle CLI):** if you have the Kaggle API set up, you can instead run
> `kaggle competitions download -c titanic -p src/data` and then unzip the file inside `src/data`.

---

## Step 3: Run the Python version

**Build the image** (first time takes 1–3 minutes):

```bash
docker build -t titanic-python src/python
```

**Run the container.** The `-v` option shares your local `src/data` folder with the container so it can read the CSV files. Use the command for your system:

| System | Command |
|---|---|
| macOS / Linux | `docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python` |
| Windows PowerShell | `docker run --rm -v "${PWD}/src/data:/app/data" titanic-python` |
| Windows Command Prompt | `docker run --rm -v "%cd%\src\data:/app/data" titanic-python` |

---

## Step 4: Run the R version

**Build the image** (first time takes several minutes because the R base image is large):

```bash
docker build -t titanic-r src/r
```

**Run the container:**

| System | Command |
|---|---|
| macOS / Linux | `docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r` |
| Windows PowerShell | `docker run --rm -v "${PWD}/src/data:/app/data" titanic-r` |
| Windows Command Prompt | `docker run --rm -v "%cd%\src\data:/app/data" titanic-r` |

---

## Expected output

Both scripts print each step to the terminal. The end of the output looks like this:

```
[5] Accuracy on training set: 0.8013
...
Predicted survivors in test set: 158 / 418
Saved predictions to /app/data/predictions_python.csv

[7] test.csv has no true labels, so predictions are compared with gender_submission.csv
Accuracy on test set (vs gender_submission.csv): 0.9474

Done!
```

After running, the predictions are saved on your computer as:
- `src/data/predictions_python.csv`
- `src/data/predictions_r.csv`

---

## Notes on the model

- **Features:** Pclass, Sex, Age, SibSp, Parch, Fare, FamilySize, IsAlone, Embarked (one-hot encoded).
- **Missing values:** Age and Fare are filled with the training-set median; Embarked is filled with the most common port (`S`). Cabin is dropped because most values are missing.
- **Test accuracy:** Kaggle's `test.csv` does **not** contain the `Survived` column. To still report a test accuracy, predictions are compared against `gender_submission.csv`, Kaggle's sample answer file (which predicts all women survive and all men do not). This number shows agreement with that baseline, not true accuracy.
- **Package versions:** Python packages are pinned in `src/python/requirements.txt`. The R image (`rocker/r-ver:4.4.1`) installs packages from a fixed-date CRAN snapshot, so versions are also fixed.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` / `docker is not recognized` | Make sure Docker Desktop is installed and running, then close and reopen your terminal. |
| `Cannot connect to the Docker daemon` / `error during connect` | Open Docker Desktop and wait for "Engine running". |
| `FileNotFoundError` or `cannot open file '/app/data/train.csv'` | The CSV files are not in `src/data/`, or you are not running the command from the `titanic-disaster` folder. |