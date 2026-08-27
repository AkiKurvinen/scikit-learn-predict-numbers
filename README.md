# Machine learning AI test
- Test machine learning with scikit-learn
- [semeion handwritten digit dataset](https://archive.ics.uci.edu/dataset/178/semeion+handwritten+digit)

## Requirements
- Python 3.12.10

## Installation

**Create and Activate Virtual Environment**

A. *Windows (Git Bash)*
```bash
python -m venv .venv
source .venv/Scripts/activate
```

B. *Windows (cmd or PowerShell)*
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Install Dependencies**
```bash
pip install -r requirements.txt
pip install -U scikit-learn
```

## Usage

### Preview numbers
```
python display_number.py 130
```

### Train model
```
python train_and_save.py
```

### Predict
```
python predict_from_saved.py 130
```
