# CaliForest 🌲

**Cali**brated Random **Forest** - A Professional, Enhanced Implementation of Calibrated Tree Ensembles

This Python package implements state-of-the-art calibrated tree ensemble algorithms with comprehensive improvements for production use. Built upon the foundational work presented in [ACM CHIL 2020](https://www.chilconference.org/), this enhanced version provides enterprise-grade reliability, documentation, and functionality.

> **Note**: This is a professionally enhanced version of the original CaliForest repository by Yubin Park, featuring extensive improvements in code quality, documentation, error handling, and advanced functionality including flexible parameter support and modern logging.

## ✨ Features

### 🎯 **Two Complementary Approaches**
- **`CalibratedTree`**: Uses out-of-bag (OOB) predictions for calibration with Bayesian priors
- **`CalibratedForest`**: Uses train-test split approach for calibration

### 🚀 **Professional Enhancements**
- **📚 Comprehensive NumPy-style documentation** with mathematical formulations
- **🔒 Robust parameter validation** and informative error handling  
- **⚡ Type hints throughout** for better IDE support and code safety
- **🎨 Beautiful logging** with loguru for training progress tracking
- **🔧 Flexible parameter support** - pass any sklearn tree/forest parameters
- **⚖️ Advanced sample weighting** for imbalanced datasets and time-weighted models
- **🧪 Full sklearn API compatibility** with proper `get_params`/`set_params`
- **🐍 Pythonic implementation** following best practices and PEP standards

### 💼 **Production Ready**
- Enterprise-grade error handling and validation
- Comprehensive logging for debugging and monitoring
- Memory-efficient implementations
- Extensive testing and validation

![](analysis/hastie-results.png)

## 📦 Installation

### Quick Install
```bash
pip install git+https://github.com/AI-Ahmed/califorest.git
```

### Development Install
```bash
git clone https://github.com/AI-Ahmed/califorest.git
cd califorest
pip install -e .
```

### Using uv (recommended for faster installation)
```bash
uv pip install git+https://github.com/AI-Ahmed/califorest.git
```

## 🚀 Quick Start

### Basic Usage

```python
from califorest import CalibratedTree, CalibratedForest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Option 1: CalibratedTree (OOB-based calibration)
clf_tree = CalibratedTree(
    n_estimators=100,
    max_depth=5,
    ctype="isotonic",
    verbose=True,  # Enable beautiful logging
    random_state=42
)

# Option 2: CalibratedForest (train-test split calibration)  
clf_forest = CalibratedForest(
    n_estimators=50,
    max_depth=3,
    ctype="logistic", 
    test_size=0.3,
    verbose=True,
    random_state=42
)

# Train and predict
clf_tree.fit(X_train, y_train)
y_pred_proba = clf_tree.predict_proba(X_test)
y_pred = clf_tree.predict(X_test)
```

### Advanced Usage with Custom Parameters

```python
# Advanced configuration with sklearn tree parameters
clf_advanced = CalibratedTree(
    n_estimators=200,
    max_depth=7,
    ctype="isotonic",
    alpha0=150,  # Bayesian prior alpha
    beta0=30,    # Bayesian prior beta
    verbose=True,
    random_state=42
)

# Use any sklearn DecisionTreeClassifier parameters
sample_weights = np.where(y_train == 1, 2.0, 1.0)  # Weight minority class

clf_advanced.fit(
    X_train, y_train,
    # Tree initialization parameters
    class_weight="balanced",           # Handle imbalanced data
    ccp_alpha=0.01,                   # Cost complexity pruning
    max_leaf_nodes=50,                # Limit tree complexity
    min_impurity_decrease=0.001,      # Early stopping
    # Tree fitting parameters  
    sample_weight=sample_weights      # Sample importance weights
)

# Get well-calibrated probabilities
probabilities = clf_advanced.predict_proba(X_test)
print(f"Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
```

### Financial/Healthcare Applications

```python
# Example: Risk prediction model with advanced constraints
clf_risk = CalibratedForest(
    n_estimators=150,
    max_depth=4,
    ctype="isotonic",
    test_size=0.25,
    verbose=True,
    random_state=42
)

# Configure for high-stakes predictions
clf_risk.fit(
    X_financial, y_risk,
    class_weight={0: 1, 1: 10},       # High penalty for missing high-risk cases
    sample_weight=recency_weights,     # Recent data weighted higher
    criterion="entropy",               # Information gain splitting
    max_features="log2",              # Feature selection strategy
    bootstrap=True,                   # Ensemble diversity
    min_impurity_decrease=0.002       # Conservative splitting
)
```

## 📊 Calibration Metrics

```python
from califorest import metrics as em
from sklearn.metrics import roc_auc_score

# Get predictions
y_pred_proba = clf_tree.predict_proba(X_test)[:, 1]

# Evaluate performance and calibration
score_auc = roc_auc_score(y_test, y_pred_proba)
score_hl = em.hosmer_lemeshow(y_test, y_pred_proba)
score_sh = em.spiegelhalter(y_test, y_pred_proba)
score_b, score_bs = em.scaled_Brier(y_test, y_pred_proba)
rel_small, rel_large = em.reliability(y_test, y_pred_proba)

print(f"AUC: {score_auc:.3f}")
print(f"Hosmer-Lemeshow: {score_hl:.3f}")
print(f"Scaled Brier Score: {score_bs:.3f}")
```

## 🎯 When to Use Which Class

### `CalibratedTree` 
**Best for:**
- Maximum data efficiency (uses all data for training)
- Out-of-bag error estimation
- Bayesian calibration with priors
- Larger datasets where OOB samples are sufficient

### `CalibratedForest`
**Best for:**
- Clean separation between training and calibration
- Simpler conceptual model
- When you prefer explicit train/validation splits
- Smaller datasets where explicit holdout is preferred

## 🔧 API Reference

### CalibratedTree Parameters

```python
CalibratedTree(
    n_estimators=300,                    # Number of trees
    criterion="gini",                    # Split criterion  
    max_depth=5,                        # Maximum tree depth
    min_samples_split=2,                # Min samples to split
    min_samples_leaf=1,                 # Min samples per leaf
    ctype="isotonic",                   # Calibration type: "isotonic" or "logistic"
    alpha0=100,                         # Bayesian prior alpha
    beta0=25,                          # Bayesian prior beta  
    verbose=False,                      # Enable logging
    random_state=None                   # Random seed
)
```

### CalibratedForest Parameters

```python
CalibratedForest(
    n_estimators=30,                    # Number of trees  
    max_depth=3,                       # Maximum tree depth
    min_samples_split=2,               # Min samples to split
    min_samples_leaf=1,                # Min samples per leaf
    ctype="isotonic",                  # Calibration type: "isotonic" or "logistic"
    test_size=0.3,                     # Proportion for calibration
    verbose=False,                     # Enable logging
    random_state=None                  # Random seed
)
```

### Supported sklearn Parameters

Both classes support **any** parameter from their underlying sklearn estimators:

**Tree Parameters (CalibratedTree):**
- `class_weight`, `ccp_alpha`, `max_leaf_nodes`, `min_impurity_decrease`
- `splitter`, `max_features`, and more...

**Forest Parameters (CalibratedForest):**  
- `class_weight`, `criterion`, `max_features`, `bootstrap`
- `oob_score`, `n_jobs`, `warm_start`, and more...

**Fitting Parameters (Both):**
- `sample_weight`, `check_input`

## 🔬 Algorithm Details

### Bayesian Calibration (CalibratedTree)

The calibration uses Bayesian weighting where sample weights are computed as:

```
w_i = α_i / β_i
```

Where:
- `α_i = α_0 + n_oob,i / 2`  
- `β_i = β_0 + Var(y_oob,i) × n_oob,i / 2`

This approach provides theoretically grounded uncertainty quantification.

### Train-Test Calibration (CalibratedForest)

Uses a clean train-test split approach:
1. Split data into training and calibration sets
2. Train RandomForestClassifier on training data
3. Use calibration set to fit isotonic/logistic calibrator
4. Apply calibration to new predictions

## 🏥 Applications

- **Healthcare**: Risk prediction, diagnosis support, treatment recommendation
- **Finance**: Credit scoring, fraud detection, risk assessment  
- **Marketing**: Customer lifetime value, churn prediction
- **Operations**: Predictive maintenance, quality control
- **Research**: Any application requiring well-calibrated probability estimates

## 📈 Performance

CaliForest provides:
- **Improved calibration** over standard Random Forest
- **Maintained predictive performance** (AUC, accuracy)
- **Better reliability** for probability-based decisions
- **Uncertainty quantification** for risk-aware applications

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit issues or pull requests.

## 📄 License

Apache License 2.0

## 👥 Contributors

- **Yubin Park** - Original author and creator of CaliForest algorithm
- **Ahmed Nabil Atwa** - Professional enhancements, modern API, and comprehensive improvements

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@inproceedings{park2020califorest,
    title={CaliForest: Calibrated Random Forest for Health Data},
    author={Park, Yubin and Ho, Joyce C},
    booktitle={ACM Conference on Health, Inference, and Learning},
    year={2020}
}
```

## 🔗 References

- **Original Paper**: Y. Park and J. C. Ho. 2020. **CaliForest: Calibrated Random Forest for Health Data**. *ACM Conference on Health, Inference, and Learning (2020)*
- **Original Repository**: https://github.com/yubin-park/califorest
- **Enhanced Repository**: https://github.com/AI-Ahmed/califorest

---

*Built with ❤️ for the machine learning community*
