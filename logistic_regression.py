# importing modules
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay

# importing dataframe
df = pd.read_csv('UCI_Credit_Card.csv', sep=',')

# checking for any missing value in any column(s)
print(df.shape)
print(df.isna().sum().sum())
print(df.duplicated().sum())

# splitting X and y
X = df.loc[:, df.columns != 'default.payment.next.month']
y = df['default.payment.next.month']

# splitting training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# creating pipeline(s)
standard_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
])

smote_model = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(max_iter=1000))
])

under_model = Pipeline([
    ('scaler', StandardScaler()),
    ('under', RandomUnderSampler(random_state=42)),
    ('clf', LogisticRegression(max_iter=1000))
])

over_model = Pipeline([
    ('scaler', StandardScaler()),
    ('over', RandomOverSampler(random_state=42)),
    ('clf', LogisticRegression(max_iter=1000))
])

# creating model
models = {
    'standard': standard_model,
    'smote': smote_model,
    'under': under_model,
    'over': over_model
}

# keep predictions for plotting
all_preds = {}
all_probs = {}

# testing models
for name, pipe in models.items():
    pipe.fit(x_train, y_train)

    preds = pipe.predict(x_test)
    probs = pipe.predict_proba(x_test)[:, 1]

    all_preds[name] = preds
    all_probs[name] = probs

    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds)
    f1_w  = f1_score(y_test, preds, average='weighted')
    auc   = roc_auc_score(y_test, probs)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"F1 (wtd) : {f1_w:.4f}")
    print(f"ROC AUC  : {auc:.4f}")


# ---------- plots ----------

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name in models:
    RocCurveDisplay.from_predictions(y_test, all_probs[name], name=name, ax=ax)
ax.plot([0, 1], [0, 1], 'k--')  # random guess line
ax.set_title('ROC Curves')
plt.show()

# Precision-Recall curves
fig, ax = plt.subplots(figsize=(7, 6))
for name in models:
    PrecisionRecallDisplay.from_predictions(y_test, all_probs[name], name=name, ax=ax)
ax.set_title('Precision-Recall Curves')
plt.show()

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for ax, name in zip(axes.flatten(), models):
    ConfusionMatrixDisplay.from_predictions(y_test, all_preds[name], ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.show()
