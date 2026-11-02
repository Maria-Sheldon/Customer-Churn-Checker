import pandas as pd, numpy as np, json, joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv("data/telco_customer_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")   # 11 blank strings -> NaN (new customers, tenure=0)
df = df.drop(columns="customerID")                                         # identifier, no predictive value
y = (df.pop("Churn") == "Yes").astype(int)
num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat = [c for c in df.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)

def prep(scale=True):
    n = Pipeline([("imp", __import__("sklearn.impute", fromlist=["x"]).SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else []))
    return ColumnTransformer([("num", n, num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

models = {
 "Logistic Regression": Pipeline([("prep", prep()), ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
 "Decision Tree": Pipeline([("prep", prep(False)), ("m", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42))]),
 "Random Forest": Pipeline([("prep", prep(False)), ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1))]),
 "KNN": Pipeline([("prep", prep()), ("m", KNeighborsClassifier(n_neighbors=15))]),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
res = {}
for k, p in models.items():
    cvf = cross_val_score(p, X_tr, y_tr, cv=cv, scoring="f1").mean()
    p.fit(X_tr, y_tr); pr = p.predict(X_te); pb = p.predict_proba(X_te)[:, 1]
    res[k] = dict(accuracy=accuracy_score(y_te, pr), precision=precision_score(y_te, pr), recall=recall_score(y_te, pr),
                  f1=f1_score(y_te, pr), roc_auc=roc_auc_score(y_te, pb), cv_f1=cvf, confusion_matrix=confusion_matrix(y_te, pr).tolist())
    print(k, {a: round(b, 3) for a, b in res[k].items() if a != "confusion_matrix"})

# tune logistic regression (best recall/AUC balance expected)
gs = GridSearchCV(models["Logistic Regression"], {"m__C": [0.01, 0.1, 1, 10]}, cv=cv, scoring="f1").fit(X_tr, y_tr)
print("best LR params", gs.best_params_)
best = gs.best_estimator_
pr = best.predict(X_te); pb = best.predict_proba(X_te)[:, 1]
res["Logistic Regression (tuned)"] = dict(accuracy=accuracy_score(y_te, pr), precision=precision_score(y_te, pr), recall=recall_score(y_te, pr),
    f1=f1_score(y_te, pr), roc_auc=roc_auc_score(y_te, pb), cv_f1=gs.best_score_, confusion_matrix=confusion_matrix(y_te, pr).tolist())
print("tuned", {a: round(b, 3) for a, b in res["Logistic Regression (tuned)"].items() if a != "confusion_matrix"})

# select final model by F1 (recall-weighted business need) among all
final_name = max(res, key=lambda k: res[k]["f1"])
final = best if "tuned" in final_name else models[final_name]
print("FINAL:", final_name)
joblib.dump(final, "models/churn_pipeline.pkl")
# consistency test
l = joblib.load("models/churn_pipeline.pkl"); assert np.allclose(l.predict_proba(X_te), final.predict_proba(X_te))
# feature importance / coefficients
names = final.named_steps["prep"].get_feature_names_out()
m = final.named_steps["m"]
imp = m.coef_[0] if hasattr(m, "coef_") else m.feature_importances_
fi = sorted(zip([n.split("__")[1] for n in names], map(float, imp)), key=lambda t: -abs(t[1]))[:15]
json.dump(dict(final=final_name, results=res, importance=fi, n_train=len(X_tr), n_test=len(X_te),
               train_churn=float(y_tr.mean()), test_churn=float(y_te.mean())), open("models/metrics.json", "w"), indent=1)
