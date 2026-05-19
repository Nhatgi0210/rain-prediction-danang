from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import joblib

def train(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    models = {}

    # Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = (lr, X_test_scaled)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = (dt, X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = (rf, X_test)

    # Lưu model tốt nhất
    joblib.dump(rf, "models/random_forest.pkl")
    print("Đã lưu Random Forest vào models/random_forest.pkl")

    # Kết quả
    results = []
    for name, (model, X_t) in models.items():
        y_pred = model.predict(X_t)
        y_prob = model.predict_proba(X_t)[:, 1]
        results.append({
            'Mô hình': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'F1-Score': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC':  round(roc_auc_score(y_test, y_prob), 4),
        })

    df_results = pd.DataFrame(results).set_index('Mô hình')
    print(df_results)
    return models, df_results

if __name__ == "__main__":
    from preprocess import preprocess
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess()
    train(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
