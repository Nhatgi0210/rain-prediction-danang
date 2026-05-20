from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import joblib

def train(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):

    # Khởi tạo 3 mô hình cơ bản
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                class_weight='balanced', random_state=42, n_jobs=-1)

    # Train từng mô hình
    lr.fit(X_train_scaled, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Voting Classifier
    voting = VotingClassifier(estimators=[
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
        ('dt', DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                      class_weight='balanced', random_state=42, n_jobs=-1))
    ], voting='soft')
    voting.fit(X_train, y_train)

    # Lưu 4 model vào thư mục models/
    joblib.dump(lr, "models/logistic_regression.pkl")
    joblib.dump(dt, "models/decision_tree.pkl")
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(voting, "models/voting_classifier.pkl")
    print("Đã lưu mô hình vào thư mục models/")

    # Đánh giá
    models = {
        'Logistic Regression': (lr,     X_test_scaled),
        'Decision Tree':       (dt,     X_test),
        'Random Forest':       (rf,     X_test),
        'Voting Classifier':   (voting, X_test),
    }

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
