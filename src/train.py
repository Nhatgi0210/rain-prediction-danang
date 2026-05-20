from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import joblib
import numpy as np

class LogisticRegression:
    """Logistic Regression được triển khai thủ công với Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.classes_ = np.array([0, 1])
        
    def _sigmoid(self, z):
        """Hàm sigmoid"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _compute_loss(self, X, y):
        """Tính binary cross-entropy loss"""
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self._sigmoid(z)
        loss = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
        return loss
    
    def fit(self, X, y):
        """Training với Gradient Descent"""
        np.random.seed(self.random_state)
        m, n = X.shape
        
        # Khởi tạo weights và bias
        self.weights = np.zeros(n)
        self.bias = 0
        
        # Gradient Descent
        for iteration in range(self.max_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            h = self._sigmoid(z)
            
            # Backward pass - tính gradient
            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)
            
            # Update weights và bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # In loss sau mỗi 100 iteration
            if iteration % 100 == 0:
                loss = self._compute_loss(X, y)
                print(f"Iteration {iteration}: Loss = {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Trả về xác suất cho cả 2 class"""
        z = np.dot(X, self.weights) + self.bias
        prob_1 = self._sigmoid(z)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X):
        """Dự đoán class (0 hoặc 1)"""
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

def train(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):

    # Khởi tạo các mô hình
    lr_custom = LogisticRegression(learning_rate=0.01, max_iter=1000, random_state=42)
    lr_sklearn = SklearnLogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                class_weight='balanced', random_state=42, n_jobs=-1)

    # Train từng mô hình
    print("\n=== Training Logistic Regression (Custom Implementation) ===")
    lr_custom.fit(X_train_scaled, y_train)
    
    print("\n=== Training Logistic Regression (Sklearn) ===")
    lr_sklearn.fit(X_train_scaled, y_train)
    
    print("\n=== Training Decision Tree ===")
    dt.fit(X_train, y_train)
    
    print("\n=== Training Random Forest ===")
    rf.fit(X_train, y_train)

    # Voting Classifier
    voting = VotingClassifier(estimators=[
        ('lr_sklearn', SklearnLogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
        ('dt', DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5,
                                      class_weight='balanced', random_state=42, n_jobs=-1))
    ], voting='soft')
    print("\n=== Training Voting Classifier ===")
    voting.fit(X_train, y_train)

    # Lưu mô hình vào thư mục models/
    joblib.dump(lr_custom, "models/logistic_regression_custom.pkl")
    joblib.dump(lr_sklearn, "models/logistic_regression_sklearn.pkl")
    joblib.dump(lr_sklearn, "models/logistic_regression.pkl")  # giữ lại tên file cũ cho web app
    joblib.dump(dt, "models/decision_tree.pkl")
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(voting, "models/voting_classifier.pkl")
    print("\nĐã lưu mô hình vào thư mục models/")

    # Đánh giá
    models = {
        'Logistic Regression (Custom)': (lr_custom,  X_test_scaled),
        'Logistic Regression (Sklearn)': (lr_sklearn, X_test_scaled),
        'Decision Tree':                 (dt,        X_test),
        'Random Forest':                 (rf,        X_test),
        'Voting Classifier':             (voting,    X_test),
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
    print("\n=== Kết quả ===")
    print(df_results)
    return models, df_results

if __name__ == "__main__":
    from preprocess import preprocess
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess()
    train(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
