from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_sklearn(X_train, X_test, y_train, y_test):
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện mô hình
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Dự đoán
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # Độ chính xác
    acc = accuracy_score(y_test, y_pred)
    print("✅ sklearn LR test accuracy:", acc)
    
    # Các thông số khác
    print("\n🔹 Intercept (bias):", model.intercept_)
    print("🔹 Coefficients (weights):")
    print(model.coef_)
    print("🔹 Classes:", model.classes_)
    print("🔹 Number of iterations:", model.n_iter_)
    
    # Báo cáo chi tiết
    print("\n🔹 Classification report:")
    print(classification_report(y_test, y_pred))

    # Ma trận nhầm lẫn
    print("\n🔹 Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler, acc
