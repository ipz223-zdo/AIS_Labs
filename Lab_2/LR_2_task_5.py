import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# ------------------- Завантаження даних -------------------
iris = load_iris()
X, y = iris.data, iris.target

# ------------------- Поділ на тренувальні та тестові дані -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ------------------- Класифікатор Ridge -------------------
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)

# ------------------- Прогноз -------------------
y_pred = clf.predict(X_test)

# ------------------- Метрики якості -------------------
accuracy = np.round(metrics.accuracy_score(y_test, y_pred), 4)
precision = np.round(metrics.precision_score(y_test, y_pred, average='weighted'), 4)
recall = np.round(metrics.recall_score(y_test, y_pred, average='weighted'), 4)
f1 = np.round(metrics.f1_score(y_test, y_pred, average='weighted'), 4)
kappa = np.round(metrics.cohen_kappa_score(y_test, y_pred), 4)
matthews = np.round(metrics.matthews_corrcoef(y_test, y_pred), 4)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Cohen Kappa Score:", kappa)
print("Matthews Corrcoef:", matthews)

print("\n\t\tClassification Report:\n", metrics.classification_report(y_test, y_pred, target_names=iris.target_names))

# ------------------- Матриця плутанини -------------------
mat = confusion_matrix(y_test, y_pred)

sns.set(style="whitegrid")
plt.figure(figsize=(6, 5))
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Ridge Classifier')
plt.tight_layout()
plt.savefig("Confusion.jpg")
plt.show()

# Збереження SVG у пам'ять
f = BytesIO()
plt.savefig(f, format="svg")

print("\nЗображення 'Confusion.jpg' збережено у поточній директорії.")
