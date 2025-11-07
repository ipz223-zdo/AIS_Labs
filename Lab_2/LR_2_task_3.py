# Завантаження необхідних бібліотек
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# --------------------------------------------------
# КРОК 1. Завантаження даних
# --------------------------------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Перевірка розміру датасету
print("Форма датасету:", dataset.shape)

# Перші 20 рядків
print("\nПерші 20 записів:")
print(dataset.head(20))

# Статистичне зведення
print("\nСтатистичне зведення:")
print(dataset.describe())

# Кількість екземплярів кожного класу
print("\nКількість екземплярів у кожному класі:")
print(dataset.groupby('class').size())

# --------------------------------------------------
# КРОК 2. Візуалізація даних
# --------------------------------------------------

# Діаграма розмаху (boxplot)
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.suptitle("Діаграма розмаху атрибутів Iris")
pyplot.show()

# Гістограма
dataset.hist()
pyplot.suptitle("Гістограма розподілу атрибутів Iris")
pyplot.show()

# Матриця діаграм розсіювання
scatter_matrix(dataset)
pyplot.suptitle("Матриця діаграм розсіювання Iris")
pyplot.show()

# --------------------------------------------------
# КРОК 3. Розділення даних на навчальну і тестову вибірки
# --------------------------------------------------
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=0.20, random_state=1
)
print("\nКількість записів у навчальній вибірці:", len(X_train))
print("Кількість записів у тестовій вибірці:", len(X_validation))

# --------------------------------------------------
# КРОК 4. Побудова та оцінка моделей
# --------------------------------------------------

# Список моделей
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Оцінювання моделей
results = []
names = []

print("\nОцінка моделей за точністю (Accuracy):")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# --------------------------------------------------
# Порівняльна візуалізація результатів
# --------------------------------------------------
pyplot.boxplot(results, labels=names)
pyplot.title('Порівняння точності моделей')
pyplot.show()

# --------------------------------------------------
# КРОК 5. Вибір найкращої моделі та фінальна перевірка
# --------------------------------------------------
# Наприклад, зазвичай найкращі результати показує SVM
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\nОцінка якості моделі SVM на тестовому наборі:")
print("Accuracy:", accuracy_score(Y_validation, predictions))
print("Матриця помилок:")
print(confusion_matrix(Y_validation, predictions))
print("\nДокладний звіт класифікації:")
print(classification_report(Y_validation, predictions))
