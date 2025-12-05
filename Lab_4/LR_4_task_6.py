import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

np.random.seed(0)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X ** 2 + X + 2 + np.random.randn(m, 1)

def plot_learning_curves(model, X, y, title):
    train_errors, val_errors = [], []
    for m_subset in range(1, len(X)+1):
        model.fit(X[:m_subset], y[:m_subset])
        y_train_predict = model.predict(X[:m_subset])
        y_val_predict = model.predict(X)
        train_errors.append(mean_squared_error(y[:m_subset], y_train_predict))
        val_errors.append(mean_squared_error(y, y_val_predict))
    plt.plot(np.sqrt(train_errors), label='Помилка на навчальних даних')
    plt.plot(np.sqrt(val_errors), label='Помилка на перевірочних даних')
    plt.title(title)
    plt.xlabel('Кількість навчальних прикладів')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

# Лінійна регресія
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, "Криві навчання - лінійна регресія")

# Поліноміальна регресія 2-го ступеня
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly_2 = poly_features_2.fit_transform(X)
poly_reg_2 = LinearRegression()
plot_learning_curves(poly_reg_2, X_poly_2, y, "Криві навчання - поліноміальна регресія 2-го ступеня")
