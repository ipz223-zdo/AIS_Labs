import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

np.random.seed(0)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.scatter(X, y, color='green', label='Дані')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Згенеровані дані (варіант 2)')
plt.legend()
plt.show()

# Лінійна регресія
linear_reg = LinearRegression()
linear_reg.fit(X, y)
y_pred_linear = linear_reg.predict(X)

plt.scatter(X, y, color='green', label='Дані')
plt.plot(X, y_pred_linear, color='blue', linewidth=2, label='Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна регресія')
plt.legend()
plt.show()

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

print("Коефіцієнти поліноміальної регресії:", poly_reg.coef_)
print("Перехоплення (intercept):", poly_reg.intercept_)

plt.scatter(X, y, color='green', label='Дані')
plt.plot(X, y_pred_poly, color='red', linewidth=2, label='Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Поліноміальна регресія')
plt.legend()
plt.show()

# -----------------------
# Оцінка якості поліноміальної регресії
mae = mean_absolute_error(y, y_pred_poly)
mse = mean_squared_error(y, y_pred_poly)
r2 = r2_score(y, y_pred_poly)

print("\nЯкість поліноміальної регресії:")
print("Mean absolute error (MAE) =", round(mae, 2))
print("Mean squared error (MSE) =", round(mse, 2))
print("R2 score =", round(r2, 2))

# -----------------------
# Математична модель
print("\nМатематична форма моделі: y = {:.2f} + {:.2f}*X + {:.2f}*X^2".format(
    poly_reg.intercept_[0], poly_reg.coef_[0][0], poly_reg.coef_[0][1]
))
