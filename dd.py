from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = 50
X = np.random.uniform(-1, 1, size=(n, 1))
y = 3*X.squeeze() + 0.1*np.random.randn(n)  # very low noise

X_train, X_test = X[:30], X[30:]
y_train, y_test = y[:30], y[30:]

degrees = range(1, 60)
test_errors = []

for d in degrees:
    poly = PolynomialFeatures(d)
    Xtr = poly.fit_transform(X_train)
    Xte = poly.transform(X_test)

    model = Ridge(alpha=1e-3)  # small regularization
    model.fit(Xtr, y_train)

    test_errors.append(mean_squared_error(y_test, model.predict(Xte)))

plt.plot(degrees, test_errors, label="test error")
plt.axvline(len(X_train), color="black", linestyle="--", label="# train samples")
plt.xlabel("Model capacity (degree)")
plt.ylabel("MSE")
plt.legend()
plt.title("Double Descent with Ridge")
plt.show()

plt.savefig("double_descent.png", dpi=200)
print("Plot saved as double_descent.png")
