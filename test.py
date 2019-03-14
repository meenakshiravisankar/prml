from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

# Seeding
np.random.seed(42)

# Generate 100 points in (0,1)
X = np.random.uniform(low=0,high=1,size=100).reshape(-1,1)
# y = np.exp(np.sin(2*np.pi*X)) + X + np.random.normal(0,2)
y = np.sin(2*np.pi*X) + np.random.normal(0,2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=2/3, random_state=42, shuffle=True)

X_train = X_train[:10]
y_train = y_train[:10]
poly = PolynomialFeatures(9)
X_train = poly.fit_transform(X_train)

reg = LinearRegression().fit(X_train, y_train)

print(reg.score(X_train, y_train))
print(reg.coef_)

print(reg.intercept_)

