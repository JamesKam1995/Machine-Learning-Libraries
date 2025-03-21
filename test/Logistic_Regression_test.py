from data.loader import x_train, x_test, y_train, y_test
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from Regression.Logistic_regression import LogisticRegression as CustomLogisticRegression

lr = CustomLogisticRegression(lr=0.1)
print("loading")
lr.fit(x_train, y_train, n_iters=150)
pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)

model = LogisticRegression(solver='newton-cg', max_iter=150)
model.fit(x_train, y_train)
pred2 = model.predict(x_test)
accuracy2 = accuracy_score(y_test, pred2)
print(accuracy2)