# 1 = Spam, 0 = Not Spam
# X = number of "bad" words
X = [[0], [1], [2], [5], [6], [7]] 
y = [0, 0, 0, 1, 1, 1] 

from sklearn.linear_model import LogisticRegression

# We use Logistic Regression for classification (Yes/No)
model = LogisticRegression()
model.fit(X, y)

# Let's test an email with 4 bad words
test_email = [[4]]
prediction = model.predict(test_email)
probability = model.predict_proba(test_email)

print(f"Is it Spam? (1=Yes, 0=No): {prediction[0]}")
print(f"Confidence score: {probability[0][1]*100:.2f}%")