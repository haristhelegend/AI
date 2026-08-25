from sklearn.neural_network import MLPClassifier

# Simple data: Inputs
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
# Outputs (Logical XOR pattern: different inputs = 1, same inputs = 0)
y = [0, 0, 1, 1]

# 1. Initialize a Multi-Layer Perceptron (Neural Network)
# hidden_layer_sizes=(2,) means we have a hidden layer with 2 neurons
net = MLPClassifier(hidden_layer_sizes=(2,), max_iter=2000, random_state=42)

# 2. Train the network
net.fit(X, y)

# 3. Test the network with a new pattern [1, 0]
prediction = net.predict([[1, 0]])

print(f"Prediction for [1, 0]: {prediction[0]}")