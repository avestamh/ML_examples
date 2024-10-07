import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

## Logestic Regression Class

class LogesticRegression():
    
    def __init__(self, lr=0.01, iteration=1000):

        self.lr = lr
        self.iteration = iteration

    ## train the model
    def fit(self, x, y):
        ## initialize weight and Biases
        self.m, self.n = x.shape
        self.weight = np.zeros(self.n)
        self.bias = 0
        
        # Gredient descent
        for _ in range(self.iteration):
            model = np.dot(x, self.weight) + self.bias
            prediction = sigmoid(model)

            ## calculate gradient descent
            dw = (1 / self.m) * np.dot(x.T, (prediction - y))
            db = (1 / self.m) * np.sum(prediction - y)

              #  update weights  
            self.weight -= self.lr * dw
            self.bias -= self.lr * db


    ## make prediction
    def predict(self, x):
        model = np.dot(x, self.weight) + self.bias
        prediction = sigmoid(model)
        return [1 if i > 0.5 else 0 for i in prediction]

# Example usage
x = np.array([[0.1, 0.2], [0.4, 0.6], [0.8, 0.9], [0.2, 0.1], [0.5, 0.4]])
y = np.array([0, 0, 1, 0, 1])

model = LogesticRegression()
model.fit(x, y)

# predict
predictions = model.predict(x)
print("Predictions:", predictions)

### Output 
### Predictions: [0, 0, 1, 0, 0]
