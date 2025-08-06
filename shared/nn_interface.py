class BinaryNNInterface:
    def forward(self, X):
        """Perform foward pass"""
        pass

    def backward(self, X, y, learning_rate=0.01):
        """Perform backward pass"""
        pass

    def predict_proba(self, X):
        pass
    
    def predict(self, X):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass