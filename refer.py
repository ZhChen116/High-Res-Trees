from copy import deepcopy
import numpy as np
class GradientBoostingRegressor:
    def __init__(self, n_estimators, learning_rate, weak_learner, n_iterations=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.weak_learner = weak_learner
        self.models = [deepcopy(self.weak_learner) for _ in range(n_estimators)]
        self.initial_model = None
        self.n_iterations = n_iterations
        self.pruning = False
        
    def fit(self, X, y, X_test=None, y_test=None, method='mean'):
        # Initialize model with a constant value (baseline)
        self.initial_model = self.initialize_model(y)
        current_pred = self.initial_model * np.ones(shape=y.shape)
        print("fitting started")
        
        # Iteratively add weak learners
        for j in range(self.n_iterations):
            for i in range(self.n_estimators):
                # Calculate the residuals (negative gradient)
                residual = y - current_pred
                print(j, i, 'training RMSE ===', np.mean(residual**2) / np.var(y))
                
                if (X_test is not None) and (y_test is not None):
                    current_pred_test = self.predict(X_test, regression_method=method, n_models = i)
                    residual_test = y_test - current_pred_test
                    print(j, i, 'testing RMSE ===', np.mean(residual_test**2) / np.var(y_test))
                
                # Fit weak learner to the residual
                self.models[i].fit(X, residual)
                
                if self.pruning:
                    self.models[i].prune()
                
                # Update current prediction incrementally
                learner_pred = self.models[i].predict(X, method)
                if learner_pred is not None:
                    current_pred += self.learning_rate * learner_pred

    def predict(self, X, regression_method='mean', n_models=None):
        # Start with initial model prediction
        y_pred = self.initial_model * np.ones(X.shape[0])
        
        # If n_models is None, use all models; otherwise, use only up to n_models
        models_to_use = self.models if n_models is None else self.models[:n_models]
        
        # Add predictions from the selected weak learners
        for learner in models_to_use:
            learner_pred = learner.predict(X, regression_method)
            if learner_pred is not None:
                y_pred += self.learning_rate * learner_pred
        
        return y_pred



    def initialize_model(self, y):
        # Typically initialize with mean of y for regression tasks
        return np.mean(y)

