import numpy as np 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from copy import deepcopy

from TensorDecisionTreeRegressorP import *
#Debugging import
import importlib
var = 'TensorDecisionTreeRegressorP'
package = importlib.import_module(var)
for name, value in package.__dict__.items():
    if not name.startswith("__"):
        globals()[name] = value

# Gradient Boosting Regressor Class
class GradientBoostingRegressor:
    def __init__(self, n_estimators, learning_rate, weak_learner, n_iterations=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        #self.loss_function = loss_function
        self.weak_learner = weak_learner
        self.models = [deepcopy(self.weak_learner) for _ in range(n_estimators)]
        self.initial_model = None
        self.n_iterations = n_iterations
        self.pruning = False
        
    def fit(self, X, y, X_test=None, y_test=None, method='mean'):
        # Initialize model with a constant value
        self.initial_model = self.initialize_model(y)
        current_pred = np.mean(y)*np.ones(shape=y.shape)
        print("fitting started")
        # Iteratively add weak learners
        for j in range(self.n_iterations):
            for i in range(self.n_estimators):
                # Calculate predictions of the current model
                current_pred = self.predict(X,regression_method=method)
                #print(j,i,'',current_pred)
                # Calculate negative gradient (residual)
                residual = y - current_pred
                print(j,i,'training RMSE ===',np.mean(residual**2)/np.var(y))
                if (X_test is not None) and (y_test is not None):
                    current_pred_test = self.predict(X_test,regression_method=method)
                    residual_test = y_test - current_pred_test
                    print(j,i,'testing RMSE ===',np.mean(residual_test**2)/np.var(y_test))
                
                #derivative = self.loss_derivative(y, self.predict(X))
                
                # Fit weak learner to residual
                self.models[i].fit(X, residual)
                if self.pruning:
                    self.models[i].prune()
                # Add the fitted weak learner to the ensemble
                #self.models.append(learner)

    def predict(self, X, regression_method='mean'):
        # Start with initial model prediction
        y_pred = self.initial_model * np.zeros(X.shape[0])
        
        # Add predictions from all weak learners
        for learner in self.models:
            #print(learner.predict(X))
            learner_pred = learner.predict(X,regression_method)
            if learner_pred is not None:
                y_pred += self.learning_rate * learner_pred
        
        return y_pred

    def initialize_model(self, y):
        # Typically initialize with mean of y for regression tasks
        return np.mean(y)

    def loss_derivative(self, y, y_pred):
        # Example: derivative of the mean squared error loss
        if self.loss_function=='mse':
            return -2 * (y - y_pred)
        else:
            raise Exception('Not implemented: loss_derivative corresponding to ',self.loss_function)

from sklearn.tree import DecisionTreeRegressor
import numpy as np
from copy import deepcopy

class GeneralizedBoostingRegressor:
    def __init__(self, n_estimators, learning_rate, weak_learner, adaboost_resampling_proportion=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        #self.loss_function = loss_function
        self.weak_learner = weak_learner
        assert adaboost_resampling_proportion <=1
        self.adaboost_resampling_proportion = adaboost_resampling_proportion
        self.models = [deepcopy(self.weak_learner) for _ in range(n_estimators)]
        self.initial_model = None

    def fit(self, X, y):
        # Initialize model with a constant value (mean of y)
        self.initial_model = np.mean(y)
        current_pred = np.full(y.shape, self.initial_model)

        # Initialize sample weights for resampling (if enabled)
        sample_weights = np.ones(X.shape[0]) / X.shape[0]

        # Iteratively fit weak learners
        for i in range(self.n_estimators):
            # Calculate residuals
            residuals = y - current_pred

            # Resample based on current sample weights (if resampling is enabled)
            if self.adaboost_resampling_proportion>0:
                indices = np.random.choice(np.arange(X.shape[0]), size=int(X.shape[0]*self.adaboost_resampling_proportion), p=sample_weights)
                X_resampled, residuals_resampled = X[indices], residuals[indices]
                self.models[i].fit(X_resampled, residuals_resampled)
            else:
                self.models[i].fit(X, residuals)

            # Update predictions
            update = self.models[i].predict(X)
            current_pred += self.learning_rate * update

            # Update sample weights (if resampling is enabled)
            if self.adaboost_resampling_proportion>0:
                # Increase weights for samples with larger residuals
                sample_weights *= np.exp(np.abs(residuals))
                sample_weights /= np.sum(sample_weights)

    def predict(self, X):
        # Start with the initial model's prediction
        y_pred = np.full(X.shape[0], self.initial_model)

        # Add predictions from all weak learners
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)

        return y_pred

from sklearn.utils import resample
class RandomForestRegressor:
    def __init__(self, n_estimators, loss_function, weak_learner):
        self.n_estimators = n_estimators
        self.loss_function = loss_function
        self.weak_learner = weak_learner
        self.models = [deepcopy(self.weak_learner) for _ in range(n_estimators)]

    def fit(self, X, y):
        for i in range(self.n_estimators):
            X_sample, y_sample = resample(X, y)  # Bootstrap sampling
            self.models[i].fit(X_sample, y_sample)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)  # Averaging predictions

    def loss_derivative(self, y, y_pred):
        # Example: derivative of the mean squared error loss
        if self.loss_function == 'mse':
            return -2 * (y - y_pred)
        else:
            raise Exception('Not implemented: loss_derivative corresponding to ', self.loss_function)

from sklearn.utils import resample
class TensorSlicingRandomForestRegressor:
    def __init__(self, n_estimators, loss_function, weak_learner, max_features=None):
        self.n_estimators = n_estimators
        self.loss_function = loss_function
        self.weak_learner = weak_learner
        self.models = [deepcopy(self.weak_learner) for _ in range(n_estimators)]
        self.max_features = max_features  # The number of features to consider when looking for the best split

    def fit(self, X, y):
        n_features = X.shape[3]
        self.selected_features = []
        for i in range(self.n_estimators):
            # Bootstrap sampling for rows
            #X_sample, y_sample = resample(X, y)
            # Randomly select features for each tree
            features_idx = np.random.choice(range(n_features), self.max_features, replace=False)  
            #X_sample_features = X_sample[:, features_idx] 
            #X_sample_features = X_sample[...,features_idx]
            X_sample_features = X[...,features_idx]
            y_sample = y
            # Fit the weak learner to the bootstrapped sample and selected features
            print(np.isnan(X_sample_features).any())
            print("Sliced Training X: ", X_sample_features)
            print("shape of X_sample_features: ", X_sample_features.shape)
            self.models[i].fit(X_sample_features, y_sample)
            self.selected_features.append(features_idx)
            #print("weak learner training squared error rate: ", np.mean(self.models[i].predict(X_sample_features)-y_sample)**2)/np.var(y_sample)
    def predict(self, X):
        # Aggregate predictions from each model
        #predictions = np.array([model.predict(X[:, model.feature_importances_ > 0]) for model in self.models])
        predictions = np.array([model.predict(X[..., features_idx]) for model, features_idx in zip(self.models, self.selected_features)])
        # Average predictions
        return np.mean(predictions, axis=0)

    def loss_derivative(self, y, y_pred):
        # Example: derivative of the mean squared error loss
        if self.loss_function == 'mse':
            return -2 * (y - y_pred)
        else:
            raise Exception('Not implemented: loss_derivative corresponding to ', self.loss_function)
        


        import numpy as np
from sklearn.model_selection import train_test_split
from skimage.measure import block_reduce
from copy import deepcopy

# Initialize the Random Forest-like structure
class TensorRandomForestRegressor:
    def __init__(self, n_estimators, max_depth, min_samples_split, split_method, split_rank, CP_reg_rank, Tucker_reg_rank, n_mode, sample_rate, const_list = None):
        self.n_estimators = n_estimators
        self.models = []
        self.partitions = []
        self.sample_rate = sample_rate
        self.const_list = const_list

        # Initialize multiple decision trees (weak learners)
        for _ in range(n_estimators):
            model = TensorDecisionTreeRegressor(max_depth=max_depth, 
                                                min_samples_split=min_samples_split, 
                                                split_method=split_method, 
                                                split_rank=split_rank, 
                                                CP_reg_rank=CP_reg_rank, 
                                                Tucker_reg_rank=Tucker_reg_rank, 
                                                n_mode=n_mode,
                                                const_array=const_list
                                                )
            model.use_mean_as_threshold = False
            model.sample_rate = sample_rate
            self.models.append(model)

    def fit(self, X, y, X_test = None, y_test = None):
        n_samples, depth, height, width = X.shape  # Assuming X is 4-mode tensor
        self.partitions = []
        
        # Randomly partition last dimension for each model
        for i in range(self.n_estimators):
            # Randomly select different slices from the last dimension (for example, groups of size 10)
            partition_indices = np.random.choice(range(width), size=10, replace=False)
            self.partitions.append(partition_indices)
            
            # Select slices for the current tree (i.e., random partition from the last dimension)
            X_partition = X[:, :, :, partition_indices]
            
            # Add small random noise to avoid zero values (optional)
            # X_partition += np.random.rand(*X_partition.shape) * 1e-3
            
            # Train the current tree with the partitioned tensor
            self.models[i].fit(X_partition, y)

            # Calculate and print relative training error (MSE normalized by variance)
            train_predictions = self.models[i].predict(X_partition)
            mse_train = np.mean((y - train_predictions) ** 2)
            var_y = np.var(y)
            rmse_train = mse_train / var_y if var_y > 0 else mse_train  # Prevent division by zero
            print(f"Tree {i+1}/{self.n_estimators}, Training RMSE (relative to variance): {rmse_train}")
            
            # Optionally, calculate and print relative test error if test data is provided
            if X_test is not None and y_test is not None:
                X_test_partition = X_test[:, :, :, partition_indices]
                test_predictions = self.models[i].predict(X_test_partition)
                mse_test = np.mean((y_test - test_predictions) ** 2)
                var_y_test = np.var(y_test)
                rmse_test = mse_test / var_y_test if var_y_test > 0 else mse_test  # Prevent division by zero
                print(f"Tree {i+1}/{self.n_estimators}, Testing RMSE (relative to variance): {rmse_test}")

            

    def predict(self, X, regression_method='mean'):
        predictions = []
        
        # Aggregate predictions from each tree
        for i, model in enumerate(self.models):
            # Select the same partition (slice) for prediction as was used during training
            partition_indices = self.partitions[i]
            X_partition = X[:, :, :, partition_indices]
            
            # Predict using the current tree and append the result
            predictions.append(model.predict(X_partition, regression_method))
        
        # Return the mean of predictions (ensemble averaging)
        return np.mean(predictions, axis=0)

# Example of using TensorRandomForestRegressor


from copy import deepcopy
import numpy as np

# Gradient Boosting Regressor Class for LAD_TreeBoost
class LAD_TreeBoost:
    def __init__(self, n_estimators, learning_rate, weak_learner, n_iterations=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.weak_learner = weak_learner
        self.models = [deepcopy(self.weak_learner) for _ in range(n_estimators)]
        self.initial_model = None
        self.n_iterations = n_iterations
        self.pruning = False

    def fit(self, X, y, X_test=None, y_test=None):
        # Initialize the model with the median of y (LAD initialization)
        self.initial_model = self.initialize_model(y)
        current_pred = self.initial_model * np.ones(shape=y.shape)
        print("fitting started")

        # Iteratively add weak learners
        for j in range(self.n_iterations):
            for i in range(self.n_estimators):
                # Calculate predictions of the current model
                current_pred = self.predict(X)

                # Calculate residual: pseudo-response for LAD is sign(y - F_{m-1}(x))
                residual = np.sign(y - current_pred)
                print(f"Iteration {j}, Estimator {i}: Training LAD Loss = {np.mean(np.abs(residual))}")
                
                # Optional: Testing performance
                if (X_test is not None) and (y_test is not None):
                    current_pred_test = self.predict(X_test)
                    residual_test = np.sign(y_test - current_pred_test)
                    print(f"Iteration {j}, Estimator {i}: Testing LAD Loss = {np.mean(np.abs(residual_test))}")

                # Fit weak learner to residual (pseudo-response)
                self.models[i].fit(X, residual)
                if self.pruning:
                    self.models[i].prune()

    def predict(self, X):
        # Start with initial model prediction (constant median value)
        y_pred = self.initial_model * np.ones(X.shape[0])

        # Add predictions from all weak learners
        for learner in self.models:
            learner_pred = learner.predict(X)
            if learner_pred is not None:
                # Update the prediction: Add gamma (median of predictions) for LAD
                y_pred += self.learning_rate * learner_pred
        
        return y_pred

    def initialize_model(self, y):
        # Initialize with the median of y for LAD
        return np.median(y)

