from copy import deepcopy
import numpy as np
#Debugging import
import importlib
var = 'TensorDecisionTreeRegressorP' #the published version of code
package = importlib.import_module(var)
for name, value in package.__dict__.items():
    if not name.startswith("__"):
        globals()[name] = value


from TensorDecisionTreeRegressorP import *
class TensorRandomForestRegressor:
    def __init__(self, n_estimators, max_depth, min_samples_split, split_method, split_rank, CP_reg_rank, Tucker_reg_rank, n_mode, sample_rate, partition_size=4, const_list = None):
        self.n_estimators = n_estimators
        self.models = []
        self.partitions = []
        self.sample_rate = sample_rate
        self.partition_size = partition_size

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

    def fit(self, X, y, X_test=None, y_test=None):
        n_samples, depth, height, width = X.shape  # Assuming X is a 4-mode tensor
        self.partitions = []
        
        for i in range(self.n_estimators):
            # Randomly select slices (features) for the current tree
            partition_indices = np.random.choice(range(width), size=self.partition_size, replace=False)
            self.partitions.append(partition_indices)
            
            # Bootstrapping for each tree
            bootstrap_indices = np.random.choice(range(n_samples), size=int(n_samples * self.sample_rate), replace=True)
            X_partition = X[bootstrap_indices][:, :, :, partition_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Train the current tree with the partitioned tensor
            self.models[i].fit(X_partition, y_bootstrap)

            # Individual tree training RMSE
            train_predictions = self.models[i].predict(X_partition)
            mse_train = np.mean((y_bootstrap - train_predictions) ** 2)
            var_y = np.var(y_bootstrap)
            rmse_train = mse_train / var_y if var_y > 0 else mse_train
            print(f"Tree {i+1}/{self.n_estimators}, Training RMSE (relative to variance): {rmse_train}")
            
            # Optional individual tree test RMSE
            if X_test is not None and y_test is not None:
                X_test_partition = X_test[:, :, :, partition_indices]
                test_predictions = self.models[i].predict(X_test_partition)
                mse_test = np.mean((y_test - test_predictions) ** 2)
                var_y_test = np.var(y_test)
                rmse_test = mse_test / var_y_test if var_y_test > 0 else mse_test
                print(f"Tree {i+1}/{self.n_estimators}, Testing RMSE (relative to variance): {rmse_test}")

            # Calculate cumulative training RMSE for the growing forest
            forest_train_predictions = self.predict(X, n_models=i+1)  # Only use the first `i+1` models
            forest_mse_train = np.mean((y - forest_train_predictions) ** 2)
            forest_rmse_train = forest_mse_train / np.var(y) if np.var(y) > 0 else forest_mse_train
            print(f"Forest after {i+1} trees, Training RMSE (relative to variance): {forest_rmse_train}")
            
            # Calculate cumulative testing RMSE for the growing forest if test data is provided
            if X_test is not None and y_test is not None:
                forest_test_predictions = self.predict(X_test, n_models=i+1)  # Only use the first `i+1` models
                forest_mse_test = np.mean((y_test - forest_test_predictions) ** 2)
                forest_rmse_test = forest_mse_test / np.var(y_test) if np.var(y_test) > 0 else forest_mse_test
                print(f"Forest after {i+1} trees, Testing RMSE (relative to variance): {forest_rmse_test}")

    def predict(self, X, regression_method='mean', n_models=None):
        # Start by averaging predictions from only the first `n_models` trees
        predictions = []
        
        # Use only the first `n_models` models if specified
        models_to_use = self.models[:n_models] if n_models is not None else self.models
        partitions_to_use = self.partitions[:n_models] if n_models is not None else self.partitions
        
        for model, partition_indices in zip(models_to_use, partitions_to_use):
            # Select the same partition (slice) for prediction as was used during training
            X_partition = X[:, :, :, partition_indices]
            
            # Predict using the current tree and append the result
            predictions.append(model.predict(X_partition, regression_method))
        
        # Return the mean of predictions (ensemble averaging)
        return np.mean(predictions, axis=0)