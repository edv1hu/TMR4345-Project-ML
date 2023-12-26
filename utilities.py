 # Import necessary libraries
import numpy as np  # For mathematical operations
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For plotting 
import pandas as pd    # For data handling
from IPython.display import display

import warnings

# Regression libraries
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV #use ridge or lasso - rigdecv or lassoCV, add range for alfa term
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


sns.set(style="darkgrid", font_scale=1.5)
warnings.filterwarnings('ignore', category=FutureWarning)


def plot_histograms(data_frame):
    # Create a figure with subplots to plot the histograms
    fig, axs = plt.subplots(7, 2, figsize=(15, 30))

    # Loop over each column in the DataFrame
    for i, col in enumerate(data_frame.columns):
        # Calculate the row and column indices for the current plot
        row = i // 2
        cl = i % 2

        # Calculate the mean, median, and mode of the data
        mean = data_frame[col].mean()
        median = data_frame[col].median()
        mode = data_frame[col].mode()

        # Plot a histogram of the data on the current subplot
        sns.histplot(data=data_frame[col], bins=100, kde=False, ax=axs[row, cl])
        axs[row, cl].set_xlabel(col)

        # Mark the mean, median, and mode on the plot
        axs[row, cl].axvline(mode[0], color='b', linestyle='dashed', linewidth=2, label='Mode')

        # If there is more than one mode, mark them all on the plot
        if len(mode) > 1:
            for j in range(1, len(mode)):
                axs[row, cl].axvline(mode[j], color='b', linestyle='dashed', linewidth=2)

        axs[row, cl].axvline(mean, color='r', linestyle='dashed', linewidth=2, label='Mean')
        axs[row, cl].axvline(median, color='g', linestyle='dashed', linewidth=2, label='Median')

        axs[row, cl].legend()

    # Adjust the layout of the subplots and display the figure
    plt.tight_layout()
    plt.show()


def plot_grouped_scatterplots(data_frame, group_column, outlier_data_frame=None):
    """
    Plots scatterplots for each group in the given DataFrame.

    :param data_frame: DataFrame to plot
    :param group_column: Column name to group by
    :param outlier_data_frame: DataFrame containing outlier data (optional)
    """
    # Group the data by the specified column
    grouped_data = data_frame.groupby(group_column)

    # Loop through each group
    for group, df_group in grouped_data:
        # Create a figure with subplots
        fig, axs = plt.subplots(7, 2, figsize=(35, 25))

        # Add the group number to the figure title
        plt.suptitle(f'Group {group}', fontsize=40)

        # Loop through each feature column of the group data
        for i, col in enumerate(df_group.columns):
            # Determine the row and column position of the subplot
            row = i // 2
            cl = i % 2

            # Plot the values of the feature for each point in the group
            sns.scatterplot(data=df_group[col], ax=axs[row, cl], color='blue', marker='.')

            # If outlier data is provided, plot it
            if outlier_data_frame is not None:
                sns.scatterplot(data=outlier_data_frame.loc[outlier_data_frame[group_column] == group, col], 
                                ax=axs[row, cl], color='red', marker='.')

            # Set the x-axis label of the subplot to the feature name
            axs[row, cl].set_xlabel(col)

        # Adjust the layout of the subplots to fit in the figure
        plt.tight_layout()
        # Display the figure
        plt.show()


def divide_data(df, test_trip_nos, valid_trip_no):
    """
    Divides the data into training, validation, and test sets based on trip numbers.

    :param df: DataFrame to be divided.
    :param test_trip_nos: List of trip numbers to be used for the test set.
    :param valid_trip_no: Single trip number to be used for the validation set.
    :return: A tuple containing the training, validation, and test DataFrames.
    """
    testSamps = df[df['Trip_no'].isin(test_trip_nos)].index.to_numpy()
    validSamps = df[df['Trip_no'] == valid_trip_no].index.to_numpy()
    trainSamps = np.setxor1d(df.index.to_numpy(), np.union1d(testSamps, validSamps))

    train_df = df.loc[trainSamps].copy()
    valid_df = df.loc[validSamps].copy()
    test_df = df.loc[testSamps].copy()

    # Optional: Print the data division ratio
    print('Data division ratio (training : validation : test) = (%.3f : %.3f : %.3f)' % 
          (len(train_df)/len(df), len(valid_df)/len(df), len(test_df)/len(df)))

    return train_df, valid_df, test_df


def plot_data_scatter(train_df, valid_df, test_df, x_var, y_var):
    """
    Plots scatter plots for training, validation, and test DataFrames.

    :param train_df: DataFrame for training data.
    :param valid_df: DataFrame for validation data.
    :param test_df: DataFrame for test data.
    :param x_var: The variable to be plotted on the x-axis.
    :param y_var: The variable to be plotted on the y-axis.
    """
    plt.figure(figsize=[10, 10])

    sns.scatterplot(x=train_df[x_var], y=train_df[y_var], marker='.', color='blue')
    sns.scatterplot(x=valid_df[x_var], y=valid_df[y_var], marker='.', color='orange')
    sns.scatterplot(x=test_df[x_var], y=test_df[y_var], marker='.', color='red')
    
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend(['Training', 'Validation', 'Test'])
    plt.show()


def standardization(df_train, df_val, df_test):
    mean = df_train.mean()
    sigma = df_train.std() 
    std_df_train = (df_train - mean)/ sigma
    std_df_val = (df_val - mean)/ sigma
    std_df_test = (df_test - mean)/ sigma
    return std_df_train, std_df_val, std_df_test

def split_df_to_XY(df_train, df_val, df_test, var_y, var_x):
    X_train = df_train[var_x]
    y_train = df_train[var_y]
    X_val = df_val[var_x]
    y_val = df_val[var_y]
    X_test = df_test[var_x]
    y_test = df_test[var_y]

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_predictions(model, X_train, X_val, X_test, y_train, y_val, y_test):
    '''
    Predicts the target values using a linear regression model.
    returns: dataframes with prediced values and the correct indexes
    '''

    y_train_pred = pd.DataFrame(model.predict(X_train), index=y_train.index)
    y_val_pred = pd.DataFrame(model.predict(X_val), index=y_val.index)
    y_test_pred = pd.DataFrame(model.predict(X_test), index=y_test.index)

    return y_train_pred, y_val_pred, y_test_pred

def calculate_scores(y_true, y_pred):
    '''
    Calculates the R2 score, RMSE, and MAE between the true and predicted target values.

    Parameters:
    y_true (array-like): the true target values.
    y_pred (array-like): the predicted target values.

    Returns:
    r2 (float): the R2 score.
    rmse (float): the root mean squared error.
    mae (float): the mean absolute error.
    '''
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return r2, rmse, mae

def plot_predictions(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, ax, i):
    '''
    Plots the true target values against the predicted target values for the training,
    validation, and test sets.

    Parameters:
    y_train (array-like): the true target values for the training set.
    y_train_pred (array-like): the predicted target values for the training set.
    y_val (array-like): the true target values for the validation set.
    y_val_pred (array-like): the predicted target values for the validation set.
    y_test (array-like): the true target values for the test set.
    y_test_pred (array-like): the predicted target values for the test set.
    ax (Matplotlib Axes): the Axes on which to plot the data.
    col (int): the column index of the Axes.
    row (int): the row index of the Axes.

    Returns:
    None (displays the plot).
    '''

    # Training set
    sns.scatterplot(x=y_train, y=y_train_pred[0], marker='.', color='blue', label='Training', ax=ax[i])

    # Validation set
    sns.scatterplot(x=y_val, y=y_val_pred[0], marker='.', color='orange', label='Validation', ax=ax[i])

    # Test set
    sns.scatterplot(x=y_test, y=y_test_pred[0], marker='.', color='red', label='Test', ax=ax[i])

    # Add diagonal line
    ax[i].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=1)
    ax[i].set_title(y_test.name, fontsize=20)
    ax[i].set_xlabel('True Values')
    ax[i].set_ylabel('Predictions')
    ax[i].legend(loc='upper left')


def plot_timeseries_pred(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, ax, i):
    
    # Real timeseries
    sns.scatterplot(data=y_train, marker='.', color='grey',edgecolor='whitesmoke',label='Real timeseries', ax=ax[i])
    sns.scatterplot(data=y_val, marker='.', color='grey',edgecolor='whitesmoke', ax=ax[i])
    sns.scatterplot(data=y_test, marker='.', color='grey',edgecolor='whitesmoke', ax=ax[i])

    # Training set
    sns.scatterplot(data=y_train_pred[0], marker='.', color='blue',edgecolor='lightsteelblue', label='Training', ax=ax[i])

    # Validation set
    sns.scatterplot(data=y_val_pred[0], color='orange',edgecolor='bisque', marker='.', label='Validation', ax=ax[i])

    # Test set
    sns.scatterplot( data=y_test_pred[0], marker='.', color='r',edgecolor='lightcoral', label='Test', ax=ax[i])

    ax[i].set_title(y_test.name, fontsize=20)
    ax[i].set_xlabel('')
    ax[i].set_ylabel(y_train.name)
    ax[i].legend(loc='upper left')


def plot_std_reg_coef(models, std_df_train, std_df_val, std_df_test, functional_relationship):

    fig, axes = plt.subplots(nrows=1, ncols=len(functional_relationship), figsize=(22, 8), sharey=True, sharex=True)
    fig.suptitle('Standardized Regression Coefficients', fontsize=30)
    for i, ((yVar, xVar), model) in enumerate(zip(functional_relationship.items(),models)):
        X_train, y_train, X_val, y_val, X_test, y_test = split_df_to_XY(std_df_train, std_df_val, std_df_test, yVar, xVar)

        # Get the standardized regression coefficients
        std_coef = model.coef_ / np.std(X_train, axis=0)

        # Making a Coefficients dataframe
        coef_df = pd.DataFrame(std_coef, index=X_train.columns)

        # Create a lineplot of the coefficients
        sns.heatmap(coef_df, cmap='Blues', annot=True, ax=axes[i])
        axes[i].set_title(yVar, fontsize=20)

    plt.show()

def plot_std_reg_coef(models, std_df_train, std_df_val, std_df_test, functional_relationship):

    fig, axes = plt.subplots(nrows=len(functional_relationship), ncols=1, figsize=(12, 8), sharey=True, sharex=True)
    fig.suptitle('Standardized Regression Coefficients', fontsize=16)
    
    for i, ((yVar, xVar), model) in enumerate(zip(functional_relationship.items(),models)):
        X_train, y_train, X_val, y_val, X_test, y_test = split_df_to_XY(std_df_train, std_df_val, std_df_test, yVar, xVar)

        # Get the standardized regression coefficients
        std_coef = model.coef_ / np.std(X_train, axis=0)

        # Making a Coefficients dataframe
        coef_df = pd.DataFrame({'std_coef': std_coef, 'independent_var': X_train.columns})

        # Create a lineplot of the coefficients
        sns.lineplot(data=coef_df, x='independent_var', y='std_coef',marker ='o', ax=axes[i])
        axes[i].set_title(yVar, fontsize=14)

    plt.show()


def lasso_reg_models(functional_relationship, std_df_train, std_df_val, std_df_test, alphas=[0.1, 1.0, 10.0]):
    models = []
    for i, (yVar, xVar) in enumerate(functional_relationship.items()):
        X_train, y_train, _, _, _, _ = split_df_to_XY(std_df_train, std_df_val, std_df_test, yVar, xVar)

        # Create Lasso regression object with cross-validation
        model = LassoCV(alphas=alphas, cv=5)
        model.fit(X_train, y_train)
        models.append(model)
    return models

def ridge_reg_models(functional_relationship, std_df_train, std_df_val, std_df_test, alphas=[0.1, 1.0, 10.0]):
    models = []
    for i, (yVar, xVar) in enumerate(functional_relationship.items()):
        X_train, y_train, _, _, _, _ = split_df_to_XY(std_df_train, std_df_val, std_df_test, yVar, xVar)

        # Create ridge regression object with cross-validation
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(X_train, y_train)
        models.append(model)
    return models

def lin_reg_models(functional_relationship, std_df_train, std_df_val, std_df_test):
    models = []
    for i, (yVar, xVar) in enumerate(functional_relationship.items()):
        X_train, y_train, _, _, _, _ = split_df_to_XY(std_df_train, std_df_val, std_df_test, yVar, xVar)

        # Create linear regression object
        model = LinearRegression()
        model.fit(X_train, y_train)
        models.append(model)
    return models


def evaluate_models(models, functional_relationship, std_df_train, std_df_val, std_df_test):
    
    results_df = pd.DataFrame(columns=['yVar','Model','Training R2 score', 'Training RMSE', 'Training MAE', 'Validation R2 score', 'Validation RMSE', 'Validation MAE', 'Test R2 score', 'Test RMSE', 'Test MAE'])

    fig1, axs1 = plt.subplots(1, len(functional_relationship), figsize=(30, 5))
    fig2, axs2 = plt.subplots(len(functional_relationship), 1, figsize=(30, 20))
    fig1.suptitle('Original vs predicted values', fontsize=40, y=1.1)
    fig2.suptitle('Time-series: Original vs predicted values', fontsize=40, y=1)

    for i, ((yVar, xVar), model) in enumerate(zip(functional_relationship.items(),models)):
        X_train, y_train, X_val, y_val, X_test, y_test = split_df_to_XY(std_df_train, std_df_val, std_df_test, yVar, xVar)
        
        # make predictions on the training, validation, and test sets
        y_train_pred, y_val_pred, y_test_pred = make_predictions(model, X_train, X_val, X_test, y_train, y_val, y_test)

        # calculate the evaluation metrics for each set
        train_scores = calculate_scores(y_train, y_train_pred[0])
        val_scores  =  calculate_scores(y_val, y_val_pred[0])
        test_scores =  calculate_scores(y_test, y_test_pred[0])

        # Store results in DataFrame
        new_row = pd.DataFrame({
            'yVar': [yVar],
            'Model': [type(model).__name__],
            'Training R2 score': [train_scores[0]],
            'Training RMSE': [train_scores[1]],
            'Training MAE': [train_scores[2]],
            'Validation R2 score': [val_scores[0]],
            'Validation RMSE': [val_scores[1]],
            'Validation MAE': [val_scores[2]],
            'Test R2 score': [test_scores[0]],
            'Test RMSE': [test_scores[1]],
            'Test MAE': [test_scores[2]]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True) 



        # plot the predicted values against the actual values for the test set
        plot_predictions(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, axs1, i)
        
        # plot the predicted time series values against the actual values for the test set
        plot_timeseries_pred(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, axs2, i)
        

    # Display results
    display(results_df)
    plt.tight_layout()
    plt.show()
    if isinstance(models[0], LinearRegression) | isinstance(models[0], RidgeCV) | isinstance(models[0], LassoCV):
        plot_std_reg_coef(models, std_df_train, std_df_val, std_df_test, functional_relationship)
    
    return results_df


def MLP_models(functional_relationship, std_df_train, std_df_val, std_df_test):
    models = []
    for (yVar, xVar) in functional_relationship.items():

        X_train, y_train, X_val, y_val, X_test, y_test = split_df_to_XY(std_df_train, std_df_val, std_df_test, yVar, xVar)

        # Train an MLP model
        model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', verbose=1, batch_size=int(0.05 * len(X_train)))
        model.fit(X_train, y_train)
        
        models.append(model)
    
    return models

def plot_residuals(models, functional_relationship, std_df_train, std_df_val, std_df_test):
    
    for i, ((yVar, xVar), model) in enumerate(zip(functional_relationship.items(),models)):
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_df_to_XY(std_df_train, std_df_val, std_df_test, yVar, xVar)
        y_train_pred, y_val_pred, y_test_pred = make_predictions(model, X_train, X_val, X_test, y_train, y_val, y_test)

        residuals_train = y_train - y_train_pred[0]
        residuals_val = y_val - y_val_pred[0]
        residuals_test = y_test - y_test_pred[0]


        # Create a figure with one row and a number of columns equal to the number of features
        fig, axs = plt.subplots(ncols=len(X_train.columns), figsize=(20, 4), sharey=True)
        fig.suptitle(f'Residual plot for {yVar}', fontsize=25, y=1.1)

        # Plot residuals against each variable in a separate subplot
        for j, column in enumerate(X_train.columns):
            sns.scatterplot(x=X_train[column], y=residuals_train, marker='.', color='blue', edgecolor='lightsteelblue', ax=axs[j])
            sns.scatterplot(x=X_val[column], y=residuals_val, marker='.', color='orange', edgecolor='bisque', ax=axs[j])
            sns.scatterplot(x=X_test[column], y=residuals_test, marker='.', color='r', edgecolor='lightcoral', ax=axs[j])
            axs[j].axhline(y=0, linestyle='--', color='black')
            axs[j].set_xlabel(column, fontsize=14)
            axs[j].set_ylabel('Residuals', fontsize=14)
            axs[j].tick_params(labelsize=12)

        # Adjust subplot spacing and show the plot
        plt.show()


def train_evaluate_NLmodels(functional_relationship, std_df_train, std_df_val, std_df_test):
    
    results_df = pd.DataFrame(columns=['yVar', 'Model','Training R2 score', 'Training RMSE', 'Training MAE', 'Validation R2 score', 'Validation RMSE', 'Validation MAE', 'Test R2 score', 'Test RMSE', 'Test MAE'])

    for yVar,xVar in functional_relationship.items():

        X_train, y_train, X_val, y_val, X_test, y_test = split_df_to_XY(std_df_train, std_df_val, std_df_test, yVar, xVar)
        
        # Initialize models
        models = [
            XGBRegressor(n_estimators=100),
            MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam'),
            SVR(kernel='rbf'),
            keras.Sequential([
                layers.Dense(50, activation=tf.nn.relu, input_dim=len(xVar)),
                layers.Dense(1)])
        ]

        for model in models:
            # If the model is a Keras model, compile it before training
            if isinstance(model, keras.models.Sequential):
                model.compile(loss='mean_squared_error', optimizer='adam')
            
            # Train model on training set
            model.fit(X_train, y_train)
            
            # make predictions on the training, validation, and test sets
            y_train_pred, y_val_pred, y_test_pred = make_predictions(model, X_train, X_val, X_test, y_train, y_val, y_test)

            # calculate the evaluation metrics for each set
            train_scores = calculate_scores(y_train, y_train_pred)
            val_scores  =  calculate_scores(y_val, y_val_pred)
            test_scores =  calculate_scores(y_test, y_test_pred)

            # Store results in DataFrame
            new_row = pd.DataFrame({
                    'yVar': [yVar],
                    'Model': [type(model).__name__],
                    'Training R2 score': [train_scores[0]],
                    'Training RMSE': [train_scores[1]],
                    'Training MAE': [train_scores[2]],
                    'Validation R2 score': [val_scores[0]],
                    'Validation RMSE': [val_scores[1]],
                    'Validation MAE': [val_scores[2]],
                    'Test R2 score': [test_scores[0]],
                    'Test RMSE': [test_scores[1]],
                    'Test MAE': [test_scores[2]]
                })
            results_df = pd.concat([results_df, new_row], ignore_index=True) 
            
    
    # Display results
    display(results_df)