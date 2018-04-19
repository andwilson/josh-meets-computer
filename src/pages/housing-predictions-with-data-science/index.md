---
title: "Housing Predictions with Data Science"
date: "2018-04-11"
path: "/housing-predictions-with-data-science/"
category: "Projects"
thumbnail: "/thumbnail.jpg/"
---

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import gridspec
from keras.datasets import boston_housing
from sklearn import preprocessing as p
from sklearn import model_selection as mdl
from keras.models import Sequential
from keras.layers import Dense
```

    Using TensorFlow backend.



```python
# Visualization settings
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (16, 10)
matplotlib.rcParams['ytick.major.pad']='1'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams["axes.labelsize"] = 15

# Print Settings
pd.set_option('display.width', 2000)
pd.set_option('precision', 3)
np.set_printoptions(precision=3, suppress=True)
```

## The Dataset
Boston Housing prices, columns are as follows:

    CRIM - per capita crime rate by town
    ZN -proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS - proportion of non-retail business acres per town.
    CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    NOX - nitric oxides concentration (parts per 10 million)
    RM - average number of rooms per dwelling
    AGE - proportion of owner-occupied units built prior to 1940
    DIS - weighted distances to five Boston employment centres
    RAD - index of accessibility to radial highways
    TAX - full-value property-tax rate per 10,000
    PTRATIO - pupil-teacher ratio by town
    B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT - Percent lower status of the population
    MEDV - Median value of owner-occupied homes in 1000's


```python
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```


```python
x_headers = ["CRIM", "ZN","INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
y_headers = ["MEDV"]
```


```python
trainDF = pd.DataFrame(x_train, columns=x_headers)
trainDF['MEDV'] = y_train
trainDF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.232</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.142</td>
      <td>91.7</td>
      <td>3.977</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>18.72</td>
      <td>15.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.022</td>
      <td>82.5</td>
      <td>2.03</td>
      <td>0.0</td>
      <td>0.415</td>
      <td>7.610</td>
      <td>15.7</td>
      <td>6.270</td>
      <td>2.0</td>
      <td>348.0</td>
      <td>14.7</td>
      <td>395.38</td>
      <td>3.11</td>
      <td>42.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.898</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.631</td>
      <td>4.970</td>
      <td>100.0</td>
      <td>1.333</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>375.52</td>
      <td>3.26</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.040</td>
      <td>0.0</td>
      <td>5.19</td>
      <td>0.0</td>
      <td>0.515</td>
      <td>6.037</td>
      <td>34.5</td>
      <td>5.985</td>
      <td>5.0</td>
      <td>224.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>8.01</td>
      <td>21.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.693</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>6.376</td>
      <td>88.4</td>
      <td>2.567</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>391.43</td>
      <td>14.65</td>
      <td>17.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Number of zero values per column:\n')
_ = [print(h, end='\t') for h in x_headers]
print()
for h in x_headers:
    print((trainDF[h] == 0).sum(), end='\t')
```

    Number of zero values per column:

    CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT
    0	300	0	379	0	0	0	0	0	0	0	0	0

## Observations
 - Whether a house borders the charles river is a boolean value, and most are set to 0.
 - The proportion of residential land zoned for lots over 25,000sq-ft is usually none of it.


```python
fig, axes= plt.subplots(4, 3)
_ = pd.DataFrame(x_train, columns=x_headers).drop('CHAS',1).plot(kind='box',
                                                                 subplots=True,
                                                                 ax=axes,
                                                                 showfliers=False,
                                                                 vert=False,
                                                                 colormap='jet')

_ = fig.suptitle("Spread of Features",fontsize=20)
```


![png](output_9_0.png)



```python
fig, axes= plt.subplots(4, 3)
_ = pd.DataFrame(p.scale(x_train), columns=x_headers).drop('CHAS',1).plot(kind='hist',
                                                                          bins=20,
                                                                          subplots=True,
                                                                          ax=axes)

_ = fig.suptitle("Distribution of Scaled Features",fontsize=20)

for ax in axes.flat: ax.set_ylabel('')
```


![png](output_10_0.png)



```python
fig, axes= plt.subplots(4, 3)
_ = fig.suptitle('Univariate Analysis; Feature vs Housing Price', fontsize=20)

df = pd.DataFrame(x_train, columns=x_headers).drop('CHAS',1)
headers = df.columns
i = 0

for ax in axes.flat:
    ax.set_title(headers[i], fontweight='bold', fontsize=12)
    ax.scatter(df.iloc[:,i], y_train, s=15, marker='o', edgecolor='grey', linewidth='0.5')
    i += 1

plt.subplots_adjust(top=0.92, hspace=0.4)
```


![png](output_11_0.png)


## Observations
- The only feature with a nicely distributed range of values is average number of rooms. The rest seem skewed
- Property Tax rate looks nearly categorical
- accessibility to highways looks pretty categorical
- Room count is correlated to housing price
- Percent lower status of the population is inversely correlated to housing price

## Preprocessing Techniques


### Why Scale?
Most predictive models work best under the assumption that the data is centered about zero, and that all features have an equal magnitude of variance. This equal variance allows the cost function of the model to weight all features equally. If some features are more important than others, they can be scaled differently so that the features variability contributes more to the cost function. Because of this, it's a good idea to translate the dataset so it is centered around zero and scale each feature to have a unit standard deviation.

It's worth noting that some models, like Decision Tree algorithms are robust to different scales.


### Are any two features highly correlated?


```python
scaled_trainDF = pd.DataFrame(p.scale(trainDF), columns=(x_headers + y_headers))
_ = sns.pairplot(
    data = scaled_trainDF,
    vars=scaled_trainDF.columns,
    hue='MEDV')
```


![png](output_16_0.png)





```python
%reload_ext autoreload
from dataviz import graph_animator
from matplotlib import rc
from IPython.display import HTML
%matplotlib notebook

rc('animation', html='html5')
viz = graph_animator()
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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" width="647.2025923260572">


## Pre-step to Model building, visualization
Just to reduce the visualization code in my model training, I'm going to put together a plotting helper.


```python
class plot_handler():
    """
    ' Plot handler to help me control the shape of my subplots better.
    """
    def __init__(self, plot_rows, plot_cols):
        self.rows = plot_rows
        self.cols = plot_cols
        self.fig = plt.figure(facecolor='white', figsize=(16,16))
        self.grid = gridspec.GridSpec(self.rows, self.cols)

        self.grid.update(left=0.1,
                         right=0.9,
                         wspace=0.2,
                         hspace=.15,
                         top=0.9,
                         bottom=0.1)

        self.ax = {}
        self.xlimit = None
        self.ylimit = None

    def __call__(self):
        plt.show

    def add_plot(self, top, bottom, left, right, name, title):
        self.ax[name] = self.fig.add_subplot(self.grid[top:bottom, left:right])
        self.ax[name].set_title(title,fontweight="bold", size=14)

        # self.ax[name].set_xticks([])
        # self.ax[name].set_yticks([])
    def plot_exists(self, name):
        if name in self.ax:
            return True
        else:
            return False

    def plot(self, data, plot_name, ylim=None, c='b', alpha=1.0):
        self.ax[plot_name].plot(data,  '-', c=c, alpha=alpha)

        if not ylim:
            self.ax[plot_name].set_ylim([0,ylim])

```

# Creating Model
The chosen model architecture for this problem was a two layer neural network with a mean squared error loss function. Below is the implementation, which includes the creation of the architecture, calculation of the loss function, as well as the training algorithm.


```python
# Construct simple model
def base_model():
    model = Sequential()

    model.add(Dense(units=64,
                    input_dim=num_features,
                    kernel_initializer='normal',
                    activation='relu',
                    use_bias=True))
    model.add(Dense(units=64,
                    kernel_initializer='normal',
                    activation='relu',
                    use_bias=True))
    model.add(Dense(units=1,
                    kernel_initializer='normal',
                    activation='linear',
                    use_bias=True))
    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['mae'])

    return model


class NeuralNetwork():
    """
    Two (hidden) layer neural network model.
    First and second layer contain the same number of hidden units
    """
    def __init__(self, input_dim, units, std=0.0001):
        self.params = {}
        self.input_dim = input_dim

        self.params['W1'] = np.random.rand(self.input_dim, units)
        self.params['W1'] *= std
        self.params['b1'] = np.zeros((units))

        self.params['W2'] = np.random.rand(units, units)
        self.params['W2'] *= std * 10  # Compensate for vanishing gradients
        self.params['b2'] = np.zeros((units))

        self.params['W3'] = np.random.rand(units, 1)
        self.params['b3'] = np.zeros((1,))


    def mse_loss(self, x, y=None, drop_p=0.9, reg=0.01, evaluate=False, predict=False):

        # Unpack variables
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        N, D = x.shape

        ###############################################
        # Forward Pass
        ###############################################
        Fx = None

        # First Layer
        x1 = np.dot(x, W1) + b1

        # Activation
        a1 = np.maximum(x1, 0)

        # Drop Out
        drop1 = np.random.choice([1,0],size=x1.shape, p=[drop_p, 1-drop_p]) / drop_p
        a1 *= drop1

        # Second Layer
        x2 = np.dot(a1, W2) + b2  

        # Activation
        a2 = np.maximum(x2, 0)

        # Drop Out
        drop2 = np.random.choice([1,0], size=x2.shape, p=[drop_p, 1-drop_p]) / drop_p
        a2 *= drop2

        # Final Layer
        x3 = np.dot(a2, W3) + b3

        # Output
        Fx = x3

        if predict:
            return Fx

        # Mean Squared Error Cost Function
        mse_loss = np.sum((Fx - y.reshape(-1,1))**2, axis=0) / N
        mae_loss = np.sum(np.absolute(Fx - y.reshape(-1,1)), axis=0) / N
        wght_loss = 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**3))
        loss = mse_loss + wght_loss

        if evaluate:
            return {'loss':loss,
                    'mean_absolute_error': mae_loss[0],
                    'mean_squared_error': mse_loss[0],
                    'weight_loss': wght_loss}

        #############################################
        # Backpropagation
        #############################################

        grads = {}

        # Output
        dFx = 2 * (Fx.copy() - y) / N  # [50, 1]

        # Final Layer
        dx3 = np.dot(dFx, W3.T)   
        dW3 = np.dot(x2.T, dFx)
        db3 = np.sum(dFx * N, axis=0)

        # Drop Out
        dx3 *= drop2

        # activation
        da2 = a2.copy()
        da2[da2 > 0] = 1
        da2[da2 <= 0] = 0
        da2 *= dx3

        # Second Layer
        dx2 = np.dot(da2, W2.T)
        dW2 = np.dot(x1.T, da2)
        db2 = np.sum(da2, axis=0)

        # Drop out
        dx2 *= drop1

        # activation
        da1 = a1.copy()
        da1[da1 > 0] = 1
        da1[da1 < 0] = 0
        da1 *= dx2

        # First Layer
        dx1 = np.dot(da1, W1.T)
        dW1 = np.dot(x.T, da1)
        db1 = np.sum(da1, axis=0)

        grads['W3'] = dW3
        grads['b3'] = db3
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1

        grads['W3'] += dW3 * reg
        grads['W2'] += dW2 * reg
        grads['W1'] += dW1 * reg

        return mae_loss, loss, grads


    def fit(self, X, y, validation_data, epochs=80,
            learning_rate=1e-3, learning_rate_decay=0.99,
            reg=1e-5, batch_size=50, dropout_val=0.95):

        assert type(validation_data) == tuple
        x_val, y_val = validation_data

        num_train = X.shape[0]
        iters_per_epoch = max(num_train // batch_size, 1)
        val_acc = 0

        loss_history = []
        val_loss_history = []
        mae_history = []
        val_mae_history = []

        for e in range(epochs):
            for it in range(iters_per_epoch):
                x_batch = None
                y_batch = None

                batch_indices = np.random.choice(num_train,
                                                 batch_size,
                                                 replace=False)

                x_batch = X[batch_indices]
                y_batch = y[batch_indices]

                mae, loss, grads = self.mse_loss(x_batch,
                                            y_batch,
                                            drop_p=dropout_val,
                                            reg=reg)

                val_mae, val_loss, _ = self.mse_loss(x_val, y_val)

                for key in self.params:
                    self.params[key] -= learning_rate * grads[key]

                if it % iters_per_epoch == 0:
                    learning_rate *= learning_rate_decay

            # Record cost values for this epoch
            loss_history.append(loss)
            mae_history.append(mae)
            val_loss_history.append(val_loss)
            val_mae_history.append(val_mae)

        return {'loss': loss_history,
                'mean_absolute_error': mae_history,
                'val_loss': val_loss_history,
                'val_mean_absolute_error': val_mae_history}

    def evaluate(self, X, y):
        return self.mse_loss(X, y, drop_p=1, evaluate=True)

    def predict(self, X):
        return self.mse_loss(X, drop_p=1, predict=True)
```

## Base Model Notes

If the dataset is small, then there won't be enough samples to be statistically representative of the data at hand. Few data points means the model will be very prone to overfitting, and an overly complex model with a large architecture will optimize very well to the training data but will be too finely tuned to be able to generalize to new examples. For this reason, the model was built as a two layer neural network. Relu activations were included for predicting with non-linearities.

The loss function / optimization function on this model is the Mean Squared Error, which calculates the distance between the predicted value and the ground truth value, squares the difference then sums this value across all examples. Squaring the difference will exponentially penalize inaccurate estimates, creating a steeper gradient descent in initial training.

The model will be measured by the mean absolute error. This is just the absolute difference between the predicted value and the actual value. This will describe in how many dollars how off was the estimate.


```python

```

### Validating Model using K-folds
K-fold validation is the technique of splitting your data during training and averaging the results of a model trained on the seperate sets. The process goes like this: split your data into k segments. Create k batches of data, where for each batch, the training set has a unique k-1 segments and the validation set is the remaining 1 segment. Train a new model on each of these batches (or folds), averaging the results.

Generally, if different shuffling of data before splitting into test and train yield a different model performance, then there is not enough data and k-folding would be a good approach.


```python
import matplotlib.patches as mpatches

# join the x and y together for shuffling
y_train = y_train.reshape((y_train.shape[0],1))
data = np.hstack([x_train, y_train])

# Make random generation repeatable
seed = 7
np.random.seed(seed)

# small dataset, shuffling and folding the data
shuff_count, i = 3, 0
split_count = 4
epoch_count = 80
b = 0

# Fit a scaling function to the train data
scale = p.StandardScaler().fit(x_train)

# Plotting Parameters #####################
subplot = ['Mean Absolute Error', 'Loss'] * 2
metrics = ['mean_absolute_error', 'loss', 'val_mean_absolute_error', 'val_loss']
legends = ['Train', 'Train', 'Validation', 'Validation']

history_set = {metric: np.zeros(shape=(epoch_count, shuff_count * split_count)) for metric in metrics}

plotter = plot_handler(2, 1)
plotter.add_plot(top=0,bottom=1,left=0,right=1,
                 name='Loss',
                 title='Average Loss')
plotter.add_plot(top=1,bottom=2, left=0, right=1,
                 name='Mean Absolute Error',
                 title='House Price Error: Estimate vs Actual (in $1000s)')
############################################

best_model = NeuralNetwork(input_dim=x_train.shape[1], units=64)  # base_model()

while i < shuff_count:
    i += 1
    data_folds = mdl.KFold(n_splits=split_count,
                           shuffle=True,
                           random_state=seed).split(data)

    for dfold in data_folds:
        train, valid = dfold

        # Seperate each k fold of train data into a train and validation set
        xt = scale.transform(data[train,:-1])
        yt = data[train,-1:]

        # Transform validation set based on scaling on training set
        xv = scale.transform(data[valid,:-1])
        yv = data[valid,-1:]

        model = NeuralNetwork(input_dim=x_train.shape[1], units=64)  # base_model()
        history = model.fit(xt, yt, validation_data=(xv, yv), epochs=epoch_count)

        for m in metrics:
            # history_set[m][:, b] = history.history[m]
            history_set[m][:, b] = history[m]

        for metric, plot in zip(metrics, subplot):
            color = 'blue' if  metric[:3] == 'val' else 'green'
            plotter.ax[plot].plot(history[metric],
                                  c=color,
                                  linewidth=2,
                                  alpha=0.2)

        # Evaluate each model, keep the best one
        curr_evaluation = model.evaluate(scale.transform(x_test), y_test)  #, verbose=0)
        best_evaluation = best_model.evaluate(scale.transform(x_test), y_test)  #, verbose=0)

        if curr_evaluation['mean_squared_error'] < best_evaluation['mean_squared_error']:
            best_model = model

        b += 1

# Plot average of all folds
for metric, plot in zip(metrics, subplot):
    color = 'blue' if metric[:3] == 'val' else 'green'
    plotter.ax[plot].plot(np.mean(history_set[metric], axis=1),
                          c=color,
                          linewidth=3,
                          alpha=1.0)

# Add Legend
train = mpatches.Patch(color='green', label='Train')
valid = mpatches.Patch(color='blue', label='Validation')

for plot in subplot[:2]:
    plotter.ax[plot].legend(handles=[train, valid])

plotter.ax['Loss'].set_ylim([0,100])
plotter.ax['Mean Absolute Error'].set_ylim([0,10])

plotter()
```


![png](output_26_0.png)


### Notes:

The faded lines show each training session on a unique instantiation of a model. The model's absolute mean error is recorded at the end of each epoch, and the training loss and mean absolute error is plotted over each epoch. A new partition of shuffled data is selected, a new model instiantied, trained and subsequently plotted.

All these training sessions are then averaged together to get a representation of the training and validation loss and mean absolute error.



```python
print('Base Model mean absolute error on test results: $', end='')
mae = best_model.evaluate(scale.transform(x_test),y_test[:, None])['mean_absolute_error']
print(round(mae * 1000, 2))
```

    Base Model mean absolute error on test results: $3056.36


## Determining Feature Importance
Are there certain features that play a larger role in the final prediction? One way to see the importance of certain features is to corrupt one of the features and see how it affects the prediction capabilities of the model. Features can be ranked based on the results of the model's prediction capabilities when the feature is rendered useless.


```python
import pandas as pd
def rankFeatures(x, y, model, column_names, metric='mean_squared_error'):
    """
    Determines which feature contributes most to a prediction
    by shuffling each feature before running an evaluation, then
    seeing which features cause the greatest impact to the models
    prediction capabilities. The act of shuffling has the effect
    of rendering the values of one feature useless.
    """
    num_features = x.shape[1]
    errors = []

    base_err = model.evaluate(x, y[:, None])[metric]

    for i in range(num_features):
        hold = x[:, i]
        np.random.shuffle(x[:, i])

        shuffled_acc = model.evaluate(x, y[:, None])[metric]
        errors.append(shuffled_acc)

        x[:,i] = hold

    max_error = np.max(errors)
    feat_rank = [err / max_error for err in errors]

    errors = [round(err - base_err,2)*1000 for err in errors]
    data = pd.DataFrame({'Features':column_names, 'Increased Error ($)':errors, 'Importance':feat_rank})
    data.sort_values(by=['Importance'], ascending=[0], inplace=True)
    data.reset_index(inplace=True, drop=True)

    return data
```


```python
rankFeatures(scale.transform(x_test), y_test, best_model, x_headers, metric='mean_absolute_error')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>Importance</th>
      <th>Increased Error ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LSTAT</td>
      <td>1.000</td>
      <td>6500.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>0.816</td>
      <td>4750.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PTRATIO</td>
      <td>0.784</td>
      <td>4440.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TAX</td>
      <td>0.761</td>
      <td>4220.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RAD</td>
      <td>0.745</td>
      <td>4070.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AGE</td>
      <td>0.725</td>
      <td>3870.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DIS</td>
      <td>0.716</td>
      <td>3790.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RM</td>
      <td>0.685</td>
      <td>3490.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NOX</td>
      <td>0.497</td>
      <td>1690.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CHAS</td>
      <td>0.421</td>
      <td>970.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>INDUS</td>
      <td>0.418</td>
      <td>940.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ZN</td>
      <td>0.395</td>
      <td>710.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CRIM</td>
      <td>0.377</td>
      <td>550.0</td>
    </tr>
  </tbody>
</table>
</div>



## Notes
Based on the ranking output above, it looks like LSTAT (which is the percent of the lower status population in a neighborhood) Plays the largest role in the models predictive powers. Shuffling of this features data causes significant errors in the model, creating an additional 6,500 dollar deviation in housing predictions from the base models prediction error.


```python

```


```python

```
