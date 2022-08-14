# from https://github.com/onnx/sklearn-onnx/blob/main/docs/tutorial/plot_abegin_convert_pipeline.py

"""
.. _l-simple-deploy-1:
Train and deploy a scikit-learn pipeline
========================================
.. index:: pipeline, deployment
This program starts from an example in :epkg:`scikit-learn`
documentation: `Plot individual and voting regression predictions
<https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_regressor.html>`_,
converts it into ONNX and finally computes the predictions
a different runtime.
.. contents::
    :local:
Training a pipeline
+++++++++++++++++++
"""
from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
import numpy as np
import pandas as pd

from onnxruntime import InferenceSession
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor,
    VotingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skl2onnx import to_onnx
from mlprodict.onnxrt import OnnxInference

# import Py3GUI swlda
from swlda import swlda

import matplotlib.pyplot as plt

from sklearn import datasets
# import vanilla lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.model_selection import train_test_split

# load synthetic data
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels

# load EEG data - what does the data look like?

# TODO LOAD DATASET USING

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# dataset = pd.read_csv(url, names=names) # Returns a DF

def load_csv_data():
    pass
def load_txt_data():
    pass
def load_xdf_data():
    pass

def data_loader(file_name, window, file_type = None, window_in_samples = False):
    if file_type == None:
        if fname.lower().endswith('.txt'):
            file_type = 'txt'
        elif fname.lower().endswith('.xdf'):
            file_type = 'xdf'
    
    elif file_type == 'txt':
        pass

    elif file_type == 'xdf':
        pass
    else
        return '%s file type not supported.' % str(file_type)

iris = datasets.load_iris()

# Preprocess data
# example data from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
# X = dataset.iloc[:, 0:4].values # train and test data
# TODO Question: Labels will be P300 active/inactive, rows/columns or letters?
# y = dataset.iloc[:, 4].values # label and features

X = iris.data
print(X)

y = iris.target
print(y)
target_names = iris.target_names

# Define number of components
# num_components = len(list(set(y)))-1

# NOTE If needed
# Separate data into correct categories
# TODO Check test_size and other params
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# NOTE If needed
# feature scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Create model
lda = LDA(n_components=num_components)
X_r2 = lda.fit(X, y).transform(X)
print("Done")

# Plot (temporary)
plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of IRIS dataset')

plt.show()

# swlda = swlda()

# Train classifiers

# reg1 = GradientBoostingRegressor(random_state=1, n_estimators=5)
# reg2 = RandomForestRegressor(random_state=1, n_estimators=5)
# reg3 = LinearRegression()

# ereg = Pipeline(steps=[
#     ('voting', VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])),
# ])
# ereg.fit(X_train, y_train)

#################################
# Converts the model
# ++++++++++++++++++
#
# The second argument gives a sample of the data
# used to train the model. It is used to infer
# the input type of the ONNX graph. It is converted
# into single float and ONNX runtimes may not fully
# support doubles.

onx = to_onnx(lda, X_train[:1].astype(np.float32),
              target_opset=12)

###################################
# Prediction with ONNX
# ++++++++++++++++++++
#
# The first example uses :epkg:`onnxruntime`.

# sess = InferenceSession(onx.SerializeToString())
# pred_ort = sess.run(None, {'X': X_test.astype(np.float32)})[0]

# pred_skl = ereg.predict(X_test.astype(np.float32))

# print("Onnx Runtime prediction:\n", pred_ort[:5])
# print("Sklearn rediction:\n", pred_skl[:5])

####################################
# .. _l-diff-dicrepencies:
#
# Comparison
# ++++++++++
#
# Before deploying, we need to compare that both
# *scikit-learn* and *ONNX* return the same predictions.


# def diff(p1, p2):
#     p1 = p1.ravel()
#     p2 = p2.ravel()
#     d = np.abs(p2 - p1)
#     return d.max(), (d / np.abs(p1)).max()


# print(diff(pred_skl, pred_ort))

############################################
# It looks good. Biggest errors (absolute and relative)
# are within the margin error introduced by using
# floats instead of doubles.
# We can save the model into ONNX
# format and compute the same predictions in many
# platform using :epkg:`onnxruntime`.

####################################
# Python runtime
# ++++++++++++++
#
# A python runtime can be used as well to compute
# the prediction. It is not meant to be used into
# production (it still relies on python), but it is
# useful to investigate why the conversion went wrong.
# It uses module :epkg:`mlprodict`.

oinf = OnnxInference(onx, runtime="python_compiled")
print(oinf)

##########################################
# It works almost the same way.

# pred_pyrt = oinf.run({'X': X_test.astype(np.float32)})['variable']
# print(diff(pred_skl, pred_pyrt))

#############################
# Final graph
# You may need to install graphviz from https://graphviz.org/download/
# +++++++++++

# ax = plot_graphviz(oinf.to_dot())
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)