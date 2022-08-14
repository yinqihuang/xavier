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
# import vanilla lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# load synthetic data
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels

# load EEG data - what does the data look like?
def load_txt_data():

def load_xdf_data():

def data_loader(file_name, window, file_type = None, window_in_samples = False):
    if file_type == None:
        if fname.lower().endswith('.txt'):
            file_type = 'txt'
        elif fname.lower().endswith('.xdf'):
            file_type = 'xdf'
    
    elif file_type == 'txt':

    elif file_type == 'xdf':
    
    else
        return '%s file type not supported.' % str(file_type)


# X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# call classifiers

swlda = swlda()

lda = LinearDiscriminantAnalysis()

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

onx = to_onnx(ereg, X_train[:1].astype(np.float32),
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

# oinf = OnnxInference(onx, runtime="python_compiled")
# print(oinf)

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