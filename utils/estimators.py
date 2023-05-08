import time
import os

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorboard.plugins.hparams import api as hp


import utils.metrics as losses

HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-3, 1e-4, 1e-5, 1e-6, 1e-7]))
HP_REG_L2 = hp.HParam('reg_l2', hp.Discrete([0.1, 0.01, 0.001, 0.0001, 0.00001]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([16, 32, 64, 128, 256]))
METRIC_LOSS = 'loss'


def predict_by_COM(model, x, keras_model=False):
    """
    Conditional Outcome Modeling (COM), where t is viewed as one feature of the
    covariates. This is also called a T-learner.
    :param model:
    :param x:
    :return:
    """
    if not keras_model:
        y0 = model.predict(
            np.concatenate((x, np.zeros(shape=(x.shape[0], 1))), axis=1)
        )
        y1 = model.predict(
            np.concatenate((x, np.ones(shape=(x.shape[0], 1))), axis=1)
        )
    else:
        y0 = model.predict(
            np.concatenate((x, np.zeros(shape=(x.shape[0], 1))), axis=1),
            verbose=0,
        )
        y1 = model.predict(
            np.concatenate((x, np.ones(shape=(x.shape[0], 1))), axis=1),
            verbose=0,
        )
    return y0, y1


def predict_by_GCOM(models, x, keras_model=False):
    """
    Grouped Conditional Outcome Modeling (GCOM), where for binary t, each t
    corresponds to one model. This is also called a S-learner.
    covariates
    :param models: two models for t=0 and t=1 separately
    :param x:
    :return:
    """
    if not keras_model:
        y0 = models[0].predict(x)
        y1 = models[1].predict(x)
    else:
        y0 = models[0].predict(x, verbose=0)
        y1 = models[1].predict(x, verbose=0)
    return y0, y1


def train_and_estimate_OLS1(x, t, yf, x_test):
    """
    Training using Ordinary Least Squares (OLS), where t is viewed as one
    feature of the covariates
    :param x: covariates
    :param t: treatment
    :param yf: factual outcomes
    :param x_test: out-of-sample covariates
    :return:
    """

    new_x = np.concatenate((x, t[:, np.newaxis]), axis=1)
    OLS = LinearRegression()
    OLS.fit(new_x, yf)

    y0_in, y1_in = predict_by_COM(OLS, x)
    ate_in = np.mean(y1_in - y0_in)
    y0_out, y1_out = predict_by_COM(OLS, x_test)
    ate_out = np.mean(y1_out - y0_out)
    return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out


def train_and_estimate_OLS2(x, t, yf, x_test):
    """
    Training using Ordinary Least Squares (OLS). For binary t, each t
    corresponds to one model.
    :param x: covariates
    :param t: treatment
    :param yf: factual outcomes
    :param x_test: out-of-sample covariates
    :return:
    """
    mask = t == 0  # find the boolean mask of t is 0

    t0 = np.arange(len(t))[mask]
    t1 = np.arange(len(t))[~mask]

    x0 = x[t0]
    x1 = x[t1]

    # fit an estimator for each treatment separately
    OLS_t0 = LinearRegression()
    OLS_t0.fit(x0, yf[t0])
    OLS_t1 = LinearRegression()
    OLS_t1.fit(x1, yf[t1])

    y0_in, y1_in = predict_by_GCOM([OLS_t0, OLS_t1], x)
    ate_in = np.mean(y1_in - y0_in)
    y0_out, y1_out = predict_by_GCOM([OLS_t0, OLS_t1], x_test)
    ate_out = np.mean(y1_out - y0_out)
    return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out


def train_and_estimate_RF1(x, t, yf, x_test):
    """
    Training using Random Forest (RF), where t is viewed as one
    feature of the covariates
    :param x: covariates
    :param t: treatment
    :param yf: factual outcomes
    :param x_test: out-of-sample covariates
    :return:
    """

    new_x = np.concatenate((x, t[:, np.newaxis]), axis=1)
    RF = RandomForestRegressor()
    RF.fit(new_x, yf)

    y0_in, y1_in = predict_by_COM(RF, x)
    ate_in = np.mean(y1_in - y0_in)
    y0_out, y1_out = predict_by_COM(RF, x_test)
    ate_out = np.mean(y1_out - y0_out)
    return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out


def train_and_estimate_RF2(x, t, yf, x_test):
    """
    Training using Random Forest (RF). For binary t, each t
    corresponds to one model.
    :param x: covariates
    :param t: treatment
    :param yf: factual outcomes
    :param x_test: out-of-sample covariates
    :return:
    """
    mask = t == 0  # find the boolean mask of t is 0

    t0 = np.arange(len(t))[mask]
    t1 = np.arange(len(t))[~mask]

    x0 = x[t0]
    x1 = x[t1]

    # fit an estimator for each treatment separately
    RF_t0 = RandomForestRegressor()
    RF_t0.fit(x0, yf[t0])
    RF_t1 = RandomForestRegressor()
    RF_t1.fit(x1, yf[t1])

    y0_in, y1_in = predict_by_GCOM([RF_t0, RF_t1], x)
    ate_in = np.mean(y1_in - y0_in)
    y0_out, y1_out = predict_by_GCOM([RF_t0, RF_t1], x_test)
    ate_out = np.mean(y1_out - y0_out)
    return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out


def train_and_estimate_IPW(x, t, yf, x_test):
    """
    Training using Inverse Probability Weighting (IPW). For binary t, each t
    corresponds to one model.
    :param x: covariates
    :param t: treatment
    :param yf: factual outcomes
    :param x_test: out-of-sample covariates
    :return:
    """
    LR = LogisticRegression()
    LR.fit(x, t)
    propensity = LR.predict_proba(x)[:, 1]  # T = 1

    mask = t == 0  # find the boolean mask of t is 0
    t0 = np.arange(len(t))[mask]
    t1 = np.arange(len(t))[~mask]

    yf0 = yf[t0]
    denominator0 = (1 - propensity)[t0]
    yf1 = yf[t1]
    denominator1 = propensity[t1]

    ate_in = np.mean(yf1 / denominator1) - np.mean(yf0 / denominator0)

    y0_in, y1_in = None, None
    y0_out, y1_out = None, None
    ate_out = None
    return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out


def create_simple_NN(input_dim):
    # model = keras.Sequential(
    #     [
    #         layers.InputLayer(input_shape=(input_dim,), name="input_layer"),
    #         layers.Dense(input_dim, activation="relu", name="first_layer"),
    #         layers.Dense(1, name="output_layer"),
    #     ]
    # )

    # use the backbone NN from TAR-Net
    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(input_dim,), name="input_layer"),
            layers.Dense(
                units=200, activation="elu", kernel_initializer="RandomNormal"
            ),
            layers.Dense(
                units=200, activation="elu", kernel_initializer="RandomNormal"
            ),
            layers.Dense(
                units=200, activation="elu", kernel_initializer="RandomNormal"
            ),
            layers.Dense(1, name="output_layer"),
        ]
    )
    return model


def train_and_estimate_NN1(x, t, yf, x_test):
    """
    Training using simple fully-connected neural networks, where t is viewed as
    one feature of the covariates
    :param x: covariates
    :param t: treatment
    :param yf: factual outcomes
    :param x_test: out-of-sample covariates
    :return:
    """

    new_x = np.concatenate((x, t[:, np.newaxis]), axis=1)
    model = create_simple_NN(new_x.shape[-1])
    model.compile(loss="mean_squared_error", optimizer="adam", metrics="mse")
    model.fit(new_x, yf, epochs=50, verbose=0)
    y0_in, y1_in = predict_by_COM(model, x, keras_model=True)
    ate_in = np.mean(y1_in - y0_in)
    y0_out, y1_out = predict_by_COM(model, x_test, keras_model=True)
    ate_out = np.mean(y1_out - y0_out)
    return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out


def train_and_estimate_NN2(x, t, yf, x_test):
    """
    Training using simple fully-connected neural networks (NN). For binary t,
    each t corresponds to one model.
    :param x: covariates
    :param t: treatment
    :param yf: factual outcomes
    :param x_test: out-of-sample covariates
    :return:
    """
    mask = t == 0  # find the boolean mask of t is 0

    t0 = np.arange(len(t))[mask]
    t1 = np.arange(len(t))[~mask]

    x0 = x[t0]
    x1 = x[t1]

    models = [None, None]
    for i in range(len(models)):
        model = create_simple_NN(x.shape[-1])
        model.compile(
            loss="mean_squared_error", optimizer="adam", metrics="mse"
        )
        if i == 0:
            model.fit(x0, yf[t0], epochs=50, verbose=0)
        elif i == 1:
            model.fit(x1, yf[t1], epochs=50, verbose=0)
        models[i] = model
    y0_in, y1_in = predict_by_GCOM(models, x, keras_model=True)
    ate_in = np.mean(y1_in - y0_in)
    y0_out, y1_out = predict_by_GCOM(models, x_test, keras_model=True)
    ate_out = np.mean(y1_out - y0_out)
    return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out


def train_and_estimate_DML(x, t, yf, x_test):
    """
    Training using double machine learning (DML), i.e., R-Learner.
    :param x: covariates
    :param t: treatment
    :param yf: factual outcomes
    :param x_test: out-of-sample covariates
    :return:
    """
    x_aux, x_new, t_aux, t_new, yf_aux, yf_new = train_test_split(
        x, t, yf, test_size=0.5, shuffle=False
    )

    LR = LogisticRegression()
    LR.fit(x_aux, t_aux)

    LS = Ridge()
    LS.fit(x_aux, yf_aux)

    # get residuals
    t_r = t_new - LR.predict_proba(x_new)[:, 1]
    y_r = yf_new - LS.predict(x_new)

    OLS = LinearRegression()
    OLS.fit(t_r.reshape(-1, 1), y_r)  # single feature

    y_new_predict = OLS.predict(t_r.reshape(-1, 1)) + LS.predict(x_new)
    mask = t_new == 0
    t_new_0 = np.arange(len(t_new))[mask]
    t_new_1 = np.arange(len(t_new))[~mask]

    ate_in = np.mean(y_new_predict[t_new_1]) - np.mean(y_new_predict[t_new_0])

    y0_in, y1_in = None, None
    y0_out, y1_out = None, None
    ate_out = None
    return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out


class EpsilonLayer(layers.Layer):
    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(
            name="epsilon",
            shape=[1, 1],
            initializer="RandomNormal",
            trainable=True,
        )
        super(EpsilonLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]


def make_dragonnet(input_dim, reg_l2):
    """
    Create the dragonnet which has three heads.
    :param input_dim:
    :param reg_l2:
    :return:
    """
    inputs = keras.Input(shape=(input_dim,), name="input")

    # representation
    x = layers.Dense(
        units=200, activation="elu", kernel_initializer="RandomNormal"
    )(inputs)
    x = layers.Dense(
        units=200, activation="elu", kernel_initializer="RandomNormal"
    )(x)
    x = layers.Dense(
        units=200, activation="elu", kernel_initializer="RandomNormal"
    )(x)

    t_predictions = layers.Dense(units=1, activation="sigmoid")(x)

    # two heads for two hypotheses
    y0_hidden = layers.Dense(
        units=100, activation="elu", kernel_regularizer=regularizers.L2(reg_l2)
    )(x)
    y1_hidden = layers.Dense(
        units=100, activation="elu", kernel_regularizer=regularizers.L2(reg_l2)
    )(x)

    # second layer
    y0_hidden = layers.Dense(
        units=100, activation="elu", kernel_regularizer=regularizers.L2(reg_l2)
    )(y0_hidden)
    y1_hidden = layers.Dense(
        units=100, activation="elu", kernel_regularizer=regularizers.L2(reg_l2)
    )(y1_hidden)

    # third layer
    y0_predictions = layers.Dense(
        units=1, kernel_regularizer=regularizers.L2(reg_l2), name="y0_pred"
    )(y0_hidden)
    y1_predictions = layers.Dense(
        units=1, kernel_regularizer=regularizers.L2(reg_l2), name="y1_pred"
    )(y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name="epsilon")
    concat_pred = layers.Concatenate(axis=1)(
        [y0_predictions, y1_predictions, t_predictions, epsilons]
    )
    model = keras.Model(inputs=inputs, outputs=concat_pred, name="dragonnet")
    return model


def make_tarnet(input_dim, reg_l2):
    """
    Create the TARNet which has three heads.
    :param input_dim:
    :param reg_l2:
    :return:
    """
    inputs = keras.Input(shape=(input_dim,), name="input")

    # representation
    x = layers.Dense(
        units=200, activation="elu", kernel_initializer="RandomNormal"
    )(inputs)
    x = layers.Dense(
        units=200, activation="elu", kernel_initializer="RandomNormal"
    )(x)
    x = layers.Dense(
        units=200, activation="elu", kernel_initializer="RandomNormal"
    )(x)

    # different from Dragonnet, here TARNet directly uses inputs to predict t
    t_predictions = layers.Dense(units=1, activation="sigmoid")(inputs)

    # two heads for two hypotheses
    y0_hidden = layers.Dense(
        units=100, activation="elu", kernel_regularizer=regularizers.L2(reg_l2)
    )(x)
    y1_hidden = layers.Dense(
        units=100, activation="elu", kernel_regularizer=regularizers.L2(reg_l2)
    )(x)

    # second layer
    y0_hidden = layers.Dense(
        units=100, activation="elu", kernel_regularizer=regularizers.L2(reg_l2)
    )(y0_hidden)
    y1_hidden = layers.Dense(
        units=100, activation="elu", kernel_regularizer=regularizers.L2(reg_l2)
    )(y1_hidden)

    # third layer
    y0_predictions = layers.Dense(
        units=1, kernel_regularizer=regularizers.L2(reg_l2), name="y0_pred"
    )(y0_hidden)
    y1_predictions = layers.Dense(
        units=1, kernel_regularizer=regularizers.L2(reg_l2), name="y1_pred"
    )(y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name="epsilon")
    concat_pred = layers.Concatenate(axis=1)(
        [y0_predictions, y1_predictions, t_predictions, epsilons]
    )
    model = keras.Model(inputs=inputs, outputs=concat_pred, name="tarnet")
    return model


def _split_output(yt_hat, t, y_scaler, type_of_output="in"):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1))
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1))
    g = yt_hat[:, 2].copy()
    if type_of_output == "in":
        print(
            f"average propensity for "
            f"treated: {g[t.squeeze() == 1.].mean()} "
            f"and untreated: {g[t.squeeze() == 0.].mean()}."
        )
    return q_t0.squeeze(), q_t1.squeeze()


def train_and_estimate_Dragonnet(x, t, y_unscaled, x_test, estimator_name, hparams, logdir=None):
    """
    Training using Drogonnet.
    :param x: covariates
    :param t: treatment
    :param yf: factual outcomes
    :param x_test: out-of-sample covariates
    :return:
    """

    reg_l2 = hparams[HP_REG_L2]
    batch_size = hparams[HP_BATCH_SIZE]

    # 6 cont. features + 19 discrete features => 19 discrete + 6 cont. features
    num_features = x.shape[1]
    if num_features == 25:  # IHDP
        perm = np.arange(6, 25).tolist() + np.arange(6).tolist()
        x = x[:, perm]
        x_test = x_test[:, perm]
    t = t[:, None]  # add a dummy axis (from original dragonnet code)
    y_unscaled = y_unscaled[:, None]

    # y_scaler = StandardScaler().fit(y_unscaled)  # from dragonnet code
    # y = y_scaler.transform(y_unscaled) # normalize y from original code
    y = y_unscaled
    if estimator_name == "Dragonnet":
        model = make_dragonnet(x.shape[-1], reg_l2)
    elif estimator_name == "TARNet":
        model = make_tarnet(x.shape[-1], reg_l2)

    metrics = [
        losses.regression_loss,
        losses.binary_classification_loss,
        losses.treatment_accuracy,
        losses.track_epsilon,
    ]

    loss = losses.make_tarreg_loss(
        ratio=1.0, dragonnet_loss=losses.dragonnet_loss_binarycross
    )

    yt_train = np.concatenate([y, t], 1)
    start_time = time.time()

    # # from original dragonnet code (very wierd to train twice as the second
    # # part will overwrite the first part)
    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    #     loss=loss,
    #     metrics=metrics,
    # )
    #
    # verbose = 0
    # adam_callbacks = [
    #     keras.callbacks.TerminateOnNaN(),
    #     keras.callbacks.EarlyStopping(
    #         monitor="val_loss", patience=2, min_delta=0.0
    #     ),
    #     keras.callbacks.ReduceLROnPlateau(
    #         monitor="loss",
    #         factor=0.5,
    #         patience=5,
    #         verbose=verbose,
    #         mode="auto",
    #         min_delta=1e-8,
    #         cooldown=0,
    #         min_lr=0,
    #     ),
    # ]
    #
    # model.fit(
    #     x,
    #     yt_train,
    #     callbacks=adam_callbacks,
    #     validation_split=0.2,
    #     epochs=100,
    #     batch_size=64,
    #     verbose=verbose,
    # )

    verbose = 0
    sgd_callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=40, min_delta=0.0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=5,
            verbose=verbose,
            mode="auto",
            min_delta=0.0,
            cooldown=0,
            min_lr=0,
        ),
    ]

    if logdir is not None:
        sgd_callbacks.append(
            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        )
        sgd_callbacks.append(
            hp.KerasCallback(logdir, hparams),  # log hparams
        )

    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=hparams[HP_LEARNING_RATE], momentum=0.9, nesterov=True
        ),
        loss=loss,
        metrics=metrics,
    )

    # split x, yt_train into train and validation
    x_train, x_val, yt_train, yt_val = train_test_split(
        x, yt_train, test_size=0.2, random_state=42
    )

    model.fit(
        x_train,
        yt_train,
        callbacks=sgd_callbacks,
        validation_split=0.2,
        epochs=300,
        batch_size=batch_size,
        verbose=verbose,
    )

    res = model.evaluate(x_val, yt_val, verbose=0)
    # print("res: ", res)
    # print(model.metrics_names)
    tf.summary.scalar(METRIC_LOSS, res[0], step=1)

    elapsed_time = time.time() - start_time
    text = f"Elapsed_time is: {elapsed_time}"
    print(f"{text:-^79}")

    yt_hat_test = model.predict(x_test)
    yt_hat_train = model.predict(x)
    # y0_in, y1_in = _split_output(yt_hat_train, t, y_scaler, 'in')
    # y0_out, y1_out = _split_output(yt_hat_test, None, y_scaler, 'out')
    y0_in, y1_in = yt_hat_train[:, 0], yt_hat_train[:, 1]
    ate_in = np.mean(y1_in - y0_in)
    y0_out, y1_out = yt_hat_test[:, 0], yt_hat_test[:, 1]
    ate_out = np.mean(y1_out - y0_out)

    # propensity_in = yt_hat_train[:, 2]
    # propensity_out = yt_hat_test[:, 2]

    tf.keras.backend.clear_session()
    # return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out, propensity_in, propensity_out
    return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out


def train_and_evaluate(x, t, yf, x_test, estimator_name, tune_hparams=False):
    if estimator_name == "OLS1":
        return train_and_estimate_OLS1(x, t, yf, x_test)
    elif estimator_name == "OLS2":
        return train_and_estimate_OLS2(x, t, yf, x_test)
    elif estimator_name == "NN1":
        return train_and_estimate_NN1(x, t, yf, x_test)
    elif estimator_name == "NN2":
        return train_and_estimate_NN2(x, t, yf, x_test)
    elif estimator_name == "RF1":
        return train_and_estimate_RF1(x, t, yf, x_test)
    elif estimator_name == "RF2":
        return train_and_estimate_RF2(x, t, yf, x_test)
    elif estimator_name == "IPW":
        return train_and_estimate_IPW(x, t, yf, x_test)
    elif estimator_name == "DML":
        return train_and_estimate_DML(x, t, yf, x_test)
    elif estimator_name == "Dragonnet":
        # tune hyperparameters, output is not saved
        if tune_hparams:
            hparam_tune(x, t, yf, x_test, estimator_name)
        
        # no hparam tuning, hard code the hyperparameters
        # values chosen here were based on hparam tuning on jobs dataset
        y0_in, y1_in, ate_in, y0_out, y1_out, ate_out = train_and_estimate_Dragonnet(
                    x, t, yf, x_test, estimator_name="Dragonnet", hparams={
                        HP_LEARNING_RATE: 0.001,
                        HP_REG_L2: 0.01,
                        HP_BATCH_SIZE: 16,
                    }
                )

        return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out
    elif estimator_name == "TARNet":
        # tune hyperparameters, output is not saved
        if tune_hparams:
            hparam_tune(x, t, yf, x_test, estimator_name)

        # no hparam tuning, hard code the hyperparameters
        # values chosen here were based on hparam tuning on jobs dataset
        return train_and_estimate_Dragonnet(
            x, t, yf, x_test, estimator_name="TARNet", hparams={
                HP_LEARNING_RATE: 0.0001,
                HP_REG_L2: 0.01,
                HP_BATCH_SIZE: 64,
            }
            
        )
    else:
        text = (
            f"The estimator {estimator_name} is not implemented. "
            f"Please try another estimator."
        )
        raise Exception(text)

def hparam_tune(x, t, yf, x_test, estimator_name):
    for k, reg_l2 in enumerate(HP_REG_L2.domain.values):
        for j, batch_size in enumerate(HP_BATCH_SIZE.domain.values):
            for i, learning_rate in enumerate(HP_LEARNING_RATE.domain.values):
                hparams = {
                        HP_LEARNING_RATE: learning_rate,
                        HP_REG_L2: reg_l2,
                        HP_BATCH_SIZE: batch_size,
                    }
                logdir = f"logs/{estimator_name}/hparam_tuning/lr_{learning_rate}/reg_l2_{reg_l2}/batch_size_{batch_size}"
                with tf.summary.create_file_writer(logdir).as_default():
                    hp.hparams_config(
                            hparams=[HP_LEARNING_RATE, HP_REG_L2, HP_BATCH_SIZE],
                            metrics=[hp.Metric(METRIC_LOSS, display_name='Validation Loss')],
                        )
                
                    print(f"Starting training with params: lr: {learning_rate}, reg_l2: {reg_l2}, bs: {batch_size}")
                    hp.hparams(hparams)  # record the values used in this trial
                    train_and_estimate_Dragonnet(
                        x, t, yf, x_test, estimator_name=estimator_name, hparams=hparams, logdir=logdir
                    )
            