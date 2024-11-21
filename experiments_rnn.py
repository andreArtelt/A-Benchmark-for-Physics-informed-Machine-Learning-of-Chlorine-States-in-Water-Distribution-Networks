import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

from data_loader import DataLoader
from evaluation import Evaluator


class RecurrentNeuralNetworkModel():
    def __init__(self, input_shape: list[int], output_size: int = 1,
                 hidden_layer_sizes: list[int] = [128, 64],
                 activation: str = 'tanh'):
        self.model = tf.keras.Sequential(
                [tf.keras.layers.InputLayer(shape=input_shape)] +
                [tf.keras.layers.LSTM(hidden_layer_sizes[0], activation=activation,
                                      return_sequences=True)] +
                [tf.keras.layers.LSTM(ls, activation=activation, return_sequences=True)
                 for ls in hidden_layer_sizes[1:]] +
                [tf.keras.layers.Dense(output_size, activation="relu")])
        self.solver = "adam"

    def save(self, f_out: str) -> None:
        self.model.save(f_out)

    def load(self, f_in: str) -> None:
        self.model = tf.keras.models.load_model(f_in)

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 500, callbacks: list = [],
            val: tuple[np.ndarray, np.ndarray] = None) -> None:
        self.model.compile(optimizer=self.solver,
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=["mse"])
        self.model.fit(X, y, epochs=n_epochs, verbose=True, callbacks=callbacks,
                       validation_data=val, shuffle=True)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model(X, training=False).numpy().reshape(X.shape[0], -1)



def train_model(net_desc: str, target_node_id: str, data_configs: list[dict],
                path_to_data: str = "data", dir_out: str = "results") -> None:
    """
    TODO
    """
    X_train, y_train = [], []
    X_val, y_val = [], []
    test_data = {}

    d = DataLoader(path_to_data)
    for d_config in data_configs:
        train, val, test = d.load_data(train_size=700, val_size=100, net_desc=net_desc,
                                       **d_config,
                                       shuffle=True, target_node_id=target_node_id)

        X_train.append(train[0]);y_train.append(train[1])
        X_val.append(val[0]);y_val.append(val[1])
        test_data[f"{d_config['cl_injection_pattern_desc']}-rand_demands={d_config['random_demands']}"] = test

    X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
    X_val, y_val = np.concatenate(X_val), np.concatenate(y_val)

    # Pre-processing -- scaling the data
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    print(f"Training data: {X_train.shape, y_train.shape}")

    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    print(f"Validation data: {X_val.shape, y_val.shape}")

    for c_id in test_data.keys():
        X_test, y_test = test_data[c_id]
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        test_data[c_id] = X_test, y_test

    # Create and fit model
    model = RecurrentNeuralNetworkModel(input_shape=(X_train.shape[1], X_train.shape[2]))

    earlystopping_mgr = tf.keras.callbacks.EarlyStopping(monitor='val_mse', min_delta=0,
                                                         patience=10, verbose=0,
                                                         mode='min', baseline=None,
                                                         restore_best_weights=True,
                                                         start_from_epoch=0)
    model.fit(X_train, y_train, n_epochs=500, callbacks=[earlystopping_mgr],
              val=(X_val, y_val))

    f_out = os.path.join(dir_out, f"rnn_{net_desc}_node{target_node_id}.keras")
    Path(str(Path(f_out).parent)).mkdir(parents=True, exist_ok=True)
    model.save(f_out)

    f_out = os.path.join(dir_out, f"scaler_{net_desc}_node{target_node_id}.bin")
    Path(str(Path(f_out).parent)).mkdir(parents=True, exist_ok=True)
    dump(scaler, f_out, compress=True)

    # Evaluate on test data
    eval_results = {}
    for c_id, (X_test, y_test) in test_data.items():
        y_test_pred = model.predict(X_test)

        # Evaluate predictions
        eval_results[c_id] = Evaluator.evaluate_predictions(y_test_pred, y_test)

    f_out = os.path.join(dir_out, f"eval-test_{net_desc}_node{target_node_id}.bin")

    Path(str(Path(f_out).parent)).mkdir(parents=True, exist_ok=True)
    dump(eval_results, f_out, compress=True)


def eval_model_on_data_config(net_desc: str, target_node_id: str, data_configs: list[dict],
                              f_out: str, path_to_data: str = "data", dir_in: str = "results"
                              ) -> None:
    """
    TODO
    """
    test_data = {}

    scaler = load(os.path.join(dir_in, f"scaler_{net_desc}_node{target_node_id}.bin"))

    d = DataLoader(path_to_data)
    for d_config in data_configs:
        train, val, test = d.load_data(train_size=700, val_size=100, net_desc=net_desc,
                                       **d_config,
                                       shuffle=True, target_node_id=target_node_id)

        X = np.concatenate((train[0], val[0], test[0]))
        y = np.concatenate((train[1], val[1], test[1]))
        X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        test_data[f"{d_config['cl_injection_pattern_desc']}-rand_demands={d_config['random_demands']}"] = X, y

    eval_results = {}
    model = None
    for c_id, (X, y) in test_data.items():
        if model is None:
            model = RecurrentNeuralNetworkModel(input_shape=(X.shape[1], X.shape[2]))
            model.load(os.path.join(dir_in, f"rnn_{net_desc}_node{target_node_id}.keras"))

        y_pred = model.predict(X)

        # Evaluate predictions
        eval_results[c_id] = Evaluator.evaluate_predictions(y_pred, y)

    Path(str(Path(f_out).parent)).mkdir(parents=True, exist_ok=True)
    dump(eval_results, f_out, compress=True)
