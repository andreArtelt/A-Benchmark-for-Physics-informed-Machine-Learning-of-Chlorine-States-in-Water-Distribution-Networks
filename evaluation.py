from typing import Callable
import numpy as np
from epyt_flow.metrics import running_mse as epytflow_running_mse
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error



def evaluate_transport_delay(y_pred: np.ndarray, y_true: np.ndarray,
                             metric: Callable[[np.ndarray, np.ndarray], float],
                             nodes_id: list[str], transport_delay_per_node: dict) -> dict:
    """
    Evaluates the prediction for an entire network (i.e. every node in the network) under some
    given evaluation metric and group the results by the transport delays of the nodes.

    Parameters
    ----------
    y_pred : `numpy.ndarray`
        Predicted outputs -- must be a two-dimensional array where the first axis corresponds
        to a node and the second axis encodes time.
    y_true : `numpy.ndarray`
        Ground truth outputs -- must be a two-dimensional array where the first axis corresponds
        to a node and the second axis encodes time.
    metric : `Callable[[np.ndarray, np.ndarray], float]`
        A callable function that computes some metric for a single node prediction --
        i.e. two 1d arrays are compared.
    nodes_id : `list[str]`
        List of nodes ID -- ordering must be the same as in 'y_pred' and 'y_true'.
    transport_delay_per_node : `dict`
        Dictionary mapping a node ID to the corresponding transport delay.

    Returns
    -------
    `dict`
        Maps transport delays to scores (i.e. results of the evaluation metric).
    """
    r = {}

    for idx, node_id in enumerate(nodes_id):
        score = metric(y_pred[idx, :], y_true[idx, :])
        t_delay = transport_delay_per_node[node_id]

        if t_delay not in r:
            r[t_delay] = []
        r[t_delay].append(score)

    return r


class Evaluator:
    """
    Class for evaluating the predictions for a single node.
    """
    @staticmethod
    def evaluate_predictions(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Computes and returns all evaluation metrics.

        Parameters
        ----------
        y_pred : `numpy.ndarray`
            Predicted outputs.
        y_true : `numpy.ndarray`
            Ground truth outputs.

        Returns
        -------
        `dict`
            All metrics.
        """
        return {"y_true": y_true, "y_pred": y_pred,
                "MSE": Evaluator.mean_squared_error(y_pred, y_true),
                "MAE": Evaluator.mean_absolute_error(y_pred, y_true),
                "RunningMSE": Evaluator.running_mse(y_pred, y_true),
                "RunningMAE": Evaluator.running_mae(y_pred, y_true)
                }

    @staticmethod
    def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes and returns the mean-squared-error (MSE).

        Parameters
        ----------
        y_pred : `numpy.ndarray`
            Predicted outputs.
        y_true : `numpy.ndarray`
            Ground truth outputs.

        Returns
        -------
        `float`
            The mean-squared-error.
        """
        if len(y_true.shape) > 1:
            return root_mean_squared_error(y_true, y_pred, multioutput='raw_values')**2
        else:
            return root_mean_squared_error(y_true, y_pred)**2

    @staticmethod
    def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes and returns the Mean-Absolute error.

        Parameters
        ----------
        y_pred : `numpy.ndarray`
            Predicted outputs.
        y_true : `numpy.ndarray`
            Ground truth outputs.

        Returns
        -------
        `float`
            The mean absolute error.
        """
        if len(y_true.shape) > 1:
            return mean_absolute_error(y_true, y_pred, multioutput="raw_values")
        else:
            return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def running_mse_(y_pred: np.ndarray, y_true: np.ndarray) -> list[float]:
        """
        Computes and returns the running mean-squared-error --
        i.e. the MSE for every point in time.

        Parameters
        ----------
        y_pred : `numpy.ndarray`
            Predicted outputs.
        y_true : `numpy.ndarray`
            Ground truth outputs.

        Returns
        -------
        `list[float]`
            The running mean-squared-error.
        """
        if len(y_true.shape) > 1:
            return [epytflow_running_mse(y_pred[i], y_true[i]) for i in range(y_true.shape[0])]
        else:
            return epytflow_running_mse(y_pred, y_true)

    @staticmethod
    def mean_absolued_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes and returns the Mean-Absolute error.

        Parameters
        ----------
        y_pred : `numpy.ndarray`
            Predicted outputs.
        y_true : `numpy.ndarray`
            Ground truth outputs.

        Returns
        -------
        `float`
            The mean absolute error.
        """
        if len(y_true.shape) > 1:
            return mean_absolute_error(y_true, y_pred, multioutput="raw_values")
        else:
            return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def running_mae_(y_pred: np.ndarray, y_true: np.ndarray) -> list[float]:
        """
        Computes and returns the running mean-absolute-error --
        i.e. the MAE for every point in time.

        Parameters
        ----------
        y_pred : `numpy.ndarray`
            Predicted outputs.
        y_true : `numpy.ndarray`
            Ground truth outputs.

        Returns
        -------
        `list[float]`
            The running mean-absolute-error.
        """
        def my_running_mae(y_pred_, y_) -> list[float]:
            r = []

            for t in range(2, len(y_pred_)):
                r.append(mean_absolute_error(y_[:t], y_pred_[:t]))

            return r

        if len(y_true.shape) > 1:
            return [my_running_mae(y_pred[i], y_true[i]) for i in range(y_true.shape[0])]
        else:
            return my_running_mae(y_pred, y_true)

    '''
    (Luca): Putting this here, running_average_metric is a fast version of any running metric
    By default running_mae and running_mse use the last axis, i.e. shape should be [..., time],
    or axis has to be something else. 
    '''
    @staticmethod
    def running_average_metric(values, axis=0):
        return np.swapaxes(np.swapaxes(
            np.add.accumulate(values, axis=axis), axis1=axis, axis2=-1) 
            / np.arange(1, values.shape[axis] + 1), axis1=axis, axis2=-1
        )
    @staticmethod
    def running_mae(y_pred, y_true, axis=-1):
        values = np.abs(y_true - y_pred)
        return Evaluator.running_average_metric(values, axis=axis)
    @staticmethod
    def running_mse(y_pred, y_true, axis=-1):
        values = np.square(y_true - y_pred)
        return Evaluator.running_average_metric(values, axis=axis)