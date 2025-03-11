from typing import Callable
import numpy as np
from sklearn.metrics import mean_absolute_error



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


# Maximum tranport delay (i.e. length/size of the system's memory) for each of three three networks
memory_length = {
    "Net1": 236,
    "Hanoi": 32,
    "CY-DBP": 50
}


class Evaluator:
    """
    Class for evaluating the predictions for a single node.
    """
    @staticmethod
    def evaluate_predictions(y_pred: np.ndarray, y_true: np.ndarray,
                             cl_injection: np.ndarray) -> dict:
        """
        Computes and returns all evaluation metrics.

        Parameters
        ----------
        y_pred : `numpy.ndarray`
            Predicted outputs.
        y_true : `numpy.ndarray`
            Ground truth outputs.
        cl_injection : `numpy.ndarray`
            Chlorine concentration at the injection node.

        Returns
        -------
        `dict`
            All metrics.
        """
        return {"y_true": y_true, "y_pred": y_pred,
                "cl_injection": cl_injection,
                "lower_bound": Evaluator.eval_lowerbound(y_pred),
                "overunderestimates": Evaluator.eval_overunderestimate(y_pred, y_true),
                "MAE": Evaluator.mean_absolute_error(y_pred, y_true),
                "RunningMAE": Evaluator.running_mae(y_pred, y_true),
                }

    @staticmethod
    def eval_overunderestimate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the percentage of times where the prediction overshoots the true value.

        Parameters
        ----------
        y_pred : `numpy.ndarray`
            Predicted outputs.
        y_true : `numpy.ndarray`
            Ground truth outputs.

        Returns
        -------
        `float`
            Percentage of overshoots.
        """
        def ReLU(x):
            return x * (x > 0)

        if len(y_true.shape) > 1:
            return [np.sum(ReLU(y_pred[i, :] - y_true[i, :])) -
                    np.sum(ReLU(-1 * (y_pred[i, :] - y_true[i, :])))
                    for i in range(y_true.shape[0])]
        else:
            overestimate = np.sum(ReLU(y_pred - y_true))
            underestimate = np.sum(ReLU(-1 * (y_pred - y_true)))
            return overestimate - underestimate

    @staticmethod
    def eval_lowerbound(y_pred: np.ndarray) -> float:
        """
        Computes the percentage of times where the prediction adhers to the lower bound --
        i.e. concentrations can not be negative!

        Parameters
        ----------
        y_pred : `numpy.ndarray`
            Predicted concetrations.
        y_true : `numpy.ndarray`
            Ground truth concetrations.

        Returns
        -------
        `float`
            Percentage of lower bound violations.
        """
        return np.sum(y_pred >= 0, axis=1) / len(y_pred)

    @staticmethod
    def eval_upperbound(y_pred: np.ndarray, cl_injection: np.ndarray, memory_length: int) -> list:
        """
        Computes the percentage of times where the prediction adhers to an upper bound
        based on the system's memory.

        Parameters
        ----------
        y_pred : `numpy.ndarray`
            Predicted concetrations.
        cl_injection : `numpy.ndarray`
            Chlorine concentration at the injection node.
        memory_length : `int`
            Size/Lengths of the WDS memory.

        Returns
        -------
        `list`
            True or False for each point in time.
        """
        def __r_eval(y_pred_, cl_injection_):
            r = []

            start_idx = int(memory_length)
            for i in range(start_idx, y_pred_.shape[0]):
                upper_bound = max(cl_injection_[i-memory_length:i])
                r.append(y_pred_[i] <= upper_bound)

            return r

        if len(y_pred.shape) > 1:
            return [__r_eval(y_pred[i, :], cl_injection[i, :]) for i in range(y_pred.shape[0])]
        else:
            return __r_eval(y_pred, cl_injection)

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
            return [mean_absolute_error(y_true[i, :], y_pred[i, :]) for i in range(y_true.shape[0])]
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

    @staticmethod
    def running_average_metric(values, axis=0):
        return np.swapaxes(np.swapaxes(
            np.add.accumulate(values, axis=axis), axis1=axis, axis2=-1) 
            / np.arange(1, values.shape[-1] + 1), axis1=axis, axis2=-1
        )

    @staticmethod
    def running_mae(y_pred, y_true, axis=-1):
        values = np.abs(y_true - y_pred)
        return Evaluator.running_average_metric(values, axis=axis)

    @staticmethod
    def running_mse(y_pred, y_true, axis=-1):
        values = np.square(y_true - y_pred)
        return Evaluator.running_average_metric(values, axis=axis)
