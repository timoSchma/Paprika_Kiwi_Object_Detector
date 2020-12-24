from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.utils import split_dict
from src.model import TinyYoloV3
import keras.backend as K
import numpy as np


def grid_search(annotations_dict, image_directory, param_grid, cv, **kwargs):
    """
    Perform a grid search on the given data with the given parameter grid.

    Parameters
    ----------
    annotations_dict : dict
        The annotation dict describing the input data.
    image_directory : path-like
        The path to the directory containing the image data.
    param_grid : dict
        A dict containing lists of parameters to the YoloEstimator.
    cv : Any
        The cross-validation parameter to sklearn.model_selection.GridSearchCV
    kwargs : Any
        Any further parameters to the GridSearchCV.

    Returns
    -------
    search : GridSearchCV
        The fitted search object.
    """
    YoloEstimator.annotations = annotations_dict
    YoloEstimator.image_directory = image_directory

    X = np.arange(len(YoloEstimator.annotations))
    y = np.zeros(len(YoloEstimator.annotations))

    search = GridSearchCV(YoloEstimator(), param_grid, cv=cv, refit=False, **kwargs)
    search.fit(X, y)

    return search


class YoloEstimator(BaseEstimator):
    """
    A sklearn-estimator wrapper for the tiny-yolo v3 training.
    The training consists of two steps, first training only the last few layers of the frozen model,
    followed by unfreezing and fine-tuning all layers.

    The expected input data to the training and scoring is indices to the annotation dict for X. y is unused.
    Please set the annotations and image_directory class variables before usage.
    """

    # set this before the grid search
    annotations = None
    image_directory = None

    def __init__(self, freeze_learning_rate=1e-2, freeze_batch_size=32, freeze_epochs=1,
                 freeze_jittering_params=(0.5, 0.5, 0.1, 1.5, 1.5), fine_tuning_learning_rate=1e-4,
                 fine_tuning_batch_size=32, fine_tuning_epochs=1,
                 fine_tuning_jittering_params=(0.5, 0.5, 0.1, 1.5, 1.5)):
        """
        Create a new estimator with the given parameters.

        Parameters
        ----------
        freeze_learning_rate : double
            The learning rate for the first training step using a frozen model.
        freeze_batch_size : int
            The batch size for the first training step using a frozen model.
        freeze_epochs : int
            The number of epochs for the first training step using a frozen model.
        freeze_jittering_params : tuple
            The jittering parameters for the first training step using a frozen model.
            The following format is expected:
            (rotation probability, jittering probability, hue range, saturation range, value range)
        fine_tuning_learning_rate : double
            The learning rate for the second training step fine-tuning the model.
        fine_tuning_batch_size : int
            The batch size for the second training step fine-tuning the model.
        fine_tuning_epochs : int
            The number of epochs for the second training step fine-tuning the model.
        fine_tuning_jittering_params : tuple
            The jittering parameters for the second training step fine-tuning the model.
            The following format is expected:
            (rotation probability, jittering probability, hue range, saturation range, value range)
        """
        self.freeze_learning_rate = freeze_learning_rate
        self.freeze_batch_size = freeze_batch_size
        self.freeze_epochs = freeze_epochs
        self.freeze_jittering_params = freeze_jittering_params
        self.fine_tuning_learning_rate = fine_tuning_learning_rate
        self.fine_tuning_batch_size = fine_tuning_batch_size
        self.fine_tuning_epochs = fine_tuning_epochs
        self.fine_tuning_jittering_params = fine_tuning_jittering_params

    def get_data(self, X):
        """
        Constructs the input data
        Parameters
        ----------
        X : np.array of int
            An array of indices in [0, len(cls.annotations)] used to select the training data. All other indices are
            used as validation data.

        Returns
        -------
        List[dict]
            A list containing two annotation dicts, one for the training and one for the validation data.
        """
        if len(X.shape) == 2:  # work-around for check_estimator
            X = np.random.choice(np.arange(len(self.annotations)), size=int(len(self.annotations) * 0.8), replace=False)
        X = X.tolist()
        val_X = list(set(range(len(self.annotations))) - set(X))
        return split_dict(self.annotations, index_lists=[X, val_X])

    def get_train_params(self, step, initial_epoch=0):
        """
        Converts this estimators parameters to arguments to the model.train function.

        Parameters
        ----------
        step : string
            The current training step ("freeze" or "fine-tuning")
        initial_epoch : int
            The epoch this training step starts from (0 for freeze, freeze_epochs for fine-tuning.

        Returns
        -------
        dict
            The named arguments to the train function.
        """
        params = self.get_params()
        train_params = {key: params[f"{step}_{key}"] for key in ["learning_rate", "batch_size", "epochs"]}
        train_params["epochs"] += initial_epoch
        jittering_params = params[f"{step}_jittering_params"]
        jittering_probas = dict(zip(["rotation_probability", "jittering_probability"], jittering_params[0:2]))
        jittering_range = dict(zip(["hue", "sat", "val"], jittering_params[2:]))
        jittering_params = {**jittering_probas, "jittering_range": jittering_range}
        return {**train_params, "initial_epoch": initial_epoch, "jittering_params": jittering_params}

    def fit(self, X, y):
        """
        Fits this estimator using the given data.

        Parameters
        ----------
        X : np.array of int
            An array of indices in [0, len(cls.annotations)] used to select the training data. All other indices are
            used as validation data.
        y : Any
            Unused, the labels are extracted from the annotations dict set in the annotations class variable.

        Returns
        -------
        self : YoloEstimator
            This estimator.
        """
        # Set Keras to learning-mode --> fix constantly adapting batch-normalizations
        K.set_learning_phase(1)

        # Setup Model
        self.model = TinyYoloV3()
        self.model.replace_output_layers()
        self.model.training_mode()

        train_data, val_data = self.get_data(X)

        params_freeze = self.get_train_params("freeze", initial_epoch=0)
        params_fine_tuning = self.get_train_params("fine_tuning", initial_epoch=params_freeze["epochs"])

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                       patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7)
        callbacks = [early_stopping, reduce_lr]

        # 1. Freeze Training
        self.model.freeze_all_but_output()
        history_freeze = self.model.train(train_data, val_data, self.image_directory, callbacks=callbacks,
                                          **params_freeze, verbose=1)

        # 2. Fine-Tuning
        self.model.unfreeze()
        history_fine_tuning = self.model.train(train_data, val_data, self.image_directory, callbacks=callbacks,
                                               **params_fine_tuning, verbose=1)

        self.history = [history_freeze, history_fine_tuning]
        return self

    def score(self, X, y):
        """
        Scores this model on the given data. Computes and prints the loss on the given data. Returns the negative loss
        for easier integration with the grid search.

        Parameters
        ----------
        X : np.array of int
            An array of indices in [0, len(cls.annotations)] used to select the training data. All other indices are
            used as validation data.
        y : Any
            Unused, the labels are extracted from the annotations dict set in the annotations class variable.

        Returns
        -------
        -score : double
            The score (= negative loss) on the given data.
        """
        data, _ = self.get_data(X)
        score = self.model.evaluate(data, self.image_directory)
        print(f"Score {score} with params: {self.get_params()}")
        return -score
