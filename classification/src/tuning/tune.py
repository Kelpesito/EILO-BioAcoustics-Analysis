"""
tune.py

Module for hyperparameter tuning
"""


from pathlib import Path

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.study import Study
from optuna.trial import Trial

from src.models.cnn2d import build
from src.training.train import fit
from src.utils.seed import set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 120
N_TRIALS = 100

CLASSIFICATION_PATH = Path("classification")
RESULTS_PATH = CLASSIFICATION_PATH / "results"
RESULTS_OPTUNA_PATH = RESULTS_PATH / "optuna"
TENSORBOARD_OPTUNA_PATH = RESULTS_PATH / "tensorboard" / "optuna"


def create_folder_structure(study_name: str) -> None:
    """
    Create the following folders:
    - classification/results
    - classification/results/optuna
    - classification/results/{study_name}

    Parameters
    ----------
    study_name: str
        Study names
    """
    RESULTS_PATH.mkdir(exist_ok=True)
    RESULTS_OPTUNA_PATH.mkdir(exist_ok=True)
    (RESULTS_OPTUNA_PATH / study_name).mkdir(exist_ok=True)


def log_hparams(writer: SummaryWriter, params: dict, score: float, state: str) -> None:
    """
    Writes the hyperparameters of a trial and its score to TensorBoard (HPARAMS tab), in the same
    run as the trial's training curves.

    Parameters
    ----------
    writer: SummaryWriter
        TensorBoard writer of the trial
    params: dict
        Hyperparameters of the trial
    score: float
        Best smoothed monitored score reached by the trial
    state: str
        Final state of the trial: "complete" or "pruned"
    """
    hparams = {k: (v if v is not None else "None") for k, v in params.items()}
    hparams["state"] = state
    writer.add_hparams(hparams, {"hparam/best_score_ema": score}, run_name=".")


def objective(
    trial: Trial,
    df: pd.DataFrame,
    df_config: dict[str, str],
    origin: str,
    class_to_idx: dict[str, int],
    fold: int = 1,
    in_channels: int = 1,
    num_classes: int = 7,
    max_epochs: int = MAX_EPOCHS,
    device: torch.device = DEVICE,
) -> float:
    """
    Objective function for hyperparameter tuning:
        1. Select hyperparameters
        2. Build the model
        3. Train the model
        4. Get objective variable: best EMA of validation macro PR-AUC (the same smoothed score
           that drives ReduceLROnPlateau, EarlyStopping and pruning in `fit`)

    The training curves and the hyperparameters of the trial (also if pruned) are logged to
    TensorBoard in classification/results/tensorboard/optuna/{study_name}/trial_{number}.

    Hyperparameters:
    - **Model hyperparameters:**
        - depth: int [2, 5]
        - base_filters: {8, 16, 32, 64}
        - alpha_leaky_relu: float [0.001, 0.3] log
        - embedding_dim: {32, 64, 128, 256, 512, 1024}
        - hidden_dim: {32, 64, 128, 256, 512, 1024}
        - dropout_cnn: float [0.0, 0.1]
        - dropout_fc: float [0.0, 0.4]

    - **Training hyperparameters:**
        - batch_size: {16, 32, 64}
        - lr_max: float [1e-4, 1e-2] log
        # - lr_min_ratio: float [3e-3, 1e-1] log
        # - num_epochs: int [20, 100]
        - weight_decay: float [5e-5, 5e-2] log
        - optimizer_name: "adamw" (fixed)
        - loss_name: "fl" (fixed)
        - gamma_focal: float [0.0, 3.0]
    
    Parameters
    ----------
    trial: Trial
        Trial information for hyperparameter tuning
    df: pd.DataFrame
        The DataFrame containing the data information
    df_config: dict[str, str]
        Dictionary with the name of the columns containing the image path names ("image_col"), the
        labels ("label_col") and the fold idx ("fold_col").
    origin: str
        Path with the input images
    class_to_idx: dict[str, int]
        Dictionary relating the original label to ordinal encoding label
    fold: int, optional
        Value of fold to analize in validation (default = 1)
    in_channels: int, optional
        Number of input channels (default = 1)
    num_classes: int, optional
        Number of output classes (default = 7)
    max_epochs: int, optional
        Number of epochs to train for so long (default = MAX_EPOCHS)
    device: torch.device, optional
        The device object ("cuda" or "cpu") (default = DEVICE)
    
    Returns
    -------
    float
        Objective variable: best EMA of validation macro PR-AUC
    """
    
    set_seed()
    # Model hyperparameters
    depth = trial.suggest_int("depth", 2, 5)
    base_filters = trial.suggest_categorical("base_filters", [8, 16, 32])
    alpha_leaky_relu = trial.suggest_float("alpha_leaky_relu", 0.001, 0.3, log=True)
    embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128, 256, 512, 1024])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512, 1024])
    dropout_cnn = trial.suggest_float("dropout_cnn", 0.0, 0.1)
    dropout_fc = trial.suggest_float("dropout_fc", 0.0, 0.4)
    
    # Training hyperparameters
    batch_size = 32
    lr_max = trial.suggest_float("lr_max", 5e-5, 1e-3, log=True)
    # lr_min_ratio = trial.suggest_float("lr_min_ratio", 3e-3, 1e-1, log=True)
    # num_epochs = trial.suggest_int("num_epochs", 20, 100)
    weight_decay = trial.suggest_float("weight_decay", 5e-5, 5e-2, log=True)
    optimizer_name = "adamw"
    loss_name = "fl"
    gamma_focal = trial.suggest_float("gamma_focal", 0.0, 4.0)
        
    params = {
        "depth": depth,
        "base_filters": base_filters,
        "alpha_leaky_relu": alpha_leaky_relu,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "dropout_cnn": dropout_cnn,
        "dropout_fc": dropout_fc,
        "batch_size": batch_size,
        "lr_max": lr_max,
        # "lr_min_ratio": lr_min_ratio,
        # "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "optimizer_name": optimizer_name,
        "loss_name": loss_name,
        "gamma_focal": gamma_focal if loss_name == "fl" else None,
    }    
    
        
    # Build the model
    model = build(
        depth=depth,
        base_filters=base_filters,
        alpha_leaky_relu=alpha_leaky_relu,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout_cnn=dropout_cnn,
        dropout_fc=dropout_fc,
        in_channels=in_channels,
        num_classes=num_classes,
    ).to(device)

    # Train the model
    log_dir = TENSORBOARD_OPTUNA_PATH / trial.study.study_name / f"trial_{trial.number}"
    with SummaryWriter(log_dir=log_dir) as writer:
        try:
            _, final_metrics = fit(
                model=model,
                df=df,
                origin=origin,
                fold=fold,
                df_config=df_config,
                params=params,
                num_classes=num_classes,
                class_to_idx=class_to_idx,
                writer=writer,
                max_epochs=max_epochs,
                device=device,
                trial=trial,
            )
        except optuna.TrialPruned:
            # Best smoothed score reached before pruning (stored by `fit`)
            log_hparams(writer, params, trial.user_attrs["best_score_ema"], "pruned")
            raise

        log_hparams(writer, params, final_metrics["best_score_ema"], "complete")

    return final_metrics["best_score_ema"]


def tune(
    df: pd.DataFrame,
    df_config: dict[str, str],
    origin: str,
    class_to_idx: dict[str, int],
    study_name: str,
    fold: int = 1,
    n_trials: int = N_TRIALS,
    in_channels: int = 1,
    num_classes: int = 7,
    max_epochs: int = MAX_EPOCHS,
    device: torch.device = DEVICE,
    save: bool = False,
) -> Study:
    """
    Creates and optimizes and Optuna's hyperparameter tuning study.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the data information
    df_config: dict[str, str]
        Dictionary with the name of the columns containing the image path names ("image_col"), the
        labels ("label_col") and the fold idx ("fold_col").
    origin: str
        Path with the input images
    class_to_idx: dict[str, int]
        Dictionary relating the original label to ordinal encoding label
    study_name: str
        Study names 
    fold: int, optional
        Value of fold to analize in validation (default = 1)
    n_trials: int, optional
        Number of trials in the study
    in_channels: int, optional
        Number of input channels (default = 1)
    num_classes: int, optional
        Number of output classes (default = 7)
    max_epochs: int, optional
        Number of epochs to train for so long (default = MAX_EPOCHS)
    device: torch.device, optional
        The device object ("cuda" or "cpu") (default = DEVICE)
    save: bool, optional
        Whether to save the study or not (default = False)
    """
    if save:
        create_folder_structure(study_name)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=10,
            max_resource=max_epochs,
            reduction_factor=3,
        ),
        storage=f"sqlite:///{RESULTS_OPTUNA_PATH / study_name / study_name}.db" if save else None,
        load_if_exists=True,
    )
    
    study.optimize(
        lambda trial: objective(
            trial=trial,
            df=df,
            df_config=df_config,
            origin=origin,
            class_to_idx=class_to_idx,
            fold=fold,
            in_channels=in_channels,
            num_classes=num_classes,
            max_epochs=max_epochs,
            device=device,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    print("Best score (EMA of macro PR-AUC):", study.best_value)
    print("Best params:")
    for param, value in study.best_params.items():
        print(f"{param}: {value}")

    return study
