"""
train.py

Module with the utilities to perform a training of a model or a cross-validation analysis
"""


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import optuna
from optuna.trial import Trial

from src.data.dataloaders import get_dataloaders
from src.models.cnn2d import build
from src.utils.build_loss import build_loss_function
from src.utils.build_optimizer import build_optimizer
from src.utils.build_scheduler import build_scheduler
from src.utils.seed import set_seed

from .early_stopping import EarlyStopping


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WARMUP_EPOCHS = 5
MAX_EPOCHS = 120
IMG_SIZE = 224

CLASSIFICATION_PATH = Path("classification")
RESULTS_PATH = CLASSIFICATION_PATH / "results"
RESULTS_CV_PATH = RESULTS_PATH / "cv"
MODELS_PATH = CLASSIFICATION_PATH / "models"
MODELS_CV_PATH = MODELS_PATH / "cv"


def create_folder_structure(model_name: str) -> None:
    """
    Creates the following folders:
    - classification/results
    - classification/results/cv
    - classification/results/cv/{model_name}
    - classification/models
    - classification/models/cv
    - classification/models/cv/{model_name}

    Parameter
    ---------
    model_name: str
        Model name
    """
    RESULTS_PATH.mkdir(exist_ok=True)
    RESULTS_CV_PATH.mkdir(exist_ok=True)
    (RESULTS_CV_PATH / model_name).mkdir(exist_ok=True)
    
    MODELS_PATH.mkdir(exist_ok=True)
    MODELS_CV_PATH.mkdir(exist_ok=True)
    (MODELS_CV_PATH / model_name).mkdir(exist_ok=True)


def train_one_epoch(
        model: nn.Module, 
        loader: DataLoader, 
        criterion: nn.Module, 
        optimizer: Optimizer, 
        device: torch.device = DEVICE
    ) -> float:
    """
    Performs one training epoch. For each batch:
        1. Perform a forward pass -> logits
        2. Calculate the loss
        3. Perform backward pass -> gradients
        4. Perform a step of the optimizerr
        5. Compute the new loss
    
    Parameters
    ----------
    model: nn.Module
        The Deep Learning model to be trained
    loader: DataLoader
        The training DataLoader
    criterion: nn.Module
        The loss function
    optimizer: Optimizer
        The Optimizer algorithm
    device: torch.device, optional
        The device object ("cuda" or "cpu") (default = DEVICE)
    
    Returns
    -------
    float
        Reduced loss
    """
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    class_to_idx: dict[str, int],
    device: torch.device = DEVICE
) -> dict:
    """
    Evaluates a Deep Learning model. 
        1. For each batch:
            1. Perform a forward pass -> logits
            2. Calculate the loss
            3. Calculate the probabilities: logits -> probs
        2. Calculate the predictions: probs -> pred
        3. Calculate metrics
    
    Parameters
    ----------
    model: nn.Module
        The Deep Learning model to be evaluated
    loader: DataLoader
        The evaluation DataLoader
    criterion: nn.Module
        The loss function
    num_classes: int
        Number of output classes
    class_to_idx: dict[str, int]
        Dictionary relating the original label to ordinal encoding label
    device: torch.device, optional
        The device object ("cuda" or "cpu") (default = DEVICE)

    Returns
    -------
    metrics: dict
        Dictionary with:
        - loss: float
            Evaluation loss
        - balanced_accuracy: float
            Evaluation balanced accuracy
        - f1_macro: float
            Evaluation macro F1
        - f1_weighted: float
            Evaluation weighted F1
        - mcc: float
            Evaluation Mathew's Correlation Coefficient
        - pr_auc: float
            Evaluation macro PR-AUC
        - f1_per_class: dict
            Evaluation F1 score for each class
        - support_per_class: dict
            Support value for each class
    """
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = np.argmax(y_prob, axis=1)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    labels_order = [idx_to_class[i] for i in range(num_classes)]

    _, _, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "pr_auc": average_precision_score(np.eye(num_classes)[y_true], y_prob, average="macro"),
        "f1_per_class": dict(zip(labels_order, f1_per_class)),
        "support_per_class": dict(zip(labels_order, support)),
    }

    return metrics


def format_per_class(metrics_dict: dict[str, float], support_dict: dict[str, int]) -> str:
    """
    Helper function to write metrics by class.

    Parameters
    ----------
    metrics_dict: dict[str, float]
        Dictionary where the keys are the names of each class and the values are the metric values
    support_dict: dict[str, int]
        Dictionary where the keys are the names of each class and the values are the number of rows
        of that class

    Returns
    -------
    str
        The formatted verbose to report the metric by class
    """
    return " | ".join(
        f"{cls}: {metrics_dict[cls]:.2f} (n={support_dict[cls]})"
        for cls in metrics_dict
    )


def fit(
    model: nn.Module,
    df: pd.DataFrame,
    origin: str,
    fold: int,
    df_config: dict[str, str],
    params: dict,
    num_classes: int,
    class_to_idx: dict[str, int],
    max_epochs: int = MAX_EPOCHS,
    trial: Trial | None = None,
    device: torch.device = DEVICE
) -> tuple[list[dict], dict]:
    """
    Trains a model.

    Parameters
    ----------
    model: nn.Module
        The Deep Learning model to be evaluated
    df: pd.DataFrame
        The DataFrame containing the data information
    origin: str
        Path with the input images
    fold: int
        Value of fold to analize in validation
    df_config: dict[str, str]
        Dictionary with the name of the columns containing the image path names ("image_col"), the
        labels ("label_col") and the fold idx ("fold_col").
    params: dict
        Dictionary with the hyperparameters for model and training definition
    num_classes: int
        Number of output classes
    class_to_idx: dict[str, int]
        Dictionary relating the original label to ordinal encoding label
    max_epochs: int, optional
        Number of epochs to train for so long (default = MAX_EPOCHS)
    trial: Trial | None, optional
        Trial information for hyperparameter tuning (default = None)
    device: torch.device, optional
        The device object ("cuda" or "cpu") (default = DEVICE)

    Returns
    -------
    history: list[dict]
        Epoch log for each epoch, where each log has:
        - fold: int
        - train_loss: float
        - val_loss: float
        - train_accuracy: float
        - val_accuracy: float
        - train_f1_macro: float
        - val_f1_macro: float
        - train_f1_weighted: float
        - val_f1_weighted: float
        - train_mcc: float
        - val_mcc: float
        - train_pr_auc: float
        - val_pr_auc: float
        - lr: float
        - train_f1_{class}: float
        - val_f1_{class}: float
        - val_support_{class}: int
    final_metrics: dict
        Final merics evaluation of validation set + best epoch
    """
    # Get dataloaders for the current fold
    dataloaders = get_dataloaders(
        df=df,
        origin=origin,
        fold_val=fold,
        class_to_idx=class_to_idx,
        image_col=df_config["image_col"],
        label_col=df_config["label_col"],
        fold_col=df_config["fold_col"],
        batch_size=params["batch_size"],
        img_size=IMG_SIZE,
    )
    train_loader = dataloaders["train_loader"]
    val_loader = dataloaders["val_loader"]
    
    # Build the loss function
    criterion = build_loss_function(**params)
    
    # Build the optimizer
    optimizer = build_optimizer(model=model, **params)
    
    # Build the learning rate scheduler
    scheduler = build_scheduler(
        optimizer=optimizer,
        lr_max=params["lr_max"],
        lr_min_ratio=params["lr_min_ratio"],
        num_epochs=params["num_epochs"],
        warmup_epochs=WARMUP_EPOCHS,
    )
    
    early_stopping = EarlyStopping(
        patience=10,
        mode="max",
        min_delta=1e-4,
    )
    
    # Training loop
    history = []
    for epoch in range(1, max_epochs + 1):
        
        # Train for one epoch
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        
        # Training metrics
        train_metrics = evaluate(
            model=model,
            loader=train_loader,
            criterion=criterion,
            num_classes=num_classes,
            class_to_idx=class_to_idx,
            device=device,
        )
        
        # Validation metrics
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            num_classes=num_classes,
            class_to_idx=class_to_idx,
            device=device,
        )
        
        val_mcc = val_metrics["mcc"]
        
        # Update learning rate and early stopping
        scheduler.step()
        early_stopping.step(
            score=val_mcc,
            model=model,
            epoch=epoch,
        )
        
        if trial is not None:
            trial.report(val_mcc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_log = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["balanced_accuracy"],
            "val_accuracy": val_metrics["balanced_accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "train_f1_weighted": train_metrics["f1_weighted"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "train_mcc": train_metrics["mcc"],
            "val_mcc": val_metrics["mcc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "lr": current_lr,
        }
        for cls in val_metrics["f1_per_class"]:
            safe_cls = cls.replace(" ", "_").replace("+", "_")
            epoch_log[f"train_f1_{safe_cls}"] = train_metrics["f1_per_class"][cls]
            epoch_log[f"val_f1_{safe_cls}"] = val_metrics["f1_per_class"][cls]
            epoch_log[f"val_support_{safe_cls}"] = val_metrics["support_per_class"][cls]

        history.append(epoch_log)
        
        print(
            f"Epoch {epoch}/{max_epochs}: lr: {current_lr:.4g}\n"
            f"[Train] Loss: {train_loss:.4f} | BAcc: {train_metrics['balanced_accuracy']:.4f} | "
            f"F1(macro): {train_metrics['f1_macro']:.4f} | F1(weighted): {train_metrics['f1_weighted']:.4f} | "
            f"MCC: {train_metrics['mcc']:.4f} | PR-AUC: {train_metrics['pr_auc']:.4f}\n"
            f"[Val]   Loss: {val_metrics['loss']:.4f} | BAcc: {val_metrics['balanced_accuracy']:.4f} | "
            f"F1(macro): {val_metrics['f1_macro']:.4f} | F1(weighted): {val_metrics['f1_weighted']:.4f} | "
            f"MCC: {val_metrics['mcc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}\n"
            f"[Val per-class F1] {format_per_class(val_metrics['f1_per_class'], val_metrics['support_per_class'])}\n"
        )
        
        if early_stopping.should_stop:
            print("Early stopping.")
            break
    
    # Load the best model state
    model.load_state_dict(early_stopping.best_state)
    
    # Final evaluation
    final_metrics = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        num_classes=num_classes,
        class_to_idx=class_to_idx,
        device=device,
    )
    final_metrics["best_epoch"] = early_stopping.best_epoch
    
    return history, final_metrics
    

def train_cv(
    df: pd.DataFrame,
    df_config: dict[str, str],
    origin: str,
    params: dict,
    class_to_idx: dict[str, int],
    model_name: str,
    n_folds: int = 5,
    max_epochs: int = MAX_EPOCHS,
    in_channels: int = 1,
    num_classes: int = 7,
    device: torch.device = DEVICE,
    save: bool = True,
):
    """
    Performs an n-fold Cross-Validation of the model.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the data information
    df_config: dict[str, str]
        Dictionary with the name of the columns containing the image path names ("image_col"), the
        labels ("label_col") and the fold idx ("fold_col").
    origin: str
        Path with the input images
    params: dict
        Dictionary with the hyperparameters for model and training definition
    class_to_idx: dict[str, int]
        Dictionary relating the original label to ordinal encoding label
    model_name: str
        Model name
    n_folds: int, optional
        Number of folds of Cross-Validation (default = 5)
    max_epochs: int, optional
        Number of epochs to train for so long (default = MAX_EPOCHS)
    in_channels: int, optional
        Number of input channels (default = 1)
    num_classes: int, optional
        Number of output classes (default = 7)
    device: torch.device, optional
        The device object ("cuda" or "cpu") (default = DEVICE)
    save: bool, optional
        Whether to save models and results or not (default = False)
    """
    
    set_seed()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if save:
        create_folder_structure(model_name=model_name)
        
    fold_results = {}
    fold_history = {}
    # Cross-validation loop
    for fold in range(1, n_folds + 1):
        
        print("=" * 60)
        print(f"FOLD {fold}")
        print("=" * 60)
        
        # Build the model
        model = build(
            depth=params["depth"],
            base_filters=params["base_filters"],
            alpha_leaky_relu=params["alpha_leaky_relu"],
            embedding_dim=params["embedding_dim"],
            hidden_dim=params["hidden_dim"],
            dropout_cnn=params["dropout_cnn"],
            dropout_fc=params["dropout_fc"],
            in_channels=in_channels,
            num_classes=num_classes,
        ).to(device)
        
        # Train the model
        history, final_metrics = fit(
            model=model, 
            df=df,
            origin=origin,
            fold=fold,
            df_config=df_config,
            params=params,
            num_classes=num_classes,
            class_to_idx=class_to_idx,
            max_epochs=max_epochs,
            device=device
        )
        
        fold_results[fold] = final_metrics
        fold_history[fold] = history
        
        # Save model
        if save:
            torch.save(
                model.state_dict(),
                f"{MODELS_CV_PATH}/{model_name}/{model_name}_fold_{fold}.pt"
            )
            
    results_df = pd.DataFrame(fold_results).T
    history_df = pd.DataFrame(fold_history)
    
    if save:   
        results_df.to_csv(f"{RESULTS_CV_PATH}/{model_name}/cv_results.csv", index=False)
        history_df.to_csv(f"{RESULTS_CV_PATH}/{model_name}/cv_history.csv", index=False)
        
    print("\nFINAL CV RESULTS")
    print(results_df.mean())
    print(results_df.std())
    print()
        