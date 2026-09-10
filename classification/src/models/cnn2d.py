"""
cnn2d.py

Module to build 2D-CNN and variants.
"""


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A convolutional block:
    Conv2D -> BatchNorm -> LeakyReLU -> Dropout ->
    Conv2D -> BatchNorm -> LeakyReLU -> Dropout -> MaxPool

    Parameters
    ----------
    in_channels: int
        Number of 2D channels of input
    out_channels: int
        Number of 2D channels of output
    alpha_leaky_relu: float, optional
        Slope of negative values in LeakyRelU (default = 0.01)
    dropout: float, optional
        Dropout value (default = 0.0)
    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            alpha_leaky_relu: float = 0.01, 
            dropout: float = 0.0
        ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=alpha_leaky_relu, inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=alpha_leaky_relu, inplace=True),
            nn.Dropout2d(dropout),

            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        """
        Performs a forward pass through the convolutional block
        """
        return self.block(x)
    

class CNNClassifier_MultiClass(nn.Module):
    """
    A multiclass classifier:
    n x ConvBlock -> AdaptativeAvgPool + Flatten -> FullyConnected' -> Embedding ()
    -> FullyConnected -> Linear -> logits

    FullyConnected:
    Linear + BatchNorm (+ LeakyReLU + Dropout)

    Parameters
    ----------
    depth: int
        Number of convolutional block
    base_filters: int
        Number of filters in the initial convolutional block.
        Number of filters in each convolutional block duplicates.
    alpha_leaky_relu: float
        Slope of negative values in LeakyRelU
    embedding_dim: int
        Number of features of embedding
    hidden_dim: int
        Number of neurons in hidden layer
    dropout_cnn: float
        Dropout value for convolutional layers
    dropout_fc: float
        Dropout value for fully connected layers
    in_channels: int, optional
        Number of input channels (default = 1)
    num_classes: int, optional
        Number of output classes (default = 7)
    """
    def __init__(
        self,
        depth: int,
        base_filters: int,
        alpha_leaky_relu: float,
        embedding_dim: int,
        hidden_dim: int,
        dropout_cnn: float,
        dropout_fc: float,
        in_channels: int = 1,
        num_classes: int = 7,
    ):
        super().__init__()

        filters = [base_filters * (2 ** i) for i in range(depth)]

        ## Convolutional blocks
        
        current_channels = in_channels
        blocks = []
        for f in filters:
            blocks.append(
                ConvBlock(current_channels, f, alpha_leaky_relu=alpha_leaky_relu, dropout=dropout_cnn)
            )
            current_channels = f

        self.encoder = nn.Sequential(*blocks)
        
        ## Embedding head

        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters[-1], embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        ## Classifier head
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=alpha_leaky_relu, inplace=True),
            nn.Dropout(dropout_fc),
            
            nn.Linear(hidden_dim, num_classes),
        )

    def forward_features(self, x):
        """
        Performs a forward pass through the encoder and returns feature embedding
        """
        x = self.encoder(x)
        x = self.feature_head(x)
        return x

    def forward(self, x):
        """
        Performs a forward pass through the classifier and returns the logits
        """
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x):
        """
        Performs a forward pass through the classifier, applies softmax and returns probabilities.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x):
        """
        Performs a forward pass through the classifier, and returns the predicted class.
        """
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)
    

def build(
    depth: int,
    base_filters: int,
    alpha_leaky_relu: float,
    embedding_dim: int,
    hidden_dim: int,
    dropout_cnn: float,
    dropout_fc: float,
    in_channels: int = 1,
    num_classes: int = 7,
):
    """
    Given the parameters of the CNNClassifier_Multiclass, returns the instance model.
    
    Parameters
    ----------
    depth: int
            Number of convolutional block
    base_filters: int
        Number of filters in the initial convolutional block.
        Number of filters in each convolutional block duplicates.
    alpha_leaky_relu: float
        Slope of negative values in LeakyRelU
    embedding_dim: int
        Number of features of embedding
    hidden_dim: int
        Number of neurons in hidden layer
    dropout_cnn: float
        Dropout value for convolutional layers
    dropout_fc: float
        Dropout value for fully connected layers
    in_channels: int, optional
        Number of input channels (default = 1)
    num_classes: int, optional
        Number of output classes (default = 7)
    
    Returns
    -------
    CNNClassifier_MultiClass
        Instance model
    """
    
    return CNNClassifier_MultiClass(
        depth,
        base_filters,
        alpha_leaky_relu,
        embedding_dim,
        hidden_dim,
        dropout_cnn,
        dropout_fc,
        in_channels=in_channels,
        num_classes=num_classes,
    )
    