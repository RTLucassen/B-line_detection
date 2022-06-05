"""
Implementation of loss functions and general training utility functions.
"""

import torch
import random
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 

from typing import Union

# ----------------------------  GENERAL  ----------------------------

def get_lr(optimizer: torch.optim) -> float:
    """ 
    Args:
        optimizer:  optimizer object
    
    Returns:
        current learning rate value
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def seed_worker(worker_id: int) -> None:
    """ 
    Required for reproducibility when using multi-processing.
    
    Args:
        worker_id:  worker index
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def stop_early(loss: list, patience: int) -> bool:
    """ 
    Indicates if the loss has not decreased in the past 'patience' epochs.
    
    Args:
        loss:  loss values of the past epochs (should be the validation loss)
        patience:  number of epochs that the loss value should not decrease
    
    Return:
        boolean indicating whether training should be stopped
    """
    # check if at least patience +1 epochs have passed
    if len(loss) <= patience:
        return False
    # otherwise check if the loss has not improved in the epochs within the patience range 
    for i in np.arange(1, patience+1):
        if loss[-i] < loss[-(patience+1)]:
            return False
    # if this is not the case, stop training early
    return True

def get_prediction(pred: Union[collections.OrderedDict, torch.Tensor]) -> torch.Tensor:
    """
    Args:
        pred:  prediction by Pytorch model.
    
    Returns:
        prediction:  predicted tensor.
    """ 
    output = []
    # if the prediction is an ordered dictionary, get the tensor
    if isinstance(pred, collections.OrderedDict):
        pred = pred['out']
    # else check if the prediction is or contains torch tensors and add them to the output list
    if isinstance(pred, torch.Tensor):
        output.append(pred)
    elif isinstance(pred, tuple):
        for p in pred:
            # if pred is not a tensor, raise an error
            if isinstance(p, torch.Tensor):
                output.append(p)
    else:
        raise ValueError('Unknown type for prediction')

    if len(output) == 0:
        return None
    elif len(output) == 1:
        return output[0]
    else:
        return output


# ------------------------------  LOSSES  ------------------------------

class DiceLoss(nn.Module):
    """
    Adapted from: https://docs.monai.io/en/latest/_modules/monai/losses/dice.html#DiceLoss
    """

    def __init__(
        self, 
        sigmoid: bool = False, 
        smooth_nom: float = 1, 
        smooth_denom: float = 1,
        frequency_weighting: bool = False
    ) -> None:
        """
        Args:
            sigmoid:  if True, feed logit activations through sigmoid function instead of the default softmax function.
                      If there is only a single class, the sigmoid is automatically used.
            smooth_nom:  small value added to the nominator to better handle negative cases.
            smooth_denom:  small value added to the denominator to prevent division by zero errors and to better handle negative cases.
            frequency_weighting:  if True, weight all images in the batch with the inverse amount of frames from the specific clip in the training epoch.
                                  if False, all instances have the same weight (can introduce a bias if there are a different number of instances available per clip).
        """
        super().__init__()
        # define the instance attributes
        self.sigmoid = sigmoid
        self.smooth_nom = smooth_nom
        self.smooth_denom = smooth_denom
        self.frequency_weighting = frequency_weighting

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        """ 
        Args:
            logit:  logit predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  true label volumes of matching shape: (batch, class, X, Y, ...).
            weight:  weight for every frame based on the number of frames from that particular clip in the training epoch.

        Returns:
            loss:  Dice loss (1-Dice score) averaged over all images in the batch.
        """
        # check if the logit prediction and true labels are of equal shape
        if logit.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels does not match.')

        # get the pixel-wise predicted probabilities by taking
        # the sigmoid or softmax of the logit returned by the network
        if self.sigmoid or logit.shape[1] == 1:
            y_pred = torch.sigmoid(logit)
        else:
            y_pred = torch.softmax(logit, dim=1)

        # flatten the image (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)
        
        # compute the dice loss
        intersection = torch.sum(y_true_flat * y_pred_flat, dim=-1)
        union = torch.sum(y_true_flat, dim=-1) + torch.sum(y_pred_flat, dim=-1)
        class_separated_dice = 1 - ((2. * intersection + self.smooth_nom) / (union + self.smooth_denom))
        instance_loss = torch.mean(class_separated_dice, dim=1)

        # apply a weighting based on the frequency if specified
        if self.frequency_weighting:
            if weight == None:
                raise ValueError('No instance weights were given as input argument.')
            elif weight.shape != instance_loss.shape:
                raise ValueError('Number of instances and instance weights do not match.')
            else:
                instance_loss *= weight

        # take the mean loss over the batch
        loss = torch.mean(instance_loss)

        return loss

class FocalLoss(nn.Module):
    """
    Adapted from: https://docs.monai.io/en/latest/_modules/monai/losses/focal_loss.html
    """

    def __init__(
        self,
        sigmoid: bool = False, 
        gamma: float = 2.0,
        class_weights: torch.Tensor = None,
        frequency_weighting: bool = False
    ) -> None:
        """
        Args:
            gamma:  value that influences the importance of difficult cases.
            class_weights:  if not an empty list, multiply each class with a corresponding weight, according to the index. 
            frequency_weighting:  if True, weight all images in the batch with the inverse amount of frames from the specific clip in the training epoch.
                                  if False, all instances have the same weight (can introduce a bias if there are a different number of instances available per clip).
        """
        super().__init__()
        self.sigmoid = sigmoid
        self.gamma = gamma
        self.class_weights = class_weights
        self.frequency_weighting = frequency_weighting

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        """ 
        Args:
            logit:  logit predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  true label volumes of matching shape: (batch, class, X, Y, ...).
            weight:  weight for every frame based on the number of frames from that particular clip in the training epoch.

        Returns:
            loss:  Focal loss averaged over all images in the batch.
        """
        # check if the logit prediction and true labels are of equal shape
        if logit.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels does not match.')
        
        # get the pixel-wise predicted probabilities by taking
        # the sigmoid or softmax of the logit returned by the network
        if self.sigmoid or logit.shape[1] == 1:
            y_pred = torch.sigmoid(logit)
            log_y_pred = F.logsigmoid(logit)
        else:
            y_pred = torch.softmax(logit, dim=1)
            log_y_pred = F.log_softmax(logit, dim=1)

        # flatten the images and labels (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)
        log_y_pred_flat = log_y_pred.contiguous().view(*y_pred.shape[0:2], -1)

        # calculate the pixelwise binary cross-entropy, focal weight, and pixelwise focal loss
        pixelwise_bce = -(log_y_pred_flat * y_true_flat)
        focal_weight = (1-(y_true_flat * y_pred_flat))**self.gamma
        pixelwise_focal_loss = focal_weight * pixelwise_bce
        # calculate the class-separated focal loss
        class_separated_focal_loss = torch.mean(pixelwise_focal_loss, dim=-1)
        
        # compute the instance loss (optionally weighted)
        if self.class_weights != None:
            # first check if there is a weight for every class
            if class_separated_focal_loss.shape[1] != len(self.class_weights):
                raise ValueError('Number of class channels and class weights do not match.')
            # apply weighting to every class
            for i, class_weight in enumerate(self.class_weights):
                class_separated_focal_loss[:, i] *= class_weight
        
        instance_loss = torch.sum(class_separated_focal_loss, dim=1)

        # apply weighting based on the frequency if specified
        if self.frequency_weighting:
            if weight == None:
                raise ValueError('No instance weights were given as input argument.')
            elif weight.shape != instance_loss.shape:
                raise ValueError('Number of instances and instance weights do not match.')
            else:
                instance_loss *= weight

        # take the mean loss over the batch
        loss = torch.mean(instance_loss)

        return loss

# Used for loss configuration by Roy et al.
class MSEConsistancyLoss(nn.Module):

    def __init__(
        self,
        frequency_weighting: bool = False
    ) -> None:
        """
        Args:
            frequency_weighting:  if True, weight all images in the batch with the inverse amount of frames from the specific clip in the training epoch.
                                  if False, all instances have the same weight (can introduce a bias if there are a different number of instances available per clip).
        """
        super().__init__()
        self.frequency_weighting = frequency_weighting

    def forward(self, y_pred_1: torch.Tensor, y_pred_2: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        """ 
        Args:
            y_pred_1:  logit predictions volumes of shape: (batch, class, X, Y, ...).
            y_pred_2:  logit predictions volumes of shape: (batch, class, X, Y, ...).
            weight:  weight for every frame based on the number of frames from that particular clip in the training epoch.

        Returns:
            loss:  MSE consistency loss averaged over all images in the batch.
        """
        # check if the logit prediction and true labels are of equal shape
        if y_pred_1.shape != y_pred_2.shape:
            raise ValueError('Shape of predicted and true labels does not match.')

        # flatten the images and labels (but keep the dimension of the batch and channels)
        y_pred_1_flat = y_pred_1.contiguous().view(*y_pred_1.shape[0:2], -1)
        y_pred_2_flat = y_pred_2.contiguous().view(*y_pred_2.shape[0:2], -1)
        # calculate the pixelwise L2 distance (mean squared error)
        L2_loss = (y_pred_1_flat-y_pred_2_flat)**2
        # calculate the class-separated L2 distance 
        class_separated_L2_loss = torch.mean(L2_loss, dim=-1)        
        instance_loss = torch.sum(class_separated_L2_loss, dim=1)

        # apply a weighting based on the frequency if specified
        if self.frequency_weighting:
            if weight == None:
                raise ValueError('No instance weights were given as input argument.')
            elif weight.shape != instance_loss.shape:
                raise ValueError('Number of instances and instance weights do not match.')
            else:
                instance_loss *= weight

        # take the mean loss over the batch
        loss = torch.mean(instance_loss)

        return loss

class RoyEtalLoss(nn.Module):

    def __init__(
        self,
        sigmoid: bool = False, 
        class_weights: torch.Tensor = None,
        frequency_weighting: bool = False
    ) -> None:
        """
        Args:
            sigmoid:  if True, feed logit activations through sigmoid function instead of the default softmax function.
                      If there is only a single class, the sigmoid is automatically used.
            class_weights:  if not an empty list, multiply each class with a corresponding weight, according to the index. 
            frequency_weighting:  if True, weight all images in the batch with the inverse amount of frames from the specific clip in the training epoch.
                                  if False, all instances have the same weight (can introduce a bias if there are a different number of instances available per clip).
        """
        super().__init__()
        self.sigmoid = sigmoid
        self.class_weights = class_weights
        self.frequency_weighting = frequency_weighting

        self.consistency_loss = MSEConsistancyLoss(self.frequency_weighting)
        self.pred_loss = FocalLoss(self.sigmoid, 0, self.class_weights, self.frequency_weighting)

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        """ 
        Args:
            logit:  logit predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  true label volumes of matching shape: (batch, class, X, Y, ...).
            weight:  weight for every frame based on the number of frames from that particular clip in the training epoch.

        Returns:
            loss:  Loss used by Roy et al. for fixed scale.
        """
        output_1, output_2 = torch.split(logit, split_size_or_sections=logit.shape[0] // 2)
        # compute the consistency loss based on the logits and the prediction loss
        consistancy_loss_value = self.consistency_loss(output_1, output_2, weight)
        pred_loss_value = self.pred_loss(output_1, y_true, weight)

        return consistancy_loss_value + pred_loss_value


def get_loss_function(name: str, settings: dict) -> nn.Module:
    """
    Args:
        name:  name of loss
        settings:  contains names of settings and corresponding values (key, values pairs listed below):
                   o   frequency_weighting:  indicates whether each instance in a batch 
                                             should be weighted using the frequency weighting.
                   o   gamma:  parameter to influence relative importance of wrongly classified examples (focal loss only).
                   o   class_weights:  weights to multiply each class contribution with (equal in length to the number of classes).
                   o   dice_focal_weights:  weights to multiply to the dice and focal loss respectively before aggregating (dice-focal loss only)
    Returns:
        loss_function:  class instance of selected loss function
    """
    # define the parameters
    frequency_weighting = True if 'frequency_weighting' not in settings else settings['frequency_weighting']
    gamma = 2.0 if 'gamma' not in settings else settings['gamma']
    class_weights = None if 'class_weights' not in settings else settings['class_weights']
    dice_focal_weights = [1, 1] if 'dice_focal_weights' not in settings else settings['dice_focal_weights']

    # define a class instance of the selected loss function using the specified settings
    if name == 'dice_loss':
        loss_function = DiceLoss(frequency_weighting=frequency_weighting)

    elif name == 'focal_loss':
        loss_function = FocalLoss(
            gamma=gamma, 
            class_weights=class_weights, 
            frequency_weighting=frequency_weighting
        )

    elif name == 'BCE_loss':
        loss_function = FocalLoss(
            gamma=0, 
            class_weights=None, # i.e. equal weighting per class, since classes are balanced
            frequency_weighting=frequency_weighting
        )

    elif name == 'dice_focal_loss':
        # define the individual loss functions
        dice_loss = DiceLoss(frequency_weighting=frequency_weighting)
        focal_loss = FocalLoss(
            gamma=gamma, 
            class_weights=class_weights, 
            frequency_weighting=frequency_weighting
        )
        # define a combined loss function
        def dice_focal_loss(*args, **kwargs):
            aggregated_loss = dice_focal_weights[0] * dice_loss(*args, **kwargs) + dice_focal_weights[1] * focal_loss(*args, **kwargs)
            return aggregated_loss

        loss_function = dice_focal_loss

    elif name == 'roy_etal_loss':
        loss_function = RoyEtalLoss(
            class_weights=None, # i.e. equal weighting per class, since classes are balanced
            frequency_weighting=frequency_weighting
        )
        
    else:
        raise ValueError('Name of loss function was not recognized.')
    
    return loss_function


if __name__ == '__main__':

    # create a binary prediction and label
    logit = torch.ones(1, 10, 10) * -100
    logit[0, :, 0:1] = 100
    logit = torch.cat([-logit, logit], dim=0)[None, ...]
    plt.imshow(logit[0, 1, ...]); plt.show()

    y_true = torch.zeros(1, 10, 10)
    y_true[0, 4:6, 4:6] = 1
    y_true = torch.cat([(1-y_true), y_true], dim=0)[None, ...]
    plt.imshow(y_true[0, 1, ...]); plt.show()

    frequency_weighting = torch.tensor([1])

    # ---------------------------  Dice loss test  ---------------------------

    if False:
        # test example of the dice loss
        loss = DiceLoss(frequency_weighting=True)
        print(loss(logit, y_true, frequency_weighting))
    
    # ---------------------------  Focal loss test  ---------------------------

    if False:
        # test example of the focal loss
        class_weights = torch.tensor([1, 1]).type(torch.FloatTensor)
        gamma = 1
        loss = FocalLoss(gamma=gamma, class_weights=class_weights, frequency_weighting=True)
        print(loss(logit, y_true, frequency_weighting))

    if False:
        # reproduces the figure from paper to check implementation
        for gamma in [0, 1, 2, 5]:
            # configure the loss function
            loss_function = FocalLoss(gamma=gamma, frequency_weighting=False)
            # initialize the variables to store the loss and probability
            prob_record, loss_record = [], []
            # sample the loss for different logit values
            for i in torch.linspace(-20, 20, 1000):
                logit = torch.tensor([[[[i]],[[-i]]]])
                prob_record += [torch.softmax(logit, dim=1)[0,0,0,0].tolist()]
                loss_record += [loss_function(logit, torch.tensor([[[[1]],[[0]]]]), torch.tensor(1)).tolist()]
            
            plt.plot(prob_record, loss_record, label=f'gamma: {gamma}')

        plt.xlim([0,1])
        plt.ylim([0,5])
        plt.legend()
        plt.show()
    