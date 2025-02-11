import torch
import numpy as np
from mapie.conformity_scores.sets.lac import LACConformityScore

class _PlaceHolder():
    """A placeholder class used as a stub for an estimator."""
    def __init__(self):
        self.cv = None

def fast_fit(
        score_object:LACConformityScore, 
        y_pred_proba_calib:torch.Tensor, 
        true_labels_calib:torch.Tensor, 
        alpha:float|np.ndarray, 
        **kwargs
    ) -> LACConformityScore:
    """
    Fits the given score object using calibration data and stores quantile values for given alpha(s).

    Args:
        score_object (LACConformityScore): A MAPIE instance that computes conformity scores using the LAC method.
        y_pred_proba_calib (torch.Tensor): Probability predictions outputs from a model for calibration.
        true_labels_calib (torch.Tensor): Ground truth labels corresponding to probability outputs for calibration.
        alpha (float | np.ndarray): A single value or an array of significance levels.
        **kwargs: Additional parameters passed to the `get_conformity_score_quantiles` method.

    Returns:
        LACConformityScore: The fitted score_object
    """

    if not hasattr(score_object, 'quantiles_dict'):
        score_object.quantiles_dict = {}
        
    estimator = _PlaceHolder()
    score_object.conformity_scores = score_object.get_conformity_scores(
        true_labels_calib.cpu().numpy(), 
        y_pred_proba_calib.cpu().numpy(), 
        true_labels_calib.cpu().numpy()
    )
    
    alpha = alpha if hasattr(alpha, '__iter__') else np.array([alpha])

    quantiles = score_object.get_conformity_score_quantiles(score_object.conformity_scores, alpha, estimator, **kwargs)

    for a, alpha_ in enumerate(alpha):
        score_object.quantiles_dict[alpha_] = quantiles[a]

    return score_object

def fast_get_set(
        score_object:LACConformityScore, 
        y_pred_proba:torch.Tensor, 
        alpha:float,  
        **kwargs
    ) -> torch.Tensor:
    """
    Retrieves the prediction sets for given probability predictions and a predefined alpha level. 
    Requires to have called the fast_fit() function first.

    Args:
        score_object (LACConformityScore): A MAPIE instance that computes conformity scores using the LAC method.
        y_pred_proba (torch.Tensor): Probability predictions from a model.
        alpha (float): The desired significance level for the prediction sets.
        **kwargs: Additional parameters passed to the `get_prediction_sets` method.

    Returns:
        torch.Tensor: The computed prediction sets based on given parameters.

    Raises:
        RuntimeError: If the given alpha value is not found in the quantiles dictionary of score_object.
    """
    if not alpha in score_object.quantiles_dict.keys():
        raise RuntimeError(f'Alpha value {alpha} not found in quantiles_dict. Call fast_fit() with alpha first.')
        
    estimator = _PlaceHolder()

    pred_labels = torch.argmax(y_pred_proba, dim = -1).cpu().numpy()

    conformity_scores = score_object.get_conformity_scores(pred_labels, y_pred_proba.cpu().numpy(), pred_labels)

    score_object.quantiles_ = score_object.quantiles_dict[alpha]
    prediction_sets = score_object.get_prediction_sets(
        y_pred_proba, conformity_scores, None, estimator, **kwargs
    )

    return prediction_sets
