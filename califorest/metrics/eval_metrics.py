"""
Evaluation Metrics for Conformal Prediction Calibration Assessment.

This module provides statistical tests and metrics to evaluate the calibration
quality of conformal prediction models. These metrics assess whether predicted
probabilities align with observed frequencies, which is crucial for reliable
uncertainty quantification in conformal prediction frameworks.
"""
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from scipy.stats import chi2
from scipy.stats import norm

def hosmer_lemeshow(y_true, y_score):
    """
    Compute the Hosmer-Lemeshow goodness-of-fit test for calibration assessment.

    The Hosmer-Lemeshow test evaluates whether predicted probabilities from a
    conformal predictor match the observed event rates across risk groups. This
    test is particularly useful for assessing the calibration of probability
    estimates in conformal prediction frameworks.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities for the positive class, must be in [0, 1].

    Returns
    -------
    p_value : float
        P-value from the chi-squared test. Higher values (typically > 0.05)
        indicate better calibration, suggesting that observed and expected
        event rates are consistent.

    Notes
    -----
    The test statistic is computed as:

    .. math::
        HL = \\sum_{g=1}^{G} \\frac{(O_{1g} - E_{1g})^2}{N_g \\pi_g (1 - \\pi_g)}

    where :math:`G=10` is the number of groups, :math:`O_{1g}` is the observed
    number of events in group :math:`g`, :math:`E_{1g}` is the expected number
    of events, :math:`N_g` is the group size, and :math:`\\pi_g` is the mean
    predicted probability in group :math:`g`.

    The test statistic follows a chi-squared distribution with :math:`G-2`
    degrees of freedom under the null hypothesis of perfect calibration.

    References
    ----------
    .. [1] Hosmer, D. W., & Lemeshow, S. (2000). Applied Logistic Regression
           (2nd ed.). Wiley.
    .. [2] Kramer, A. A., & Zimmerman, J. E. (2007). Assessing the calibration
           of mortality benchmarks in critical care: The Hosmer-Lemeshow test
           revisited. Critical Care Medicine, 35(9), 2052-2056.
    """
    n_grp = 10 # number of groups

    # create the dataframe
    df = pd.DataFrame({'score': y_score, 'target': y_true})

    # sort the values
    df = df.sort_values('score')
    # shift the score a bit
    df['score'] = np.clip(df['score'], 1e-8, 1-1e-8)
    df['rank'] = list(range(df.shape[0]))
    # cut them into 10 bins
    df['score_decile'] = pd.qcut(df['rank'], n_grp,
                                      duplicates='raise')
    # sum up based on each decile
    obsPos = df['target'].groupby(df.score_decile).sum()
    obsNeg = (df['target'].groupby(df.score_decile).count() - 
                obsPos)
    exPos = df['score'].groupby(df.score_decile).sum()
    exNeg = df['score'].groupby(df.score_decile).count() - exPos
    hl = (((obsPos - exPos)**2/exPos) + ((obsNeg - exNeg)**2/exNeg)).sum()

    # https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    # Re: p-value, higher the better Goodness-of-Fit
    p_value = 1 - chi2.cdf(hl, n_grp-2)
    
    return p_value

def reliability(y_true, y_score):
    """
    Compute reliability metrics for conformal prediction calibration.

    This function calculates two components of the reliability (calibration)
    curve: within-bin reliability and between-bin reliability. These metrics
    quantify how well predicted probabilities match observed frequencies,
    which is essential for evaluating conformal prediction validity.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities for the positive class, must be in [0, 1].

    Returns
    -------
    rel_small : float
        Within-bin reliability (refinement loss). Mean squared difference
        between observed and expected frequencies within deciles. Lower
        values indicate better calibration within risk groups.
    rel_large : float
        Between-bin reliability (calibration-in-the-large). Squared difference
        between overall mean observed and predicted probabilities. Lower
        values indicate better overall calibration.

    Notes
    -----
    The reliability metrics decompose calibration error into two components:

    - **Within-bin (rel_small)**: Measures local calibration quality

      .. math::
          \\text{rel}_\\text{small} = \\frac{1}{G} \\sum_{g=1}^{G} (\\bar{y}_g - \\bar{p}_g)^2

    - **Between-bin (rel_large)**: Measures global calibration bias

      .. math::
          \\text{rel}_\\text{large} = (\\bar{y} - \\bar{p})^2

    where :math:`G=10` is the number of deciles, :math:`\\bar{y}_g` and
    :math:`\\bar{p}_g` are the mean observed and predicted values in group
    :math:`g`, and :math:`\\bar{y}` and :math:`\\bar{p}` are the overall means.

    References
    ----------
    .. [1] Murphy, A. H. (1973). A new vector partition of the probability
           score. Journal of Applied Meteorology, 12(4), 595-600.
    .. [2] DeGroot, M. H., & Fienberg, S. E. (1983). The comparison and
           evaluation of forecasters. The Statistician, 32(1/2), 12-22.
    """

    n_grp = 10
    df = pd.DataFrame({'score': y_score, 'target': y_true})
    df = df.sort_values('score')
    df['rank'] = list(range(df.shape[0]))
    df['score_decile'] = pd.qcut(df['rank'], n_grp,
                                      duplicates='raise')

    obs = df['target'].groupby(df.score_decile).mean()
    exp = df['score'].groupby(df.score_decile).mean()

    rel_small = np.mean((obs - exp)**2)
    rel_large = (np.mean(y_true) - np.mean(y_score))**2

    return rel_small, rel_large

def spiegelhalter(y_true, y_score):
    """
    Compute the Spiegelhalter z-test for calibration assessment.

    The Spiegelhalter test provides a statistical assessment of whether
    predicted probabilities are well-calibrated by comparing observed outcomes
    to predictions. This test is particularly valuable for conformal prediction
    systems where calibration guarantees are critical.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities for the positive class, must be in [0, 1].

    Returns
    -------
    p_value : float
        Two-tailed p-value from the z-test. Higher values (typically > 0.05)
        indicate good calibration, suggesting that predictions are unbiased.

    Notes
    -----
    The Spiegelhalter z-statistic is computed as:

    .. math::
        z = \\frac{\\sum_{i=1}^{n} (y_i - p_i)(1 - 2p_i)}
                 {\\sqrt{\\sum_{i=1}^{n} (1 - 2p_i)^2 p_i (1 - p_i)}}

    where :math:`y_i` are the true labels and :math:`p_i` are the predicted
    probabilities. Under the null hypothesis of perfect calibration, this
    statistic follows a standard normal distribution.

    The test is sensitive to systematic over- or under-prediction across the
    entire probability range, making it complementary to the Hosmer-Lemeshow
    test which focuses on group-level calibration.

    References
    ----------
    .. [1] Spiegelhalter, D. J. (1986). Probabilistic prediction in patient
           management and clinical trials. Statistics in Medicine, 5(5), 421-433.
    .. [2] Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a
           novel method for evaluating prediction models. Medical Decision
           Making, 26(6), 565-574.
    """
    top = np.sum((y_true - y_score)*(1-2*y_score))
    bot = np.sum((1-2*y_score)**2 * y_score * (1-y_score))
    sh = top / np.sqrt(bot)

    # Two-tailed z-test for calibration
    p_value = norm.sf(np.abs(sh)) * 2

    return p_value

def scaled_brier_score(y_true, y_score):
    """
    Compute the Brier score and scaled Brier score for prediction accuracy.

    The Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes, combining both calibration and
    discrimination. The scaled version normalizes this score relative to a
    non-informative predictor, making it useful for comparing conformal
    prediction models across different datasets.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities for the positive class, must be in [0, 1].

    Returns
    -------
    brier : float
        Brier score (mean squared error of predictions). Range: [0, 1],
        where 0 indicates perfect predictions and 1 indicates worst predictions.
    brier_scaled : float
        Scaled Brier score (Brier skill score). Range: (-∞, 1], where 1
        indicates perfect predictions, 0 indicates performance equal to
        predicting the base rate, and negative values indicate worse than
        base rate predictions.

    Notes
    -----
    The Brier score is defined as:

    .. math::
        BS = \\frac{1}{n} \\sum_{i=1}^{n} (p_i - y_i)^2

    The scaled Brier score (Brier skill score) is:

    .. math::
        BSS = 1 - \\frac{BS}{BS_{\\text{ref}}}

    where :math:`BS_{\\text{ref}} = \\bar{y}(1 - \\bar{y})` is the Brier score
    of a non-informative predictor that always predicts the base rate
    :math:`\\bar{y}`.

    For conformal prediction, the Brier score provides an overall assessment
    of prediction quality that accounts for both calibration (reliability)
    and resolution (ability to discriminate between outcomes).

    References
    ----------
    .. [1] Brier, G. W. (1950). Verification of forecasts expressed in terms
           of probability. Monthly Weather Review, 78(1), 1-3.
    .. [2] Steyerberg, E. W., et al. (2010). Assessing the performance of
           prediction models: a framework for traditional and novel measures.
           Epidemiology, 21(1), 128-138.
    """
    brier = skm.brier_score_loss(y_true, y_score)
    # Calculate the mean of the probability (base rate)
    p = np.mean(y_true)  
    brier_scaled = 1 - brier / (p * (1-p))
    return brier, brier_scaled
