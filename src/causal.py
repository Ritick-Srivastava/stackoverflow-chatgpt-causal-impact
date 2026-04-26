import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize


def run_ols_counterfactual(df, treatment_col, control_cols, intervention_date, alpha=0.05):
    """
    OLS regression counterfactual.

    Fits treatment ~ controls on the pre-intervention period, then projects
    the fitted relationship forward into the post-period to construct the
    counterfactual ("what would SO look like if the intervention never happened").

    Prediction intervals come from statsmodels' get_prediction(), which gives
    proper out-of-sample uncertainty that widens as we move further from the
    training data.

    Returns a dict with counterfactual, CI bands, effect series, and the
    fitted statsmodels result for inspection.
    """
    pre_mask = df.index < pd.Timestamp(intervention_date)

    treatment = df[treatment_col]
    controls  = df[control_cols]

    X = sm.add_constant(controls)

    model  = sm.OLS(treatment[pre_mask], X[pre_mask])
    result = model.fit()

    pred   = result.get_prediction(X)
    frame  = pred.summary_frame(alpha=alpha)

    counterfactual = frame['mean']
    ci_lower       = frame['obs_ci_lower']
    ci_upper       = frame['obs_ci_upper']

    effect    = treatment - counterfactual
    post_mask = ~pre_mask
    rel_effect = (effect[post_mask] / counterfactual[post_mask]) * 100

    return {
        'counterfactual': counterfactual,
        'ci_lower':       ci_lower,
        'ci_upper':       ci_upper,
        'effect':         effect,
        'treatment':      treatment,
        'rel_effect':     rel_effect,
        'r_squared':      result.rsquared,
        'coefficients':   result.params.to_dict(),
        'statsmodels_result': result,
    }


def run_synthetic_control(df, treatment_col, control_cols, intervention_date):
    """
    Build a synthetic control via constrained convex optimization.

    Finds non-negative weights w (summing to 1) over control series that
    minimize pre-intervention RMSE against the treatment series.
    Projects the weighted combination forward as the counterfactual.

    Returns a dict with:
        weights    : {control_name: weight}
        synthetic  : pd.Series — counterfactual on original scale
        effect     : pd.Series — actual minus counterfactual
        treatment  : pd.Series — original treatment values
        pre_rmse   : float     — pre-period fit quality (lower = better)
        rel_effect : pd.Series — effect as % of synthetic (post-period only)
    """
    pre_mask = df.index < pd.Timestamp(intervention_date)

    treatment = df[treatment_col]
    controls  = df[control_cols]

    # Standardize each series to mean=0, std=1 over the pre-period
    # so the optimizer isn't sensitive to absolute scale differences
    t_mean, t_std = treatment[pre_mask].mean(), treatment[pre_mask].std()
    c_mean = controls[pre_mask].mean()
    c_std  = controls[pre_mask].std()

    t_scaled = (treatment - t_mean) / t_std
    c_scaled = (controls  - c_mean) / c_std

    pre_t = t_scaled[pre_mask].values
    pre_c = c_scaled[pre_mask].values
    n = pre_c.shape[1]

    def loss(w):
        return np.sum((pre_t - pre_c @ w) ** 2)

    result = minimize(
        loss,
        x0=np.ones(n) / n,
        method='SLSQP',
        bounds=[(0, 1)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
        options={'ftol': 1e-10, 'maxiter': 1000},
    )

    if not result.success:
        print(f"  Warning: optimizer did not fully converge ({result.message})")

    weights = result.x

    # Project synthetic back to original scale
    synthetic_scaled = c_scaled @ weights
    synthetic = synthetic_scaled * t_std + t_mean

    effect = treatment - synthetic
    post_mask = ~pre_mask
    rel_effect = (effect[post_mask] / synthetic[post_mask]) * 100

    pre_rmse = float(np.sqrt(np.mean((treatment[pre_mask] - synthetic[pre_mask]) ** 2)))

    return {
        'weights':    dict(zip(control_cols, weights.round(4))),
        'synthetic':  synthetic,
        'effect':     effect,
        'treatment':  treatment,
        'pre_rmse':   pre_rmse,
        'rel_effect': rel_effect,
    }


def summarise_synthetic(sc_result, intervention_date):
    """Print a plain-English summary of the synthetic control result."""
    post_mask = sc_result['treatment'].index >= pd.Timestamp(intervention_date)
    effect    = sc_result['effect'][post_mask]
    synth     = sc_result['synthetic'][post_mask]

    cum_effect = effect.sum()
    avg_effect = effect.mean()
    avg_rel    = sc_result['rel_effect'].mean()
    pre_rmse   = sc_result['pre_rmse']

    print("=== Synthetic Control Summary ===")
    print(f"  Donor weights      : {sc_result['weights']}")
    print(f"  Pre-period RMSE    : {pre_rmse:.2f} (lower = better pre-period fit)")
    print(f"  Avg monthly effect : {avg_effect:.2f} units  ({avg_rel:.1f}%)")
    print(f"  Cumulative effect  : {cum_effect:.1f} units over {post_mask.sum()} months")
