"""End-to-end synthetic motor-insurance retention optimisation.

Run: python end_to_end_retention.py
Dependencies: numpy pandas scikit-learn pulp and an available MILP solver (CBC).
Synthetic demonstration only; do not use offers in production without causal,
pricing, compliance, governance, and operational validation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pulp

SEED = 42
RNG = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parent / 'outputs'
OUT.mkdir(exist_ok=True)
OFFERS = (0, 5, 10)
CONTACT_COST = 3.0
SERVICING_COST = 35.0
BUDGET = 18000.0                    # expected incentive + contact spend (£)
MAX_CONTACTS = 280                  # limited operational outreach capacity
RISK_TOLERANCE = 1.06              # max 6% over no-offer expected loss


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def build_customers(n, prefix):
    tenure = RNG.integers(1, 13, n)
    prior_claims = RNG.poisson(0.25, n).clip(0, 3)
    shopping = RNG.beta(2.2, 2.5, n)
    risk = RNG.beta(2.4, 4.0, n)
    premium = np.round(470 + 560 * risk + 35 * prior_claims + RNG.normal(0, 90, n), 2).clip(300, 1200)
    return pd.DataFrame({
        'customer_id': [f'{prefix}{i:06}' for i in range(n)],
        'tenure_years': tenure, 'prior_claims': prior_claims,
        'shopping_score': shopping, 'risk_score': risk,
        'premium': premium,
    })


def true_renewal_prob(df):
    discount = df['offer_pct'].to_numpy() / 100
    return sigmoid(
        0.55 + 0.065 * df['tenure_years'].to_numpy()
        - 1.9 * df['shopping_score'].to_numpy()
        - 0.35 * df['risk_score'].to_numpy()
        - 0.20 * df['prior_claims'].to_numpy()
        + discount * (2.0 + 9.0 * df['shopping_score'].to_numpy()
                      - 0.08 * df['tenure_years'].to_numpy())
    )


def true_conditional_claim_cost(df):
    return (180 + 520 * df['risk_score'].to_numpy()
            + 80 * df['prior_claims'].to_numpy())


def simulate_pilot(n, prefix):
    df = build_customers(n, prefix)
    # IMPORTANT: Randomised historic pilot: allows identification of offer effects.
    df['offer_pct'] = RNG.choice(OFFERS, n, p=[0.5, 0.3, 0.2])
    prob = true_renewal_prob(df)
    df['renewed'] = RNG.binomial(1, prob)
    claim_mean = true_conditional_claim_cost(df)
    # Observed claim cost is only available if the policy renewed.
    df['claim_cost_if_renewed'] = np.where(
        df['renewed'].eq(1), RNG.gamma(shape=8, scale=claim_mean / 8), np.nan
    )
    return df


def features(df):
    """The SAME transformations must be used during model training and scoring."""
    offer = df['offer_pct'].to_numpy() / 100
    return pd.DataFrame({
        'tenure': df['tenure_years'].to_numpy(),
        'prior_claims': df['prior_claims'].to_numpy(),
        'shopping': df['shopping_score'].to_numpy(),
        'risk': df['risk_score'].to_numpy(),
        'premium': df['premium'].to_numpy(),
        'offer': offer,
        'offer_x_shopping': offer * df['shopping_score'].to_numpy(),
        'offer_x_tenure': offer * df['tenure_years'].to_numpy(),
    })


def train_and_evaluate(train, valid):
    response_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500))
    response_model.fit(features(train), train['renewed'])
    pred = response_model.predict_proba(features(valid))[:, 1]
    print(f'Renewal validation: AUC={roc_auc_score(valid.renewed, pred):.3f}, '
          f'Brier={brier_score_loss(valid.renewed, pred):.3f}')
    # This is a model of claim cost CONDITIONAL ON RENEWAL.
    claim_train = train.loc[train['renewed'].eq(1)].copy()
    claim_valid = valid.loc[valid['renewed'].eq(1)].copy()
    claim_columns = ['risk_score', 'prior_claims']
    claim_model = make_pipeline(StandardScaler(), Ridge(alpha=20))
    claim_model.fit(claim_train[claim_columns], claim_train['claim_cost_if_renewed'])
    valid_cost = claim_model.predict(claim_valid[claim_columns]).clip(0)
    print(f'Conditional claims validation MAE=£'
          f'{mean_absolute_error(claim_valid.claim_cost_if_renewed, valid_cost):.2f}')
    return response_model, claim_model


def offer_eligibility(row, offer):
    # Illustrative approved operational policy rules; not an assertion of regulatory compliance.
    if offer == 10 and row.tenure_years < 2:
        return False
    if offer == 10 and row.risk_score > 0.65:
        return False
    return True


def score_scenarios(future, response_model, claim_model):
    candidates = future.loc[future.index.repeat(len(OFFERS))].copy().reset_index(drop=True)
    candidates['offer_pct'] = np.tile(OFFERS, len(future))
    # Explicitly score the counterfactual scenario for each eligible offer.
    candidates['p_renew'] = response_model.predict_proba(features(candidates))[:, 1]
    candidates['expected_claim_if_renewed'] = claim_model.predict(
        candidates[['risk_score', 'prior_claims']]
    ).clip(0)
    discount = candidates['offer_pct'] / 100
    candidates['discount_amount'] = candidates['premium'] * discount
    candidates['contact_cost'] = np.where(candidates.offer_pct.gt(0), CONTACT_COST, 0.0)
    # Premium already includes offer reduction; never subtract discount twice.
    candidates['expected_profit'] = (
        candidates['p_renew'] * (
            candidates['premium'] * (1 - discount)
            - candidates['expected_claim_if_renewed'] - SERVICING_COST
        ) - candidates['contact_cost']
    )
    # Expected redeemed incentive, plus contact cost. This is the budget convention.
    candidates['budget_cost'] = (
        candidates['p_renew'] * candidates['discount_amount'] + candidates['contact_cost']
    )
    candidates['expected_loss_exposure'] = (
        candidates['p_renew'] * candidates['expected_claim_if_renewed']
    )
    allowed = [offer_eligibility(row, int(row.offer_pct))
               for row in candidates.itertuples(index=False)]
    candidates['eligible'] = allowed
    assert candidates.groupby('customer_id').apply(
        lambda g: bool(g.loc[g.offer_pct.eq(0), 'eligible'].all()),
        include_groups=False
    ).all()
    return candidates


def make_solver():
    if hasattr(pulp, 'COIN_CMD'):
        try:
            solver = pulp.COIN_CMD(msg=False, timeLimit=90)
            if solver.available():
                return solver
        except Exception:
            pass
    if hasattr(pulp, 'PULP_CBC_CMD'):
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=90)
        if solver.available():
            return solver
    raise RuntimeError('CBC not found. For PuLP 4 install with: pip install "pulp[cbc]"')


def solve_portfolio(candidates):
    eligible = candidates.loc[candidates.eligible].copy()
    pairs = list(zip(eligible.customer_id, eligible.offer_pct.astype(int)))
    indexed = eligible.set_index(['customer_id', 'offer_pct'])
    customers = candidates.customer_id.unique().tolist()
    model = pulp.LpProblem('Renewal_offer_allocation', pulp.LpMaximize)
    if hasattr(model, 'add_variable_dict'):
        x = model.add_variable_dict('choose', pairs, cat='Binary', lowBound=0, upBound=1)
    else:
        x = pulp.LpVariable.dicts('choose', pairs, lowBound=0, upBound=1, cat='Binary')
    assert all(x[k].isBinary() for k in pairs), 'Binary variable creation failed'

    profits = indexed.expected_profit.to_dict()
    spend = indexed.budget_cost.to_dict()
    exposures = indexed.expected_loss_exposure.to_dict()
    total_profit = pulp.lpSum(profits[k] * x[k] for k in pairs)
    total_spend = pulp.lpSum(spend[k] * x[k] for k in pairs)
    total_exposure = pulp.lpSum(exposures[k] * x[k] for k in pairs)
    model += total_profit, 'Expected_net_profit'
    for customer in customers:
        cust_pairs = [k for k in pairs if k[0] == customer]
        model += pulp.lpSum(x[k] for k in cust_pairs) == 1, f'One_offer_{customer}'
    model += total_spend <= BUDGET, 'Retention_budget'
    model += pulp.lpSum(x[k] for k in pairs if k[1] > 0) <= MAX_CONTACTS, 'Contact_capacity'
    baseline_exposure = float(candidates.loc[candidates.offer_pct.eq(0), 'expected_loss_exposure'].sum())
    model += total_exposure <= RISK_TOLERANCE * baseline_exposure, 'Loss_exposure_guardrail'
    result = model.solve(make_solver())
    status = getattr(result, 'status_str', None)
    if status is None:
        status = pulp.LpStatus[model.status]
    if status != 'Optimal':
        raise RuntimeError(f'No proven optimal solution (status={status}). Inspect model/solver log.')
    selected_pairs = [k for k in pairs if pulp.value(x[k]) > 0.5]
    assert len(selected_pairs) == len(customers)
    decisions = eligible.set_index(['customer_id', 'offer_pct']).loc[selected_pairs].reset_index()
    assert decisions.customer_id.is_unique
    assert float(decisions.budget_cost.sum()) <= BUDGET + 1e-4
    assert int(decisions.offer_pct.gt(0).sum()) <= MAX_CONTACTS
    assert float(decisions.expected_loss_exposure.sum()) <= RISK_TOLERANCE * baseline_exposure + 1e-4
    assert abs(float(decisions.expected_profit.sum()) - float(pulp.value(total_profit))) < 1e-4
    print('Solver:', status, '| Variables:', len(pairs), '| Constraints:', len(model.constraints()) if callable(model.constraints) else len(model.constraints))
    return decisions


def metrics(df):
    return {
        'expected_profit': round(float(df.expected_profit.sum()), 2),
        'expected_budget_spend': round(float(df.budget_cost.sum()), 2),
        'expected_renewals': round(float(df.p_renew.sum()), 2),
        'expected_loss_exposure': round(float(df.expected_loss_exposure.sum()), 2),
        'customers_contacted': int(df.offer_pct.gt(0).sum()),
    }


def main():
    # Synthetic "historical" randomised pilot split into development and holdout.
    pilot = simulate_pilot(18000, 'H')
    train = pilot.iloc[:14000].copy()
    valid = pilot.iloc[14000:].copy()
    response_model, claim_model = train_and_evaluate(train, valid)
    future = build_customers(800, 'F')
    candidates = score_scenarios(future, response_model, claim_model)
    candidates.to_csv(OUT / 'scored_customer_offer_scenarios.csv', index=False)
    decisions = solve_portfolio(candidates)
    decisions.to_csv(OUT / 'optimised_customer_offers.csv', index=False)
    baseline = candidates.loc[candidates.offer_pct.eq(0)].copy()
    baseline_metrics = metrics(baseline)
    optimal_metrics = metrics(decisions)
    summary = pd.DataFrame({'No discount': baseline_metrics,
                            'Optimised': optimal_metrics})
    summary['Change'] = summary['Optimised'] - summary['No discount']
    print('\nBusiness summary (model-estimated, NOT realised causal impact):\n')
    print(summary.to_string(float_format=lambda x: f'{x:,.2f}'))
    print('\nOffer distribution:\n', decisions.offer_pct.value_counts().sort_index().to_string())
    print('\nSaved CSVs to:', OUT)
    summary.to_csv(OUT / 'strategy_comparison.csv')
    # Production requirements: verified randomisation/propensity, calibration,
    # uncertainty, approved offer/pricing constraints, cost/revenue reconciliation,
    # contact exclusion rules, stable pipeline, experiment and live monitoring.


if __name__ == '__main__':
    main()
