import numpy as np
import pandas as pd

# Parameters from Part (a)
beta0, beta1 = 266, -7
c = 8
feasible_prices = [18, 19, 20, 21, 22]

# Scenario definitions
scenarios = {
    'Boom': {'prob': 0.25, 'shift': 0.20},
    'Normal': {'prob': 0.50, 'shift': 0.00},
    'Downturn': {'prob': 0.25, 'shift': -0.25}
}

# Compute profits for each price
results = []
for p in feasible_prices:
    base_demand = beta0 + beta1 * p
    scenario_profits = {}
    for name, data in scenarios.items():
        adj_demand = base_demand * (1 + data['shift'])
        scenario_profits[name] = (p - c) * adj_demand

    expected_profit = sum(scenario_profits[n] * scenarios[n]['prob'] for n in scenarios)
    worst_case = min(scenario_profits.values())

    results.append({
        'Price': p,
        **scenario_profits,
        'Expected Profit': expected_profit,
        'Worst-Case Profit': worst_case
    })

df = pd.DataFrame(results)
print("Scenario Analysis Table:")
print(df.round(2).to_string(index=False))

# Identify optimal prices
risk_neutral_p = df.loc[df['Expected Profit'].idxmax(), 'Price']
robust_p = df.loc[df['Worst-Case Profit'].idxmax(), 'Price']

print(f"\n(i) Risk-Neutral Optimal Price: ${risk_neutral_p}")
print(f"(ii) Robust Optimal Price: ${robust_p}")