import pandas as pd
import numpy as np

# ===== Read original results =====
# Read the Excel file containing MAE/MSE metrics and transition counts
df = pd.read_excel('../256mul/log_small_deeper2.xlsx')

# ===== Weighted error metrics =====
# Weighted Mean Absolute Error (MAE)
# Weighted by the number of valid samples in each row
weighted_mae = (df['mae_ms'] * df['total_valid']).sum() / df['total_valid'].sum()

# Weighted Mean Squared Error (MSE), then take sqrt to get RMSE
weighted_mse = (df['mse_ms'] * df['total_valid']).sum() / df['total_valid'].sum()
weighted_rmse = np.sqrt(weighted_mse)

# ===== Transition matrix statistics =====
# Columns representing all transitions from 1->1 to 5->5
transition_cols = [f"{i}_{j}" for i in range(1, 6) for j in range(1, 6)]

# Sum counts for each transition column
transition_sums = df[transition_cols].sum()

# Total number of transitions (sum of all valid transitions)
total_transitions = transition_sums.sum()

# Compute ratio (proportion) of each transition
transition_ratios = transition_sums / total_transitions

# ===== Construct summary DataFrame =====
summary = {
    "metric": ["weighted_mae_ms", "weighted_mse_ms", "weighted_rmse_ms"],
    "value": [weighted_mae, weighted_mse, weighted_rmse]
}
summary_df = pd.DataFrame(summary)

# Construct transition statistics DataFrame
transition_df = pd.DataFrame({
    "transition": transition_cols,
    "count": transition_sums.values,
    "ratio": transition_ratios.values
})

# ===== Save results to Excel with two sheets =====
with pd.ExcelWriter("analysis_summary_n.xlsx") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    transition_df.to_excel(writer, sheet_name="transitions", index=False)

print("Analysis completed. Results saved to analysis_summary_n.xlsx")
