import pandas as pd
import numpy as np

# ===== Read original results =====
# Load the Excel file containing MAE/MSE metrics and valid sample counts
df = pd.read_excel('../256_Full/log_onset.xlsx') # 256_Full only

# ===== Compute weighted error metrics =====
# Weighted Mean Absolute Error (MAE)
# Weighted by the number of valid samples in each row
weighted_mae = (df['mae_ms'] * df['total_valid']).sum() / df['total_valid'].sum()

# Weighted Mean Squared Error (MSE)
# Then take the square root to obtain the weighted RMSE
weighted_mse = (df['mse_ms'] * df['total_valid']).sum() / df['total_valid'].sum()
weighted_rmse = np.sqrt(weighted_mse)

# ===== Construct summary DataFrame =====
summary = {
    "metric": ["weighted_mae_ms", "weighted_mse_ms", "weighted_rmse_ms"],
    "value": [weighted_mae, weighted_mse, weighted_rmse]
}
summary_df = pd.DataFrame(summary)

# ===== Save results to Excel =====
# The results are saved to a single sheet named "summary"
with pd.ExcelWriter("analysis_summary_onset.xlsx") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)

print("Analysis completed. Results saved to analysis_summary_onset.xlsx")
