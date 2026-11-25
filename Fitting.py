"""
Physics-guided model (H² model) fitting + comprehensive sensitivity analysis
Includes: marginal effects at the mean, Average Marginal Effects (AME),
          and stratified marginal effects for H = 10, 15, 20, 25
Data: all 264 samples
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ================== 1. Load data ==================
file_path = '04-Table S5.xlsx'  # modify to your actual path
df = pd.read_excel(file_path, sheet_name='Table S5', header=1)
df.columns = ['k_cr', 'h_cr', 'H', 't']   # rename columns to new variable names

X_raw = df[['k_cr', 'h_cr', 'H']].values
y = df['t'].values
n = len(y)
print(f"Total samples: {n}")

# ================== 2. Construct physics-guided design matrix ==================
k_cr = X_raw[:, 0]
h_cr = X_raw[:, 1]
H = X_raw[:, 2]

inv_k_cr = 1.0 / k_cr
h_cr2 = h_cr**2
H2 = H**2
h_cr_inv_k_cr = h_cr * inv_k_cr
h_cr_H = h_cr * H
inv_k_cr_H = inv_k_cr * H

X_design = np.column_stack([
    h_cr, h_cr2, inv_k_cr,
    H, H2,
    h_cr_inv_k_cr,
    h_cr_H,
    inv_k_cr_H
])
feature_names = ['h_cr', 'h_cr^2', '1/k_cr', 'H', 'H^2', 'h_cr*(1/k_cr)', 'h_cr*H', '(1/k_cr)*H']

# ================== 3. Fit the model ==================
reg = LinearRegression(fit_intercept=True)
reg.fit(X_design, y)
intercept = reg.intercept_
coefs = reg.coef_
y_pred = reg.predict(X_design)

# ================== 4. Statistical metrics ==================
r2 = r2_score(y, y_pred)
p = X_design.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
residuals = y - y_pred
mae = np.abs(residuals).mean()
rmse = np.sqrt((residuals**2).mean())
rel_err = np.abs(residuals / y).mean() * 100

print("\n=== Fit Statistics ===")
print(f"R²          : {r2:.6f}")
print(f"Adjusted R² : {adj_r2:.6f}")
print(f"MAE (Ma)    : {mae:.4f}")
print(f"RMSE (Ma)   : {rmse:.4f}")
print(f"Mean relative error: {rel_err:.4f} %")

# ================== 5. Output the fitted formula ==================
formula = f"t = {intercept:.6f}"
for name, coef in zip(feature_names, coefs):
    if abs(coef) > 1e-8:
        formula += f" + ({coef:+.6f})*{name}"
print("\n=== Fitted formula ===")
print(formula)

print("\nCoefficient details:")
print(f"{'Variable':<18} {'Coefficient':>15} {'Absolute':>15}")
for name, coef in zip(feature_names, coefs):
    print(f"{name:<18} {coef:15.6f} {abs(coef):15.6f}")

# ================== 6. Residual statistics ==================
print("\n=== Residual Statistics ===")
print(f"Residual std dev: {np.std(residuals):.4f} Myr")
print(f"Max positive residual: {residuals.max():.4f} Myr")
print(f"Max negative residual: {residuals.min():.4f} Myr")

# ================== 7. Basic sensitivity analysis (marginal effects at the mean) ==================
means = X_raw.mean(axis=0)
stds = X_raw.std(axis=0)
var_names = ['k_cr', 'h_cr', 'H']

b1, b2, b3, b4, b5, b6, b7, b8 = coefs
k0, h0, H0 = means[0], means[1], means[2]
inv_k0 = 1.0 / k0

# Partial derivatives
dt_dk = -b3 * (inv_k0**2) - b6 * h0 * (inv_k0**2) - b8 * (inv_k0**2) * H0
dt_dh = b1 + 2*b2*h0 + b6*inv_k0 + b7*H0
dt_dH = b4 + 2*b5*H0 + b7*h0 + b8*inv_k0

# Marginal effects
marginal_k = dt_dk * stds[0]
marginal_h = dt_dh * stds[1]
marginal_H = dt_dH * stds[2]

print("\n=== Basic Sensitivity Analysis (Marginal Effects at the Mean) ===")
print(f"At mean point (k_cr={k0:.3f}, h_cr={h0:.3f}, H={H0:.3f}):")
print(f"∂t/∂k_cr = {dt_dk:.6f}  (Myr/(W/m/K))  → Marginal effect = {marginal_k:.6f} Ma")
print(f"∂t/∂h_cr = {dt_dh:.6f}  (Myr/km)       → Marginal effect = {marginal_h:.6f} Ma")
print(f"∂t/∂H    = {dt_dH:.6f}  (Myr/(pW/kg))  → Marginal effect = {marginal_H:.6f} Ma")

abs_marg = [abs(marginal_k), abs(marginal_h), abs(marginal_H)]
most_sensitive = var_names[np.argmax(abs_marg)]


# ================== 8. Advanced sensitivity analysis ==================
print("\n" + "="*60)
print("【Advanced Sensitivity Analysis】")

# ---- 8.1 Average Marginal Effect (AME) ----
k_all, h_all, H_all = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2]
inv_k_all = 1.0 / k_all

dt_dk_all = -b3 * (inv_k_all**2) - b6 * h_all * (inv_k_all**2) - b8 * (inv_k_all**2) * H_all
dt_dh_all = b1 + 2*b2*h_all + b6*inv_k_all + b7*H_all
dt_dH_all = b4 + 2*b5*H_all + b7*h_all + b8*inv_k_all

ame_dk = np.mean(dt_dk_all)
ame_dh = np.mean(dt_dh_all)
ame_dH = np.mean(dt_dH_all)

ame_marginal_k = ame_dk * stds[0]
ame_marginal_h = ame_dh * stds[1]
ame_marginal_H = ame_dH * stds[2]

print("\n【Average Marginal Effect (AME)】—— reflects average sensitivity over the entire parameter space")
print(f"∂t/∂k_cr (mean) = {ame_dk:.6f}  → AME marginal effect = {ame_marginal_k:.6f} Myr")
print(f"∂t/∂h_cr (mean) = {ame_dh:.6f}  → AME marginal effect = {ame_marginal_h:.6f} Myr")
print(f"∂t/∂H    (mean) = {ame_dH:.6f}  → AME marginal effect = {ame_marginal_H:.6f} Myr")
ame_sens = var_names[np.argmax([abs(ame_marginal_k), abs(ame_marginal_h), abs(ame_marginal_H)])]


# ---- 8.2 Stratified marginal effects (fixed H levels) ----
print("\n【Stratified Marginal Effects】—— sensitivity variation at different H levels (h_cr, k_cr fixed at their means)")
H_levels = [10, 15, 20, 25]

for H_val in H_levels:
    dt_dk_H = -b3 * (inv_k0**2) - b6 * h0 * (inv_k0**2) - b8 * (inv_k0**2) * H_val
    dt_dh_H = b1 + 2*b2*h0 + b6*inv_k0 + b7*H_val
    dt_dH_H = b4 + 2*b5*H_val + b7*h0 + b8*inv_k0

    marg_k_H = dt_dk_H * stds[0]
    marg_h_H = dt_dh_H * stds[1]
    marg_H_H = dt_dH_H * stds[2]

    print(f"\nWhen H = {H_val}:")
    print(f"  ∂t/∂k_cr = {dt_dk_H:.4f}  → marginal effect = {marg_k_H:.4f} Myr")
    print(f"  ∂t/∂h_cr = {dt_dh_H:.4f}  → marginal effect = {marg_h_H:.4f} Myr")
    print(f"  ∂t/∂H    = {dt_dH_H:.4f}  → marginal effect = {marg_H_H:.4f} Myr")

    abs_vals_H = [abs(marg_k_H), abs(marg_h_H), abs(marg_H_H)]
    sens_var_H = var_names[np.argmax(abs_vals_H)]


