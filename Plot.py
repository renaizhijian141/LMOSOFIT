"""
3D plot: Physics-guided model (h_cr^2, 1/k_cr, H^2 + interaction terms)
Variables: k_cr (thermal conductivity), h_cr (crustal thickness), H (heat production)
Response: t (LMO cooling time)
Surfaces: H = 10, 15, 20, 25
Data points: all 264, color-coded by H
Includes legend and colorbar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Rectangle

# ================== 1. Load data and fit model ==================
file_path = '04-Table S5.xlsx'  # modify to your path
df = pd.read_excel(file_path, sheet_name='Table S5', header=1)
df.columns = ['k_cr', 'h_cr', 'H', 't']   # rename columns
X_raw = df[['k_cr', 'h_cr', 'H']].values
y = df['t'].values

# Construct physics-guided design matrix
k_cr = X_raw[:, 0]
h_cr = X_raw[:, 1]
H = X_raw[:, 2]

inv_k_cr = 1.0 / k_cr
h_cr2 = h_cr**2
H2 = H**2
h_cr_inv_k_cr = h_cr * inv_k_cr
h_cr_H = h_cr * H
inv_k_cr_H = inv_k_cr * H

X_design = np.column_stack([h_cr, h_cr2, inv_k_cr, H, H2, h_cr_inv_k_cr, h_cr_H, inv_k_cr_H])

# Fit model
reg = LinearRegression(fit_intercept=True)
reg.fit(X_design, y)
intercept = reg.intercept_
coefs = reg.coef_

print("Fitted coefficients:")
print(f"Intercept: {intercept:.6f}")
names = ['h_cr', 'h_cr^2', '1/k_cr', 'H', 'H^2', 'h_cr*(1/k_cr)', 'h_cr*H', '(1/k_cr)*H']
for name, c in zip(names, coefs):
    print(f"{name}: {c:.6f}")

# ================== 2. Define prediction function ==================
def predict_t(h_cr_val, k_cr_val, H_val):
    """Return predicted t for given h_cr, k_cr, H"""
    inv_k_val = 1.0 / k_cr_val
    h_cr2_val = h_cr_val**2
    H2_val = H_val**2
    h_cr_inv_k_val = h_cr_val * inv_k_val
    h_cr_H_val = h_cr_val * H_val
    inv_k_H_val = inv_k_val * H_val
    features = np.array([h_cr_val, h_cr2_val, inv_k_val, H_val, H2_val,
                         h_cr_inv_k_val, h_cr_H_val, inv_k_H_val])
    return intercept + np.dot(coefs, features)

# ================== 3. Generate plotting data ==================
H_levels = [10, 15, 20, 25]
cmap = cm.viridis
norm = colors.Normalize(vmin=min(H_levels), vmax=max(H_levels))

h_cr_grid = np.linspace(39, 49, 40)
k_cr_grid = np.linspace(1.5, 2.0, 40)
H_grid, K_grid = np.meshgrid(h_cr_grid, k_cr_grid)

# Store predicted surfaces for each H
surfaces = {}
for H_val in H_levels:
    T_surf = np.zeros_like(H_grid)
    for i in range(H_grid.shape[0]):
        for j in range(H_grid.shape[1]):
            T_surf[i, j] = predict_t(H_grid[i, j], K_grid[i, j], H_val)
    surfaces[H_val] = T_surf

# ================== 4. Plotting ==================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# ---- Plot four surfaces (without label, will add legend manually) ----
for H_val in H_levels:
    surf = ax.plot_surface(H_grid, K_grid, surfaces[H_val],
                           alpha=0.5,
                           color=cmap(norm(H_val)),
                           edgecolor='none')

# ---- Plot data points (all 264) ----
point_colors = cmap(norm(df['H']))
sc = ax.scatter(df['h_cr'], df['k_cr'], df['t'],
                c=point_colors, s=50, edgecolors='k', linewidth=0.5,
                alpha=0.85, label='Model data (colored by H)', zorder=10)

# ---- Colorbar ----
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(H_levels)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('Heat production (pW/kg)', fontsize=12)

# ---- Create proxy legend items (surface color patches) ----
legend_elements = []
for H_val in H_levels:
    color = cmap(norm(H_val))
    legend_elements.append(Rectangle((0,0), 1, 1, fc=color, alpha=0.6,
                                     label=f'H = {H_val} pW/kg surface'))

# Add data point legend
legend_elements.append(sc)

# Add legend (upper left, avoid overlapping with colorbar)
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# ---- Axis labels and title ----
ax.set_xlabel('Crustal thickness (km)', fontsize=12)
ax.set_ylabel('Thermal conductivity (W/m/K)', fontsize=12)
ax.set_zlabel('LMO cooling time (Myr)', fontsize=12)
ax.set_title('Physics-guided model: surfaces for different H with observed data', fontsize=14)

# ---- View angle ----
ax.view_init(elev=6, azim=-174)

plt.tight_layout()
plt.savefig('Physical_model_3D_surfaces_with_legend.png', dpi=300, bbox_inches='tight')
plt.show()