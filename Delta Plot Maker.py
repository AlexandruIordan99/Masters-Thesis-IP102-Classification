import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the table
models = ["B0", "B1", "B2", "B3", "B4", "V2B0", "V2B1", "V2B2", "V2B3", "V2S"]

f1_baseline = [44.73,
               44.22,
               45.87,
               48.51,
               47.15,
               44.95,
               46.86,
               46.48,
               47.56,
               49.06
               ]
AUC_baseline =[82.33,
               82.28,
               83.22,
               83.85,
               85.25,
               82.52,
               83.93,
               83.83,
               84.63,
               84.66
               ]


f1_data_aug =[65.66,
              65.72,
              66.51,
              70.06,
              70.96,
              64.33,
              65.04,
              66.08,
              67.17,
              68.23
              ]
AUC_data_aug=[96.13,
              95.56,
              95.90,
              96.15,
              96.38,
              95.98,
              96.02,
              95.97,
              96.32,
              96.32,
              ]


# Calculate delta (BGR - RGB)
delta_f1 = np.array(f1_data_aug) - np.array(f1_baseline)
delta_AUC = np.array(AUC_data_aug) - np.array(AUC_baseline)
# Plot as a line plot
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(models))

ax.plot(models, delta_f1, marker='o', color='skyblue', label="Delta F1 (Data Augmentation - Baseline)")
# ax.plot(models, delta_accuracy, marker='o', color='red', label="Delta Accuracy, F1, AUC (Data Augmentation - Baseline)")
ax.plot(models, delta_AUC, marker='o',  color='green', label="Delta AUC (Data Augmentation - Baseline")

# Add labels and title
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Delta F1 Score", fontsize=12)
ax.set_ylim([0, 30])
ax.set_title("Delta in F1, AUC between Data Augmentation - Baseline  Models", fontsize=14)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.legend()

# Annotate the deltas on the points
for i, v in enumerate(delta_f1):
    ax.text(i, v + 0.1 if v >= 0 else v - 0.2, f"{v:.2f}", ha='center', va='bottom' if v >= 0 else 'top')

for i, v in enumerate(delta_AUC):
    ax.text(i, v + 0.1 if v >= 0 else v - 0.2, f"{v:.2f}", ha='center', va='bottom' if v >= 0 else 'top')


plt.tight_layout()
plt.show()
