import matplotlib.pyplot as plt
import numpy as np
import os
from loguru import logger

# Daten aus dem classification_report
categories = [
    "Betrugsdelikte", "Cyberkriminalität", "Eigentumsdelikte", "Gewaltverbrechen",
    "Jugendkriminalität", "Kindesmisshandlung", "Ordnungswidrigkeiten", "Sexualdelikte",
    "Sonstiges", "Tierschutzverbrechen", "Umweltkriminalität", "Verkehrsdelikte"
]

precision = [0.42, 0.00, 0.44, 0.42, 0.00,
             0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.37]


x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))
rects1 = ax.bar(x - width, precision, width, label='Precision')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Verbrechenskategorie')
ax.set_ylabel('Precision')
# ax.set_title('Precision jeder Verbrechenskategorie vom Decision Tree')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha="right")
ax.legend()

fig.tight_layout()

os.makedirs('Plots', exist_ok=True)
plt.savefig('Plots/performance_forest_all_var.png')
logger.info(f"Saved plot to Plots/performance_tree_all_var.png")
