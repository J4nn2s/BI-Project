import matplotlib.pyplot as plt
import numpy as np
import os
# Daten aus dem classification_report
categories = [
    "Betrugsdelikte", "Cyberkriminalität", "Eigentumsdelikte", "Gewaltverbrechen",
    "Jugendkriminalität", "Kindesmisshandlung", "Ordnungswidrigkeiten", "Sexualdelikte",
    "Sonstiges", "Tierschutzverbrechen", "Umweltkriminalität", "Verkehrsdelikte"
]

precision = [0.59, 0.00, 0.53, 0.53, 0.00,
             0.00, 0.00, 0.36, 0.00, 0.00, 0.00, 0.43]


x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))
rects1 = ax.bar(x - width, precision, width, label='Precision')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Verbrechenskategorie')
ax.set_ylabel('Scores')
ax.set_title('Classification Report Metriken nach Verbrechenskategorie')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha="right")
ax.legend()

fig.tight_layout()

os.makedirs('Plots', exist_ok=True)
plt.savefig('Plots/performance_tree_all_var.png')
print('CV Tree Results saved')
