import pandas as pd

# Beispiel-DataFrame
data = {
    # vergrößern für realistischere Speichernutzung
    'bool_col': [True, False, True, True] * 25000,
    'float_col': [1.0, 2.5, 3.5, 4.2] * 25000,
    'int_col': [1, 2, 3, 4] * 25000
}
df = pd.DataFrame(data)

# Speicherverbrauch vor der Optimierung
print("Vorher DataFrame:")
print(df.dtypes)
print(df.memory_usage(deep=True))

# Konvertierung der bool-Spalten zu uint8
for col in df.select_dtypes(include=['bool']).columns:
    df[col] = df[col].astype('uint8')

# Speicherverbrauch nach der Optimierung
print("\nNachher DataFrame:")
print(df.dtypes)
print(df.memory_usage(deep=True))
