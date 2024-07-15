import re
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


# Beispiel-Datenframe erstellen
data = {'Address': ['1100 Oakstreet BL', '1234 Elmstreet DR',
                    '5678 Maplestreet BL', '1110 Oakstreet']}
df = pd.DataFrame(data)

# Funktion, um die letzten zwei Buchstaben einer Zeichenkette zu extrahieren, falls vorhanden


def extract_last_two_letters(address):
    match = re.search(r'( [A-Za-z]{2})$', address)
    if match:
        return match.group(1)
    return None


# Diese Funktion auf die Spalte 'Address' anwenden und das Ergebnis in einer neuen Spalte speichern
df['LastTwoChars'] = df['Address'].apply(extract_last_two_letters)

# Den modifizierten DataFrame anzeigen
print(df)
