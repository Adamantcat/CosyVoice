import os
import random
import json

src_dir = '/resources/speech/corpora/HUI_German/others'

files_per_spk = {}

for spk in os.listdir(src_dir):
    if spk == "pttd_cache.pt":
        continue
    num_files = 0
    for root, dirs, files in os.walk(os.path.join(src_dir, spk)):
        for file in files:
            if file.endswith('.wav'):
                if not file.endswith('synthetic.wav'):
                    num_files += 1
    files_per_spk[spk] = num_files


# Gesamtsumme der Dateien
total_files = sum(files_per_spk.values())
target_test_files = int(total_files * 0.1)  # Ziel: ca. 10% der Daten für Test
target_val_files = int(total_files * 0.1)  # Ziel: ca. 10% der Daten für Validierung

# Zufällige Auswahl der Sprecher für Test- und Validierungsdatensatz
test_spk = []
val_spk = []
test_files_count = 0
val_files_count = 0

speakers = list(files_per_spk.keys())
random.shuffle(speakers)  # Zufällige Reihenfolge der Sprecher

# Sprecher für Testdaten auswählen
for spk in speakers:
    if test_files_count >= target_test_files:
        break
    test_spk.append(f"others/{spk}")
    test_files_count += files_per_spk[spk]

# Sprecher für Validierungsdaten auswählen
remaining_speakers = [spk for spk in speakers if spk not in test_spk]
for spk in remaining_speakers:
    if val_files_count >= target_val_files:
        break
    val_spk.append(f"others/{spk}")
    val_files_count += files_per_spk[spk]

# Restliche Sprecher für Trainingsdaten
train_spk = [f"others/{spk}" for spk in speakers if spk not in test_spk and spk not in val_spk]

# Ergebnisse ausgeben
print("Test-Sprecher:", test_spk)
print("Validierungs-Sprecher:", val_spk)
print("Trainings-Sprecher:", train_spk)
print(f"Testdaten: {test_files_count} Dateien (~{test_files_count / total_files:.2%})")
print(f"Validierungsdaten: {val_files_count} Dateien (~{val_files_count / total_files:.2%})")
print(f"Trainingsdaten: {total_files - test_files_count - val_files_count} Dateien (~{(total_files - test_files_count - val_files_count) / total_files:.2%})")

with open("split_speakers.json", "w", encoding='UTF-8') as f:
    json.dump({"train": train_spk, "val": val_spk, "test": test_spk}, f)