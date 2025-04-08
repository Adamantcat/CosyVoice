#!/bin/bash

# Verzeichnis, in dem die Suche gestartet werden soll
parent_directory="/mount/arbeitsdaten/textklang/synthesis/Multispeaker_PoeticTTS_Data/Sprechweisen"

# Zähle die Anzahl der WAV-Dateien, die länger als 30 Sekunden sind
count=$(find "$parent_directory" -type f -name "*.wav" -exec soxi -D {} \; | awk '$1 > 30' | wc -l)

echo "Anzahl der WAV-Dateien, die länger als 30 Sekunden sind: $count"