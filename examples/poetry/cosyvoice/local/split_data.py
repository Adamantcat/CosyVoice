from collections import defaultdict
import random
import os
from tqdm import tqdm
import pandas as pd

def split_wavs(wavs, candidates, train_ratio=0.8, dev_ratio=0.1):
    # Gruppiere WAV-Dateien nach lccn
    lccn_to_wavs = defaultdict(list)
    for wav in wavs:
        lccn = os.path.basename(wav).split("_")[0]  # Extrahiere lccn aus dem Dateinamen
        lccn_to_wavs[lccn].append(wav)

    # Trenne lccn in Kandidaten und Nicht-Kandidaten
    # candidate_lccns = set(candidates)
    candidate_wavs = {lccn: wav_list for lccn, wav_list in lccn_to_wavs.items() if lccn in candidates}
    non_candidate_wavs = {lccn: wav_list for lccn, wav_list in lccn_to_wavs.items() if lccn not in candidates}

    # print(sum(len(wav_list) for wav_list in candidate_wavs.values()))
    # print(sum(len(wav_list) for wav_list in non_candidate_wavs.values()))

    # Shuffle die lccn
    candidate_lccns = list(candidate_wavs.keys())
    random.shuffle(candidate_lccns)

    # Berechne die Gesamtanzahl der WAV-Dateien (Kandidaten + Nicht-Kandidaten)
    total_wavs = sum(len(wav_list) for wav_list in lccn_to_wavs.values())
    dev_size = int(total_wavs * dev_ratio)
    test_size = int(total_wavs * dev_ratio)  # Test und Dev haben das gleiche Verhältnis
    train_size = total_wavs - dev_size - test_size

    # Teile die Kandidaten-WAVs in Train, Dev und Test auf
    train_wavs = []
    dev_wavs = []
    test_wavs = []

    current_count = 0

    # Füge WAV-Dateien aus Nicht-Kandidaten zu Train hinzu
    for wav_list in non_candidate_wavs.values():
        train_wavs.extend(wav_list)
        current_count += len(wav_list)

    for lccn, wav_list in candidate_wavs.items():
        if current_count < train_size:
            train_wavs.extend(wav_list)
        elif current_count < train_size + dev_size:
            dev_wavs.extend(wav_list)
        else:
            test_wavs.extend(wav_list)
        current_count += len(wav_list)

    return train_wavs, dev_wavs, test_wavs

def main():
    input_file = '/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/examples/poetry/cosyvoice/data.json'
    src_dir = "/mount/arbeitsdaten/textklang/synthesis/Multispeaker_PoeticTTS_Data/Sprechweisen"
    data_df = pd.read_json(input_file)
    # Lade die Daten und Kandidaten
   # get lccn of test candidates
    with open("test_candidates.txt", "r") as f:
        candidates = []
        for line in f:
            title_long = line.split(";")[0].strip()  # Extrahiere den Titel
            matching_row = data_df[data_df['title_long'] == title_long]  # Finde die Zeile mit übereinstimmendem Titel
            if not matching_row.empty:
                lccn = matching_row.iloc[0]['lccn']  # Extrahiere den lccn-Wert
                candidates.append(lccn)  # Füge den lccn-Wert zur Liste hinzu
        # print(candidates)

    # Sammle alle WAV-Dateien
    wavs_in_dir = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.wav'):
                wavs_in_dir.append(os.path.join(file))

    # Filtere WAV-Dateien, die in data.json vorkommen
    wavs = []
    for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
        file_name = row['lccn'] + "_" + str(row['strophe']) + ".wav"
        if file_name in wavs_in_dir:
            wavs.append(os.path.join(file_name))

    # Splitte die Daten
    print(len(wavs))
    train_wavs, dev_wavs, test_wavs = split_wavs(wavs, candidates)
    print(len(train_wavs), len(dev_wavs), len(test_wavs))

    # Speichere die Splits
    with open("train_wavs.txt", "w") as f:
        f.writelines("\n".join(train_wavs))
    with open("dev_wavs.txt", "w") as f:
        f.writelines("\n".join(dev_wavs))
    with open("test_wavs.txt", "w") as f:
        f.writelines("\n".join(test_wavs))

    print(f"Train: {len(train_wavs)}, Dev: {len(dev_wavs)}, Test: {len(test_wavs)}")


if __name__ == "__main__":
    main()