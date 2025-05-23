import json

# Pfad zur JSON-Datei
data_file_path = "/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/examples/poetry/cosyvoice/data.json"

# JSON-Datei laden
with open(data_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

unique_data = {item.get("title_long", "lccn").lower().strip(): item for item in data}.values()
sorted_data = sorted(unique_data, key=lambda x: x.get("title_long", "lccn").lower())

with open("titles.txt", "w", encoding="utf-8") as f:
    for item in sorted_data:
        f.write(item['title_long'].strip() + "\n")