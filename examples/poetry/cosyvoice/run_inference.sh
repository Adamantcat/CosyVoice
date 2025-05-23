. ./path.sh || exit 1;

stage=1
stop_stage=1

# make tts_text.josn
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge  0 ]; then

#!/bin/bash

# Eingabedatei
input_file="/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/examples/poetry/cosyvoice/data/test/text"
# Ausgabedatei
output_file="/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/examples/poetry/cosyvoice/data/test/tts_text.json"

# JSON-Start
echo "{" > "$output_file"

# Zeilenweise Verarbeitung der Eingabedatei
while IFS= read -r line; do
    # Extrahiere den Dateinamen (Schlüssel) und den Text (Wert)
    key=$(echo "$line" | awk '{print $1}')
    value=$(echo "$line" | cut -d' ' -f2-)

    # Escape von Anführungszeichen im Wert
    value=$(echo "$value" | sed 's/"/\\"/g')

    # Füge den Eintrag zur JSON-Datei hinzu
    echo "  \"$key\": \"$value\"," >> "$output_file"
done < "$input_file"

# Entferne das letzte Komma und schließe die JSON-Datei
sed -i '$ s/,$//' "$output_file"
echo "}" >> "$output_file"

echo "JSON-Datei wurde erstellt: $output_file"

fi

# inference
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M-Instruct
llm_dir=/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/examples/hui_german/cosyvoice/exp_hui_instruct/cosyvoice/llm/torch_ddp
result_dir=../../../Evaluation/eval_audios/CosyVoice_hui_instruct_41/poetry_test
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  for mode in sft zero_shot crosslingual; do
    python cosyvoice/bin/inference.py --mode $mode \
      --gpu 2 \
      --config conf/cosyvoice.yaml \
      --prompt_data data/test/parquet/data.list \
      --prompt_utt2data data/test/parquet/utt2data.list \
      --tts_text `pwd`/tts_text.json \
      --llm_model $llm_dir/llm_hui_instruct_41.pt \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir $result_dir/$mode
  done
fi
