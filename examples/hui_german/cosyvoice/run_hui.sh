#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=6
stop_stage=6

pretrained_model_dir=../../../pretrained_models/CosyVoice-300M

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"

  # Extrahiere die Listen aus split_speakers.json
  split_file="split_speakers.json"
  hui_train_spk=$(jq -r '.train[]' $split_file)
  hui_test_spk=$(jq -r '.test[]' $split_file)
  hui_dev_spk=$(jq -r '.val[]' $split_file)

  # Füge zusätzliche Sprecher zur train_spk-Liste hinzu
  additional_train_spk=("Bernd" "Eva" "Friedrich" "Hokus" "Karlsson")
  for spk in "${additional_train_spk[@]}"; do
    hui_train_spk="$hui_train_spk"$'\n'"$spk"
  done

  # echo "Train-Speakers: $hui_train_spk"
  # echo "Test-Speakers: $hui_test_spk"
  # echo "Dev-Speakers: $hui_dev_spk"

 for x in hui_train hui_dev hui_test; do
    mkdir -p data/$x
    # Wähle die richtige Sprecherliste basierend auf dem Dataset
    if [ "$x" == "hui_train" ]; then
        spk_list="$hui_train_spk"
    elif [ "$x" == "hui_dev" ]; then
        spk_list="$hui_dev_spk"
    elif [ "$x" == "hui_test" ]; then
        spk_list="$hui_test_spk"
    fi

    # Rufe das Python-Skript mit der entsprechenden Sprecherliste auf
    echo "$spk_list" > data/$x/speakers.txt
    python local/prepare_data.py --src_dir /resources/speech/corpora/HUI_German --des_dir data/$x --speakers data/$x/speakers.txt
done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in hui_train hui_dev hui_test; do
    tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in hui_train hui_dev hui_test; do
    tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v1.onnx
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in hui_train hui_dev hui_test; do
    mkdir -p data/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi

# inference
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  for mode in sft zero_shot; do
    python cosyvoice/bin/inference.py --mode $mode \
      --gpu 0 \
      --config conf/cosyvoice.yaml \
      --prompt_data data/test-clean/parquet/data.list \
      --prompt_utt2data data/test-clean/parquet/utt2data.list \
      --tts_text `pwd`/tts_text.json \
      --llm_model $pretrained_model_dir/llm.pt \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir `pwd`/exp/cosyvoice/test-clean/$mode
  done
fi

# train llm
export CUDA_VISIBLE_DEVICES="7,8"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat data/hui_train/parquet/data.list > data/train.data.list
  cat data/hui_dev/parquet/data.list > data/dev.data.list
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp_hui/cosyvoice/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice_hui/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm; do
    decode_checkpoint=`pwd`/exp_hui/cosyvoice/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp_hui/cosyvoice/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi