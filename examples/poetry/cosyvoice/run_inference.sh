
src_dir=.
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M_poetry
result_dir=eval_audios/poetry_dev/CosyVoice-300M_poetry_backup

echo "Run inference. Please make sure utt in tts_text is in prompt_data"
for mode in sft zero_shot; do
python cosyvoice/bin/inference.py --mode $mode \
    --gpu 1 \
    --config $src_dir/conf/cosyvoice.yaml \
    --prompt_data $src_dir/cosyvoice/data/poetry_test_data/parquet/data.list \
    --prompt_utt2data $src_dir/cosyvoice/data/poetry_test_data/parquet/utt2data.list \
    --tts_text `pwd`/tts_text_poetry.json \
    --llm_model $pretrained_model_dir/llm_backup.pt \
    --flow_model $pretrained_model_dir/flow.pt \
    --hifigan_model $pretrained_model_dir/hift.pt \
    --result_dir $result_dir/$mode
done