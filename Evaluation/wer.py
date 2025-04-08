import whisper
import soundfile as sf
import os
import torch
from torcheval.metrics import WordErrorRate
import string



asr = whisper.load_model("large-v3", download_root='/mount/arbeitsdaten/textklang/synthesis/Whisper')
options = whisper.DecodingOptions(language="de")


def transcribe(eval_audios):
    asr_transcripts = {}
    for audio in eval_audios:
        asr_transcripts[audio] = asr.transcribe(audio, language="de")
    return asr_transcripts


def wer(predicted, reference):
    metric = WordErrorRate()
    metric.update(predicted, reference)
    return metric.compute()


if __name__ == "__main__":
    # audio_dir = "/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/Evaluation/eval_audios/CosyVoice-300M_poetry_backup/poetry_dev/sft"
    audio_dir = "/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/Evaluation/eval_audios/CosyVoice-300M_poetry_hui_100/poetry_dev/sft"
    text_file = "/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/examples/poetry/cosyvoice/data/poetry_test_data/text"
    eval_audios = []
    gold_transcripts = {}
    with open(text_file, 'r') as f:
        for line in f:
            line = line.strip()
            audio_id, transcript = line.split(' ', 1)
            wav_path = audio_id.split('.')[0] 
            wav_path = os.path.join(audio_dir, f"{wav_path}_0.wav") # tts inference adds a 0 to the name of the audio file, so we need to add it here as well
            gold_transcripts[os.path.join(audio_dir, wav_path)] = transcript
    for audio in os.listdir(audio_dir):
        if not audio.endswith('.wav'):
            continue
        wav_path = os.path.join(audio_dir, audio)
        eval_audios.append(wav_path)
    asr_transcripts = transcribe(eval_audios)

    predicted = []
    reference = []
    for key in asr_transcripts.keys():
        pred_text = asr_transcripts[key]['text'].lower().strip().translate(str.maketrans('', '', string.punctuation))
        ref_text = gold_transcripts[key].lower().strip().translate(str.maketrans('', '', string.punctuation))
        predicted.append(pred_text)
        reference.append(ref_text)
    print(predicted)
    wer_score = wer(predicted, reference)
    print(wer_score)