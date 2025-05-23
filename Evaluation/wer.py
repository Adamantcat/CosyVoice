import whisper
import soundfile as sf
import os
import torch
from torcheval.metrics import WordErrorRate
import string
import pickle



asr = whisper.load_model("medium", download_root='/mount/arbeitsdaten/textklang/synthesis/Whisper')
options = whisper.DecodingOptions(language="de")


def transcribe(eval_audios, savefile=None):
    asr_transcripts = {}
    for audio in eval_audios:
        asr_transcripts[audio] = asr.transcribe(audio, language="de")
    if savefile:
        pickle.dump(asr_transcripts, open(savefile, 'wb'))
        # with open(savefile, 'w') as f:
        #     for key in asr_transcripts.keys():
        #         f.write(f"{key}\t{asr_transcripts[key]['text']}\n")
    return asr_transcripts


# def wer(predicted, reference):
#     metric = WordErrorRate()
#     metric.update(predicted, reference)
#     return metric.compute()

def wer(predicted, reference):
    """Calculate WER for each text pair and return individual and total scores."""
    metric = WordErrorRate()
    individual_scores = []

    # Calculate WER for each pair
    for pred, ref in zip(predicted, reference):
        metric.update([pred], [ref])  # Update metric for the current pair
        individual_scores.append(metric.compute())  # Compute WER for the pair
        metric.reset()

    # Compute total WER
    metric.update(predicted, reference)  # Update metric for all pairs
    total_wer = metric.compute()

    return individual_scores, total_wer

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    # audio_dir = "/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/Evaluation/eval_audios/CosyVoice-300M_poetry_backup/poetry_dev/sft"
    audio_dir = "/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/Evaluation/eval_audios/CosyVoice_hui_instruct_41/poetry_test/zero_shot"
    text_file = "/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/examples/poetry/cosyvoice/data/test/text"
    eval_audios = []
    gold_transcripts = {}

    # savefile = audio_dir + '/asr_transcripts.pt'

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
    
    savefile = audio_dir + '/asr_transcripts.pt'
    if not os.path.exists(savefile):
        asr_transcripts = transcribe(eval_audios, savefile)
        print("Save transcript to ", savefile)
    else:
        asr_transcripts = pickle.load(open(savefile, 'rb'))
        print("Load transcript from ", savefile)

    predicted = []
    reference = []
    for key in asr_transcripts.keys():
        pred_text = asr_transcripts[key]['text'].lower().strip().translate(str.maketrans('', '', string.punctuation))
        ref_text = gold_transcripts[key].lower().strip().translate(str.maketrans('', '', string.punctuation))
        predicted.append(pred_text)
        reference.append(ref_text)

    # Calculate WER
    individual_scores, total_wer = wer(predicted, reference)
    results_file = os.path.join(audio_dir, 'wer_results.txt')
    with open(results_file, 'w') as f:
        for i, (pred, ref, score) in enumerate(zip(predicted, reference, individual_scores)):
            print(f"Pair {i + 1}:\n")
            print(f"  Predicted: {pred}")
            print(f"  Reference: {ref}")
            print(f"  WER: {score:.4f}")

            f.write(f"Pair {i + 1}:\n")
            f.write(f"  Predicted: {pred}\n")
            f.write(f"  Reference: {ref}\n")
            f.write(f"  WER: {score:.4f}\n")
        print(f"Total WER: {total_wer:.4f}")
        f.write(f"Total WER: {total_wer:.4f}\n")
   
        

    