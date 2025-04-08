import argparse
import logging
import glob
import os
from tqdm import tqdm
import pandas as pd


logger = logging.getLogger()


def main():

    with open(args.speakers, "r") as file:
        speakers = file.readlines()
    speakers = [s.strip() for s in speakers]

    print(speakers)
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    for speaker in speakers:
        print("speaker ", speaker)
        dirs = os.listdir(os.path.join(args.src_dir, speaker)) # book titles
        print("dirs ", dirs)
        for d in dirs:
            if d == "pttd_cache.pt":
                continue
            with open(os.path.join(args.src_dir, speaker, d, "metadata.csv"), "r", encoding="utf8") as file:
                lookup = file.read()
            for line in lookup.split("\n"):
                if line.strip() != "":
                    utt, text = line.split("|")
                    wav_path = os.path.join(args.src_dir, speaker, d, "wavs", line.split("|")[0] + ".wav")
                    if os.path.exists(wav_path):
                        utt2wav[utt] = wav_path
                        utt2text[utt] = text
                        utt2spk[utt] = speaker
                        if speaker not in spk2utt:
                            spk2utt[speaker] = []
                        spk2utt[speaker].append(utt)
            
    if not os.path.exists(args.des_dir):
        os.makedirs(args.des_dir)
       
    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:
        for k, v in utt2wav.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/text'.format(args.des_dir), 'w') as f:
        for k, v in utt2text.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/utt2spk'.format(args.des_dir), 'w') as f:
        for k, v in utt2spk.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/spk2utt'.format(args.des_dir), 'w') as f:
        for k, v in spk2utt.items():
            f.write('{} {}\n'.format(k, ' '.join(v)))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    parser.add_argument('--speakers', 
                        type=str)
    args = parser.parse_args()
    main()
