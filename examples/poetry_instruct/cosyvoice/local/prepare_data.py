import argparse
import logging
import glob
import os
from tqdm import tqdm
import pandas as pd


logger = logging.getLogger()


def main():

    # Load JSON file as pandas dataframe
    data_df = pd.read_json(args.data_json)

    wavs = []
    for root, _, files in os.walk(args.src_dir):
        for file in files:
            if file.endswith('.wav'):
                wavs.append(os.path.join(root, file))
    # print(wavs)

    if not os.path.exists(args.des_dir):
        os.makedirs(args.des_dir)
       
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    
    for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
        # content = ''.join(row['text_strophe'].replace('\n', ' '))
        file_name = row['lccn'] + "_" + str(row['strophe']) + ".wav"
        file_path = os.path.join(args.src_dir, row['lccn'], file_name)

        if file_path in wavs:
            # utt = file_path.split(".")[0]
            utt = row['lccn'] + "_" + str(row['strophe'])
            spk = row['performer'].strip().replace(", ", "_").replace(" ", "")
            utt2wav[utt] = file_path
            utt2text[utt] = row['text_strophe'].replace('\n', ' ')
            utt2spk[utt] = spk
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(utt)
        
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
    parser.add_argument('--data_json', 
                        type=str)
    args = parser.parse_args()
    main()
