import json
import random

def split_data(data, train_ratio=0.8):
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def main():
    input_file = '/mount/arbeitsdaten/textklang/synthesis/CosyVoice/CosyVoice/examples/poetry/cosyvoice/data.json'
    train_file = 'train_data.json'
    test_file = 'test_data.json'
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    train_data, test_data = split_data(data)
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=4)
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print(f'Train data saved to {train_file}')
    print(f'Test data saved to {test_file}')

if __name__ == "__main__":
    main()