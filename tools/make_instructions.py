import json
import logging
import os
import random

def is_valid_selection(templates, selected_keys):
    if ('shimmer' in selected_keys and len(templates['shimmer']) > 1) and ('jitter' in selected_keys and len(templates['jitter']) > 1):
        # print("jitter and shimmer both too long")
        return False
    elif ('shimmer' in selected_keys and len(templates['shimmer']) > 1) and 'pitch_var' in selected_keys:
        # print("shimmer too long and pitch_var")
        return False
    elif ('jitter' in selected_keys and len(templates['jitter']) > 1) and 'pitch_var' in selected_keys:
        # print("jitter too long and pitch_var")          
        return False
    else:
        return True

def make_instructions(utt2templates, utt, rand=5):
    templates = utt2templates[utt]
    keys = list(templates.keys())

    # while True:
    #     # select a random number of keys between 1 and rand
    #     n = random.randint(1, rand)
    #     # print(n)
    #     selected_keys = random.sample(keys, n) # select n random keys
    #     # if is_valid_selection(templates, selected_keys):
    #     #     break
    n = random.randint(1, rand)
    selected_keys = random.sample(keys, n) # select n random keys
    selected_templates = {key: random.choice(templates[key]) for key in selected_keys} # randomly select one template for each key
    # Create a string that joins all values of selected_templates with "and"
    instruction = " and ".join([str(value) for value in selected_templates.values()])
    
    return instruction

def make_instructions_simple(utt2templates, utt, rand=5):
    templates = utt2templates[utt]
    keys = list(templates.keys())

    # select a random number of keys between 1 and rand
    n = random.randint(1, rand)
    # print(n)
    selected_keys = random.sample(keys, n) # select n random keys
        
    selected_templates = {key: templates[key][0] for key in selected_keys} # always select first template
    # Create a string that joins all values of selected_templates with "and"
    instruction = " and ".join([str(value) for value in selected_templates.values()])
    
    return instruction



if __name__ == "__main__":

    with open("examples/poetry_instruct/cosyvoice/test_template.json", "r", encoding="utf-8") as f:
        utt2templates = json.load(f)
    
    utt = "tts-ayd-0450-m01-s02-t09-v01_1"
    utt = os.path.join(utt.split("_")[0], f"{utt}.wav")
    templates = make_instructions_simple(utt2templates, utt)
    print(templates)