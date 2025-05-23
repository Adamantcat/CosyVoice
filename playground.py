import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import sys


cosyvoice2 = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

for i, j in enumerate(cosyvoice2.inference_instruct2('The sun was just beginning to rise over the mountains.', 'a woman', prompt_speech_16k, stream=False)):
    torchaudio.save('audios/cosyvoice2/instruct_female.wav'.format(i), j['tts_speech'], cosyvoice2.sample_rate)

for i, j in enumerate(cosyvoice2.inference_instruct2('The sun was just beginning to rise over the mountains.', 'a man', prompt_speech_16k, stream=False)):
    torchaudio.save('audios/cosyvoice2/instruct_male.wav'.format(i), j['tts_speech'], cosyvoice2.sample_rate)
    
sys.exit(0) 

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')

for i, j in enumerate(cosyvoice.inference_instruct('The sun was just beginning to rise over the mountains.', '中文男' ,'A woman.', stream=False)):
    torchaudio.save('audios/cosyvoice/test/female_1.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct('The sun was just beginning to rise over the mountains.', '中文男' ,'A man.', stream=False)):
    torchaudio.save('audios/cosyvoice/test/male_1.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct('She carefully placed the book back on the shelf.', '中文男' ,'speaker is female', stream=False)):
    torchaudio.save('audios/cosyvoice/test/female_2.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct('She carefully placed the book back on the shelf.', '中文男' ,'speaker is male', stream=False)):
    torchaudio.save('audios/cosyvoice/test/male_2.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)





sys.exit(0)
# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct_poetry')

# for i, j in enumerate(cosyvoice.inference_instruct('All that is gold does not glitter Not all those who wander are lost The old that is strong does not wither Deep roots are not reached by the frost.', '中文男' ,'A female speaker with normal pitch and normal speaking rate.', stream=False)):
#     torchaudio.save('audios/cosyvoice/instruct/sanityceck_1.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# for i, j in enumerate(cosyvoice.inference_instruct('All that is gold does not glitter Not all those who wander are lost The old that is strong does not wither Deep roots are not reached by the frost.', '中文男' ,'A female speaker with high pitch, normal speaking rate, and happy emotion.', stream=False)):
#     torchaudio.save('audios/cosyvoice/instruct/sanityceck_2.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# for i, j in enumerate(cosyvoice.inference_instruct('All that is gold does not glitter Not all those who wander are lost The old that is strong does not wither Deep roots are not reached by the frost.', '中文男' ,'A male speaker with low pitch, slow speaking rate, and sad emotion.', stream=False)):
#     torchaudio.save('audios/cosyvoice/instruct/sanityceck_3.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# for i, j in enumerate(cosyvoice.inference_instruct('All that is gold does not glitter Not all those who wander are lost The old that is strong does not wither Deep roots are not reached by the frost.', '中文男' ,'A male speaker with low pitch, fast speaking rate, and angry emotion.', stream=False)):
#     torchaudio.save('audios/cosyvoice/instruct/sanityceck_4.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


# sys.exit(0)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct_poetry')
# instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
for i, j in enumerate(cosyvoice.inference_instruct('Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '中文男' ,'the speaker is female.', stream=False)):
    torchaudio.save('audios/cosyvoice/hui_instruct_v2/vergissmeinnicht_0.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
sys.exit(0)
for i, j in enumerate(cosyvoice.inference_instruct('Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '中文男' ,'the speaker is male and has high arousal and appears to be in a bad mood.', stream=False)):
    torchaudio.save('audios/cosyvoice/hui_instruct_v2/vergissmeinnicht_1.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct('Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '中文男' ,'the speaker is male and has high arousal and appears to be in a good mood.', stream=False)):
    torchaudio.save('audios/cosyvoice/hui_instruct_v2/vergissmeinnicht_2.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct('Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '中文男' ,'a woman is speaking and sound pressure is elevated.', stream=False)):
    torchaudio.save('audios/cosyvoice/hui_instruct_v2/vergissmeinnicht_3.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct('Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '中文男' ,'a woman is speaking and is almost silent.', stream=False)):
    torchaudio.save('audios/cosyvoice/hui_instruct_v2/vergissmeinnicht_4.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct('Schicke mir ein Blatt, doch von einem Strauche Der nicht näher als eine halbe Stunde Von deinem Haus wächst, dann Mußt du gehen und wirst stark, und ich bedanke mich für das hübsche Blatt.', '中文男' ,'has a high pitch variance and speaker is calm and has low dominance.', stream=False)):
    torchaudio.save('audios/cosyvoice/hui_instruct_v2/Blatt_1.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct('Schicke mir ein Blatt, doch von einem Strauche Der nicht näher als eine halbe Stunde Von deinem Haus wächst, dann Mußt du gehen und wirst stark, und ich bedanke mich für das hübsche Blatt.', '中文男' ,'has a low pitch variance and speaker is calm and has high dominance.', stream=False)):
    torchaudio.save('audios/cosyvoice/hui_instruct_v2/Blatt_2.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct('Schicke mir ein Blatt, doch von einem Strauche Der nicht näher als eine halbe Stunde Von deinem Haus wächst, dann Mußt du gehen und wirst stark, und ich bedanke mich für das hübsche Blatt.', '中文男' ,'has a high jitter.', stream=False)):
    torchaudio.save('audios/cosyvoice/hui_instruct_v2/Blatt_3.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct('Schicke mir ein Blatt, doch von einem Strauche Der nicht näher als eine halbe Stunde Von deinem Haus wächst, dann Mußt du gehen und wirst stark, und ich bedanke mich für das hübsche Blatt.', '中文男' ,'has a low jitter.', stream=False)):
    torchaudio.save('audios/cosyvoice/hui_instruct_v2/Blatt_4.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


# for i, j in enumerate(cosyvoice.inference_instruct('Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '中文男' ,'has low valence', stream=False)):
#     torchaudio.save('audios/cosyvoice/instruct/vergissmeinnicht_5.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# for i, j in enumerate(cosyvoice.inference_instruct('Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '中文男' ,'has high valence', stream=False)):
#     torchaudio.save('audios/cosyvoice/instruct/vergissmeinnicht_6.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct_simple')
# for i, j in enumerate(cosyvoice.inference_instruct('<|de|> Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '中文男' ,'a man is speaking and has low valence and has high arousal.', stream=False)):
#     torchaudio.save('audios/cosyvoice/instruct_simple/vergissmeinnicht_1.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# for i, j in enumerate(cosyvoice.inference_instruct('<|de|> Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '中文男' ,'a man is speaking and has high valence and has high arousal.', stream=False)):
#     torchaudio.save('audios/cosyvoice/instruct_simple/vergissmeinnicht_2.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# for i, j in enumerate(cosyvoice.inference_instruct('<|de|> Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '中文男' ,'a woman is speaking and is loud.', stream=False)):
#     torchaudio.save('audios/cosyvoice/instruct_simple/vergissmeinnicht_3.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# for i, j in enumerate(cosyvoice.inference_instruct('<|de|> Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '中文男' ,'a woman is speaking and has a low equivalent sound level.', stream=False)):
#     torchaudio.save('audios/cosyvoice/instruct_simple/vergissmeinnicht_4.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

sys.exit(0)



cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M_poetry', load_jit=False, load_trt=False, fp16=False) # or change to pretrained_models/CosyVoice-300M-25Hz for 25Hz inference
# print(cosyvoice.list_available_spks())
# zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
prompt_speech_16k_zero_shot = load_wav('./asset/zero_shot_prompt.wav', 16000)
prompt_speech_16k_crossling = load_wav('./asset/cross_lingual_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('<|de|>Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', '希望你以后能够做的比我还好呦。', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/vergissmeinnicht1_hui.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot('<|de|>Es weiß nicht viel zu reden Und alles, was es spricht, Ist immer nur dasselbe, Ist nur: Vergissmeinnicht.', '希望你以后能够做的比我还好呦。', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/vergissmeinnicht2_hui.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot("<|de|>Wenn ich zwei Äuglein sehe So heiter und so blau, So denk' ich an mein Blümchen Auf unsrer grünen Au.", '希望你以后能够做的比我还好呦。', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/vergissmeinnicht3_hui.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot("<|de|>Da kann ich auch nicht reden Und nur mein Herze spricht, So bange nur, so leise, Und nur: Vergissmeinnicht.", '希望你以后能够做的比我还好呦。', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/vergissmeinnicht4_hui.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot('<|de|>Schicke mir ein Blatt, doch von einem Strauche Der nicht näher als eine halbe Stunde Von deinem Haus wächst, dann Mußt du gehen und wirst stark, und ich bedanke mich für das hübsche Blatt.', '希望你以后能够做的比我还好呦。', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/Blatt_hui.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot('<|de|>Ich habe zu Hause ein blaues Klavier Und kenne doch keine Note.', '希望你以后能够做的比我还好呦。', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/Klavier1_hui.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot('<|de|>Es steht im Dunkel der Kellertür, Seitdem die Welt verrohte.', '希望你以后能够做的比我还好呦。', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/Klavier2_hui.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot('<|en|>All that is gold does not glitter Not all those who wander are lost The old that is strong does not wither Deep roots are not reached by the frost.', '希望你以后能够做的比我还好呦。', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/english_hui.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


# cross_lingual usage
for i, j in enumerate(cosyvoice.inference_cross_lingual('<|de|>Es blüht ein schönes Blümchen auf unsrer grünen Au. Sein Aug ist wie der Himmel, so heiter und so blau.', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/vegissmeinnicht_1_hui_crossling.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_cross_lingual('<|de|>Es weiß nicht viel zu reden Und alles, was es spricht, Ist immer nur dasselbe, Ist nur: Vergissmeinnicht.', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/vergissmeinnicht2_hui_crossling.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_cross_lingual("<|de|>Wenn ich zwei Äuglein sehe So heiter und so blau, So denk' ich an mein Blümchen Auf unsrer grünen Au.", prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/vergissmeinnicht3_hui_crossling.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_cross_lingual("<|de|>Da kann ich auch nicht reden Und nur mein Herze spricht, So bange nur, so leise, Und nur: Vergissmeinnicht.", prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/vergissmeinnicht4_hui_crossling.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_cross_lingual('<|de|>Schicke mir ein Blatt, doch von einem Strauche Der nicht näher als eine halbe Stunde Von deinem Haus wächst, dann Mußt du gehen und wirst stark, und ich bedanke mich für das hübsche Blatt.', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/Blatt_hui_crossling.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_cross_lingual('<|de|>Ich habe zu Hause ein blaues Klavier Und kenne doch keine Note.', prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/Klavier1_hui_crossling.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_cross_lingual('<|de|>Es steht im Dunkel der Kellertür, Seitdem die Welt verrohte.',  prompt_speech_16k_zero_shot, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/Klavier2_hui_crossling.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>All that is gold does not glitter Not all those who wander are lost The old that is strong does not wither Deep roots are not reached by the frost.', prompt_speech_16k_crossling, stream=False)):
    torchaudio.save('audios/cosyvoice/hui/english_hui_crossling.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)




sys.exit(0)


## CosyVoice 2 ##
cosyvoice2 = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice2.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice2.sample_rate)

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice2.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice2.sample_rate)

# instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
the_raven = "Once upon a midnight dreary, while I pondered, weak and weary, Over many a quaint and curious volume of forgotten lore"

for i, j in enumerate(cosyvoice2.inference_instruct2(the_raven, 'happy', prompt_speech_16k, stream=False)):
    torchaudio.save('audios/cosyvoice2/instruct_happy.wav'.format(i), j['tts_speech'], cosyvoice2.sample_rate)
    
for i, j in enumerate(cosyvoice2.inference_instruct2(the_raven, 'angry', prompt_speech_16k, stream=False)):
    torchaudio.save('audios/cosyvoice2/instruct_angry.wav'.format(i), j['tts_speech'], cosyvoice2.sample_rate)

for i, j in enumerate(cosyvoice2.inference_instruct2(the_raven, 'slow', prompt_speech_16k, stream=False)):
    torchaudio.save('audios/cosyvoice2/instruct_slow.wav'.format(i), j['tts_speech'], cosyvoice2.sample_rate)

for i, j in enumerate(cosyvoice2.inference_instruct2(the_raven, 'slow, like a nursery rhyme', prompt_speech_16k, stream=False)):
    torchaudio.save('audios/cosyvoice2/instruct_slow_nursery.wav'.format(i), j['tts_speech'], cosyvoice2.sample_rate)

for i, j in enumerate(cosyvoice2.inference_instruct2(the_raven, 'screaming in frustration', prompt_speech_16k, stream=False)):
    torchaudio.save('audios/cosyvoice2/instruct_frustration.wav'.format(i), j['tts_speech'], cosyvoice2.sample_rate)


# sys.exit(0)
## CosyVoice 1 ##

# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)
# # sft usage
# print(cosyvoice.list_available_spks())
# # change stream=True for chunk stream inference
# for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
#     torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M') # or change to pretrained_models/CosyVoice-300M-25Hz for 25Hz inference
# # zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# # cross_lingual usage
# prompt_speech_16k = load_wav('./asset/cross_lingual_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k, stream=False)):
#     torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# # vc usage
# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# source_speech_16k = load_wav('./asset/cross_lingual_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)):
#     torchaudio.save('vc_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
# instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
for i, j in enumerate(cosyvoice.inference_instruct(the_raven, '中文男' ,'happy', stream=False)):
    torchaudio.save('audios/cosyvoice/instruct_happy.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct(the_raven, '中文男', 'angry', stream=False)):
    torchaudio.save('audios/cosyvoice/instruct_angry.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct(the_raven, '中文男', 'slow', stream=False)):
    torchaudio.save('audios/cosyvoice/instruct_slow.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct(the_raven, '中文男', 'slow, like a nursery rhyme', stream=False)):
    torchaudio.save('audios/cosyvoice/instruct_slow_nursery.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)