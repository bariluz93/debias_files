import argparse
import os
from sacrebleu.metrics import BLEU
import copy
import sys
sys.path.append("../../") # Adds higher directory to python modules path.
from debias_files.src.debias_manager import DebiasManager
from consts import get_debias_files_from_config, EMBEDDING_SIZE, DEFINITIONAL_FILE, PROFESSIONS_FILE, \
    GENDER_SPECIFIC_FILE, EQUALIZE_FILE, DebiasMethod, get_basic_configurations, \
    TranslationModelsEnum, get_evaluate_gender_files, WordsToDebias, ENGLISH_VOCAB, param_dict,\
    LANGUAGE_OPPOSITE_STR_MAP, lang_to_gender_specific_words_map,DATA_HOME, LANGUAGE_STR_TO_INT_MAP
import torch
os.environ['TRANSFORMERS_CACHE'] = '/cs/snapless/gabis/bareluz'
from transformers import T5Tokenizer, T5ForConditionalGeneration

LANG_TO_TASK_PREFIX ={'de':"translate English to German: ", "he":"translate English to Hebrew: ","ru":"translate English to Russian: "}
# refs_path_de ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_de_5.8/newstest2012.de"
# file_path_de ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_de_5.8/newstest2012.en"
#
# refs_path_he ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_he_20.07.21/dev.he"
# file_path_he ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_he_20.07.21/dev.en"
#
# refs_path_ru ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_ru_30.11.20/newstest2019-enru.ru"
# file_path_ru ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_ru_30.11.20/newstest2019-enru.en"

tokenizer = T5Tokenizer.from_pretrained("t5-large")

bleu = BLEU()
def translate(input_file:str, output_file:str, config:str):
    USE_DEBIASED, LANGUAGE, _, DEBIAS_METHOD, _, DEBIAS_ENCODER, BEGINNING_DECODER_DEBIAS, END_DECODER_DEBIAS, WORDS_TO_DEBIAS = get_basic_configurations(
        config)
    model = T5ForConditionalGeneration.from_pretrained("t5-large", tokenizer=None, source_lang=None,
                                                       target_lang=None,
                                                       show_progress_bar=None,
                                                       use_debiased=None,
                                                       debias_method=None,
                                                       debias_encoder=None,
                                                       beginning_decoder_debias=None,
                                                       end_decoder_debias=None,
                                                       words_to_debias=None)
    if USE_DEBIASED:
        print("using debiased embeddings")
        target_lang = 'es' if tokenizer.target_lang == 'spa' else tokenizer.target_lang
        config_str = "{'USE_DEBIASED': 1" \
                     ", 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[target_lang]) + \
                     ", 'DEBIAS_METHOD': " + str(DEBIAS_METHOD) + \
                     ", 'TRANSLATION_MODEL': 1" \
                     ", 'DEBIAS_ENCODER': " + str(DEBIAS_ENCODER) + "" \
                     ", 'BEGINNING_DECODER_DEBIAS': " + str(BEGINNING_DECODER_DEBIAS) + \
                     ", 'END_DECODER_DEBIAS': " + str(END_DECODER_DEBIAS) + \
                     ", 'WORDS_TO_DEBIAS': " + str(WORDS_TO_DEBIAS) + "}"
        dict = model.state_dict()
        # option 1: debias encoder inputs
        if DEBIAS_ENCODER:
            print('debias_encoder')
            config_str = "{'USE_DEBIASED': 1" \
                         ", 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[target_lang]) + \
                         ", 'DEBIAS_METHOD': " + str(DEBIAS_METHOD) + \
                         ", 'TRANSLATION_MODEL': 1" \
                         ", 'DEBIAS_ENCODER': 1" \
                         ", 'BEGINNING_DECODER_DEBIAS': 0" + \
                         ", 'END_DECODER_DEBIAS': 0" + \
                         ", 'WORDS_TO_DEBIAS': " + str(WORDS_TO_DEBIAS) + "}"
            weights_encoder = copy.deepcopy(dict['model.encoder.embed_tokens.weight'])
            debias_manager_encoder = DebiasManager.get_manager_instance(config_str, weights_encoder, tokenizer)
            new_embeddings_encoder = torch.from_numpy(debias_manager_encoder.debias_embedding_table())
            sanity_check_debias(weights_encoder, new_embeddings_encoder, debias_manager_encoder)
            dict['model.encoder.embed_tokens.weight'] = new_embeddings_encoder
        # option 2: debias decoder inputs
        if BEGINNING_DECODER_DEBIAS:
            print('beginning_decoder_debias')
            config_str = "{'USE_DEBIASED': 1" \
                         ", 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[target_lang]) + \
                         ", 'DEBIAS_METHOD': " + str(DEBIAS_METHOD) + \
                         ", 'TRANSLATION_MODEL': 1" \
                         ", 'DEBIAS_ENCODER': 0" \
                         ", 'BEGINNING_DECODER_DEBIAS': 1" + \
                         ", 'END_DECODER_DEBIAS': 0" + \
                         ", 'WORDS_TO_DEBIAS': " + str(WORDS_TO_DEBIAS) + "}"
            weights_decoder = copy.deepcopy(dict['model.decoder.embed_tokens.weight'])
            debias_manager_decoder = DebiasManager.get_manager_instance(config_str, weights_decoder, tokenizer,
                                                                        debias_target_language=True)
            new_embeddings_decoder = torch.from_numpy(debias_manager_decoder.debias_embedding_table())
            sanity_check_debias(weights_decoder, new_embeddings_decoder, debias_manager_decoder)
            dict['model.decoder.embed_tokens.weight'] = new_embeddings_decoder

        # # option 3: debias decoder outputs
        if END_DECODER_DEBIAS:
            print('end_decoder_debias')
            config_str = "{'USE_DEBIASED': 1" \
                         ", 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[target_lang]) + \
                         ", 'DEBIAS_METHOD': " + str(DEBIAS_METHOD) + \
                         ", 'TRANSLATION_MODEL': 1" \
                         ", 'DEBIAS_ENCODER': 0" \
                         ", 'BEGINNING_DECODER_DEBIAS': 0" + \
                         ", 'END_DECODER_DEBIAS': 1" + \
                         ", 'WORDS_TO_DEBIAS': " + str(WORDS_TO_DEBIAS) + "}"
            weights_decoder_outputs = copy.deepcopy(dict['lm_head.weight'])
            debias_manager_decoder_outputs = DebiasManager.get_manager_instance(config_str, weights_decoder_outputs,
                                                                                tokenizer, debias_target_language=True)
            new_embeddings_decoder_outputs = torch.from_numpy(debias_manager_decoder_outputs.debias_embedding_table())
            sanity_check_debias(weights_decoder_outputs, new_embeddings_decoder_outputs,
                                      debias_manager_decoder_outputs)
            dict['lm_head.weight'] = new_embeddings_decoder_outputs

        model.load_state_dict(dict)

    with open(input_file,'r') as f:
        sentences= f.readlines()
    inputs = tokenizer([LANG_TO_TASK_PREFIX[LANGUAGE] + sentence for sentence in sentences], return_tensors="pt", padding=True)

    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=False, # disable sampling to test if batching affects output
    )
    translations = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    with open(output_file, 'w') as f2:
        for t in translations:
            f2.write(t+"\n")
    # with open(refs_path,'r') as f:
    #     refs = f.readlines()
    #
    # print("bleu t5 "+task_prefix )
    # print(bleu.corpus_score(translations, [refs]))

def sanity_check_debias(orig_embeddings, debiased_embeddings,model):
    c=0
    for i in range(len(orig_embeddings)):
        if (debiased_embeddings[i] != orig_embeddings[i]).any():
            c += 1
    if model.debias_target_language:
        if model.target_lang == 'he':
            assert (c == len(model.hebrew_professions))
        elif model.target_lang == 'de':
            assert (c == len(model.german_professions))
        # elif model.target_lang == 'ru':
        #     assert (c == len(model.russian_professions))
    else:
        assert (c == len(model.professions))

# if __name__ == '__main__':
#
#
#     translate_t5(task_prefix_de, file_path_de,refs_path_de,"translations_de_t5.txt","de")
#     translate_t5(task_prefix_he, file_path_he,refs_path_he,"translations_he_t5.txt","he")
#     translate_t5(task_prefix_ru, file_path_ru,refs_path_ru,"translations_ru_t5.txt","ru")
#
# input_ids = tokenizer(
#     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
# ).input_ids  # Batch size 1
# decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
#
# # forward pass
# outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
# last_hidden_states = outputs.last_hidden_state


# from easynmt import EasyNMT
# from consts import get_basic_configurations, LANGUAGE_STR_MAP, Language
# model = EasyNMT('opus-mt')
#
# def translate(input_file:str, output_file:str, config:str):
#     USE_DEBIASED, LANGUAGE, _, DEBIAS_METHOD, _ ,DEBIAS_ENCODER,BEGINNING_DECODER_DEBIAS,END_DECODER_DEBIAS,WORDS_TO_DEBIAS= get_basic_configurations(config)
#     with open(input_file, 'r') as input, open(output_file, 'w') as output:
#         translations = model.translate(input.readlines(), source_lang='en',
#                                        target_lang=LANGUAGE_STR_MAP[Language(LANGUAGE)],
#                                        show_progress_bar=True,
#                                        use_debiased = USE_DEBIASED,
#                                        debias_method=DEBIAS_METHOD,
#                                        debias_encoder=DEBIAS_ENCODER,
#                                        beginning_decoder_debias=BEGINNING_DECODER_DEBIAS,
#                                        end_decoder_debias=END_DECODER_DEBIAS,
#                                        words_to_debias=WORDS_TO_DEBIAS)
#         output.writelines(translations)
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=str,
        help="input file")
    parser.add_argument(
        '-o', '--output', type=str,
        help="output file")
    parser.add_argument(
        '-c', '--config_str', type=str, required=True,
        help="a config dictionary str that conatains: \n"
             "USE_DEBIASED= run translate on the debiased dictionary or not\n"
             "LANGUAGE= the language to translate to from english. RUSSIAN = 0, GERMAN = 1, HEBREW = 2\n"
             "DEBIAS_METHOD= the debias method. HARD_DEBIAS = 0 INLP = 1\n"
             "TRANSLATION_MODEL= the translation model. Nematus=0, EasyNMT=1\n"
             "DEBIAS_ENCODER= whether to debias the encoder\n"
             "BEGINNING_DECODER_DEBIAS= whether to debias the inputs of the decoder\n"
             "END_DECODER_DEBIAS= whether to debias the outputs of the decoder\n"""
             "WORDS_TO_DEBIAS= set of words to debias. ALL_VOCAB = 0, ONE_TOKEN_PROFESSIONS = 1, ALL_PROFESSIONS = 2 \n"

    )
    args = parser.parse_args()
    translate(args.input, args.output, args.config_str)
