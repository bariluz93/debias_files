from consts import Language, param_dict, DATA_HOME,ANTI_DATA_HOME, ENGLISH_VOCAB,LANGUAGE_STR_TO_INT_MAP, get_debias_files_from_config
from transformers import MarianTokenizer

tokenizer_de = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
tokenizer_he = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-he")
tokenizer_ru = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

config_ru = "{'USE_DEBIASED': 0, 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP["ru"]) + \
            ", 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': 0, 'TRANSLATION_MODEL': 1, " \
            "'DEBIAS_ENCODER': 0, 'BEGINNING_DECODER_DEBIAS': 0, 'END_DECODER_DEBIAS': 0, 'WORDS_TO_DEBIAS': 0}"
config_de = "{'USE_DEBIASED': 0, 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP["de"]) + \
            ", 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': 0, 'TRANSLATION_MODEL': 1, " \
            "'DEBIAS_ENCODER': 0, 'BEGINNING_DECODER_DEBIAS': 0, 'END_DECODER_DEBIAS': 0, 'WORDS_TO_DEBIAS': 0}"
config_he = "{'USE_DEBIASED': 0, 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP["he"]) + \
            ", 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': 0, 'TRANSLATION_MODEL': 1, " \
            "'DEBIAS_ENCODER': 0, 'BEGINNING_DECODER_DEBIAS': 0, 'END_DECODER_DEBIAS': 0, 'WORDS_TO_DEBIAS': 0}"

en_dataset_paths = [DATA_HOME+'en_ru_30.11.20/newstest2019-enru.en',
                    DATA_HOME+'en_de_5.8/newstest2012.en',
                    DATA_HOME+'en_he_20.07.21/dev.en',
                    ANTI_DATA_HOME+'anti.en']

_, _, _, _, EMBEDDING_DEBIASWE_FILE_RU, _, _=get_debias_files_from_config(config_ru)
_, _, _, _, EMBEDDING_DEBIASWE_FILE_DE, _, _=get_debias_files_from_config(config_de)
_, _, _, _, EMBEDDING_DEBIASWE_FILE_HE, _, _=get_debias_files_from_config(config_he)

def get_vocab(tokenizer, src_path, target_filename):
    with open(src_path, 'r') as src_file, open(target_filename, 'w+') as dst_vocab:
        lines = src_file.readlines()
        lang_vocab = set()
        with tokenizer.as_target_tokenizer():
            for line in lines:
                indices = tokenizer(line)['input_ids'][:-1]
                tokens = tokenizer.convert_ids_to_tokens(indices)

                for i in range(len(tokens)):
                    lang_vocab.add(tokens[i]+"\t"+str(indices[i]))

        lang_vocab = list(lang_vocab)

        for w in lang_vocab:
            dst_vocab.write( w + '\n')
def get_en_vocab(tokenizer, src_paths, target_filename):
    tokens_set = set()
    for p in src_paths:
        with open(p, 'r') as src_file:
            lines = [l.strip() for l in src_file.readlines()]
            for line in lines:
                tokens = tokenizer.tokenize(line)
                indices = tokenizer.convert_tokens_to_ids(tokens)
                for i in range(len(tokens)):
                    tokens_set.add(tokens[i] + "\t" + str(indices[i]))

    tokens_set = list(tokens_set)
    with open(target_filename,"w+") as f:
        for w in tokens_set:
            f.write( w + '\n')


# def combine_vocabs(vocabs_paths):
#     combined_words = set()
#     for p in vocabs_paths:
#         with open(p,'r') as f:
#             words = [(w.strip().split("\t"))[0] for w in f.readlines()]
#             combined_words = combined_words|set(words)
#     return list(combined_words)

# def get_english_vocab_per_language(english_words, general_vocab_filename,target_filename):
#     with open(general_vocab_filename,"r") as f:
#         target_lang_vocab = f.readlines()
#         target_lang_words = [l.split(" ")[0] for l in target_lang_vocab]
#     words_with_indices = []
#     for w in english_words:
#         words_with_indices.append(w+"\t"+str(target_lang_words.index(w)))
#     with open(target_filename,"w+") as f:
#         for w in words_with_indices:
#             f.write(w+"\n")

def save_vocab_inlp_form(general_vocab_filename,language_vocab_filename, target_filename):
    with open(general_vocab_filename,'r') as f1, open(language_vocab_filename,'r') as f2:
        general_vocab = f1.readlines()
        general_vocab = [l.split(" ") for l in general_vocab]
        embedding_size = general_vocab[0][1]
        language_vocab = f2.readlines()
        language_vocab = [l.split("\t") for l in language_vocab]
    with open(target_filename,"w+") as f:
        lines_to_write = []
        vocab_len = len(language_vocab)
        for w in language_vocab:
            index = int(w[1])
            if w[0] == general_vocab[index+1][0]:
                lines_to_write.append(w[0]+" "+" ".join(general_vocab[index+1][1:]))
            else:
                print(w[0] + " was not found in the vocab")
                vocab_len -=1
        lines_to_write.insert(0,str(vocab_len)+" "+embedding_size)

        for l in lines_to_write:
            f.write(l)

if __name__ == '__main__':

    get_vocab(tokenizer_ru,param_dict[Language.RUSSIAN]["BLEU_GOLD_DATA_NON_TOKENIZED"],param_dict[Language.RUSSIAN]["VOCAB"])
    get_en_vocab(tokenizer_ru,en_dataset_paths,param_dict[Language.RUSSIAN]['VOCAB_EN'])

    get_vocab(tokenizer_de,param_dict[Language.GERMAN]["BLEU_GOLD_DATA_NON_TOKENIZED"],param_dict[Language.GERMAN]["VOCAB"])
    get_en_vocab(tokenizer_de,en_dataset_paths,param_dict[Language.GERMAN]['VOCAB_EN'])

    get_vocab(tokenizer_he,param_dict[Language.HEBREW]["BLEU_GOLD_DATA_NON_TOKENIZED"],param_dict[Language.HEBREW]["VOCAB"])
    get_en_vocab(tokenizer_he,en_dataset_paths,param_dict[Language.HEBREW]['VOCAB_EN'])


    save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_RU,param_dict[Language.RUSSIAN]["VOCAB"],param_dict[Language.RUSSIAN]["VOCAB_INLP"])
    save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_RU,param_dict[Language.RUSSIAN]['VOCAB_EN'],param_dict[Language.RUSSIAN]["VOCAB_INLP_EN"])

    save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_DE,param_dict[Language.GERMAN]["VOCAB"],param_dict[Language.GERMAN]["VOCAB_INLP"])
    save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_DE,param_dict[Language.GERMAN]['VOCAB_EN'],param_dict[Language.GERMAN]["VOCAB_INLP_EN"])

    save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_HE,param_dict[Language.HEBREW]["VOCAB"],param_dict[Language.HEBREW]["VOCAB_INLP"])
    save_vocab_inlp_form(EMBEDDING_DEBIASWE_FILE_HE,param_dict[Language.HEBREW]['VOCAB_EN'],param_dict[Language.HEBREW]["VOCAB_INLP_EN"])