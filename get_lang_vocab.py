from consts import Language, param_dict, DATA_HOME



def get_vocab(tokenizer,src_path,lang_filename):
    with open(src_path, 'r') as src_file, open(lang_filename, 'w+') as dst_vocab:
        lines = src_file.readlines()
        lang_vocab = set()
        with tokenizer.as_target_tokenizer():
            for line in lines:
                indices = tokenizer(line, return_tensors="pt", padding=True)['input_ids'][0]
                tokens = tokenizer.convert_ids_to_tokens(indices)

                for i in range(len(tokens)):
                    lang_vocab.add(tokens[i]+"\t"+str(indices[i].item()))
        lang_vocab = list(lang_vocab)

        # tokens,indices=[],[]
        # for w in lang_vocab:
        #     a,b=w.split('\t')
        #     tokens.append(a)
        #     indices.append(b)
        # tokens = [x for _, x in sorted(zip(indices, tokens))]
        # indices = sorted(indices)
        # print(tokens)
        # print(indices)
        for w in lang_vocab:
            dst_vocab.write( w + '\n')

def combine_vocabs(vocabs_paths):
    combined_vocab = set()
    for p in vocabs_paths:
        with open(p,'r') as f:
            combined_vocab = combined_vocab|set(f.readlines())
    combined_vocab = list(combined_vocab)
    with open('/cs/snapless/gabis/bareluz/en_vocab_merged.txt','w+') as f:
        for w in combined_vocab:
            f.write(w)

if __name__ == '__main__':

    tokenizer=None#add tokenizer
    get_vocab(tokenizer,param_dict[Language.RUSSIAN]["BLEU_GOLD_DATA_NON_TOKENIZED"],'/cs/snapless/gabis/bareluz/data/en_ru_30.11.20/ru_vocab.txt')
    get_vocab(tokenizer,'/cs/snapless/gabis/bareluz/data/en_ru_30.11.20/newstest2019-enru.en','/cs/snapless/gabis/bareluz/data/en_ru_30.11.20/en_vocab.txt')

    get_vocab(tokenizer,param_dict[Language.GERMAN]["BLEU_GOLD_DATA_NON_TOKENIZED"],'/cs/snapless/gabis/bareluz/data/en_de_5.8/de_vocab.txt')
    get_vocab(tokenizer,'/cs/snapless/gabis/bareluz/data/en_de_5.8/newstest2012.en','/cs/snapless/gabis/bareluz/data/en_de_5.8/en_vocab.txt')

    get_vocab(tokenizer,param_dict[Language.HEBREW]["BLEU_GOLD_DATA_NON_TOKENIZED"],'/cs/snapless/gabis/bareluz/data/en_he_20.07.21/he_vocab.txt')
    get_vocab(tokenizer,'/cs/snapless/gabis/bareluz/data/en_he_20.07.21/dev.en','/cs/snapless/gabis/bareluz/data/en_he_20.07.21/en_vocab.txt')

    get_vocab(tokenizer,'/cs/snapless/gabis/bareluz/anti_data/anti.en','/cs/snapless/gabis/bareluz/anti_data/en_vocab.txt')

    combine_vocabs([DATA_HOME+'en_ru_30.11.20/en_vocab.txt',DATA_HOME+'en_de_5.8/en_vocab.txt',DATA_HOME+'en_he_20.07.21/en_vocab.txt','/cs/snapless/gabis/bareluz/anti_data/en_vocab.txt'])
