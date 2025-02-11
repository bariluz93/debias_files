import os
os.environ['TRANSFORMERS_CACHE'] = '/cs/snapless/gabis/bareluz'
from transformers import MarianTokenizer
print(1)
TRANSLATOR="opus-mt"
SRC_LANG="en"
tokenizer_he = MarianTokenizer.from_pretrained(f"Helsinki-NLP/{TRANSLATOR}-{SRC_LANG}-{'he'}")
tokenizer_de = MarianTokenizer.from_pretrained(f"Helsinki-NLP/{TRANSLATOR}-{SRC_LANG}-{'de'}")
tokenizer_ru = MarianTokenizer.from_pretrained(f"Helsinki-NLP/{TRANSLATOR}-{SRC_LANG}-{'ru'}")
print(2)
professions = set()
with open("/mt_gender/data/aggregates/en.txt", "r") as f:
    lines = f.readlines()
for line in lines:
    professions.add((line.split("\t")[-1]).strip())
print(3)

print("len before " + str(len(professions)))
def tokenize_professions(professions, tokenizer):
    tokenized_professions = []
    for p in professions:
        indices = tokenizer(p)['input_ids'][:-1]
        t = tokenizer.convert_ids_to_tokens(indices)
        if len(t) == 1:
            tokenized_professions.append(t[0])
    return tokenized_professions

he_tokenized_professions = tokenize_professions(professions,tokenizer_he)
de_tokenized_professions = tokenize_professions(professions,tokenizer_de)
ru_tokenized_professions = tokenize_professions(professions,tokenizer_ru)

print("he len before " + str(len(he_tokenized_professions)))
print("de len before " + str(len(de_tokenized_professions)))
print("ru len before " + str(len(ru_tokenized_professions)))







