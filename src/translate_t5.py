import os
import torch.nn as nn
from sacrebleu.metrics import BLEU
import torch
from tqdm import tqdm
# from accelerate import init_empty_weights
import shutil
os.environ['TRANSFORMERS_CACHE'] = '/cs/snapless/gabis/bareluz'
from transformers import MT5ForConditionalGeneration, AutoTokenizer
# refs_path_de ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_de_5.8/newstest2012.de"
# file_path_de ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_de_5.8/newstest2012.en"

refs_path_he ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_he_20.07.21/dev.he"
file_path_he ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_he_20.07.21/dev.en"

# refs_path_ru ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_ru_30.11.20/newstest2019-enru.ru"
# file_path_ru ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_ru_30.11.20/newstest2019-enru.en"

batch_size = 200  # Adjust the batch size as needed


tokenizer = AutoTokenizer.from_pretrained("google/mt5-large",src_lang="en",tgt_lang="he")
device = "cuda" if torch.cuda.is_available() else "cpu"
bleu = BLEU()
def translate_t5(task_prefix, file_path, refs_path, translations_path):
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    with open(file_path, 'r') as f:
        sentences = f.readlines()
    translations = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i:i + batch_size]
        # inputs = tokenizer([task_prefix + sentence for sentence in batch_sentences], return_tensors="pt", padding=True)
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, prefix=task_prefix)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
            max_length=512
        )
        batch_translations = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        translations.extend(batch_translations)
    with open(translations_path,'w', encoding="utf-8") as f2:
        for t in translations:
            f2.write(t+"\n")
    with open(refs_path,'r') as f:
        refs = f.readlines()

    print("bleu t5 "+task_prefix )
    print(bleu.corpus_score(translations, [refs]))
if __name__ == '__main__':

    # task_prefix_de = "translate English to German: "
    task_prefix_he = "translate English to Hebrew:"
    # task_prefix_ru = "translate English to Russian: "
    # translate_t5(task_prefix_de, file_path_de,refs_path_de,"translations_de_t5.txt")
    translate_t5(task_prefix_he, file_path_he,refs_path_he,"translations_he_t5.txt")
    # translate_t5(task_prefix_ru, file_path_ru,refs_path_ru,"translations_ru_t5.txt")
#
