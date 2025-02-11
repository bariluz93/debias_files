import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sacrebleu.metrics import BLEU
import torch.nn as nn
from tqdm import tqdm
file_path_de ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_de_5.8/newstest2012.en"
refs_path_de ="/cs/snapless/gabis/bareluz/debias_nmt_data/data/en_de_5.8/newstest2012.de"
tokenizer = T5Tokenizer.from_pretrained("t5-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
bleu = BLEU()
batch_size = 2  # Adjust the batch size as needed
def translate_t5(task_prefix, file_path,refs_path,translations_path):
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    with open(file_path, 'r') as f:
        sentences = f.readlines()
    translations = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output_sequences = model.module.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
        )
        batch_translations = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        translations.extend(batch_translations)
    with open(translations_path, 'w', encoding="utf-8") as f2:
        for t in translations:
            f2.write(t + "\n")
    with open(refs_path, 'r') as f:
        refs = f.readlines()

    print("bleu t5 " + task_prefix)
    print(bleu.corpus_score(translations, [refs]))

if __name__ == '__main__':
    translate_t5("translate English to German: ", file_path_de,refs_path_de,"translations_de_t5.txt")
#
