search_dir="/cs/usr/bareluz/gabi_labs/nematus_clean/debias_outputs/results/11-10-2022_15-08-28_1_token_professions_by_pca_false"
cd /cs/usr/bareluz/gabi_labs/nematus_clean/debias_files || return
for entry in "$search_dir"/*
do
  f="$(basename -- $entry .csv)"
  output_path=${search_dir}/tex_files
  mkdir -p ${output_path}
  if [[ $f == *"anti_accuracy"* ]]; then
    eval_method="Anti Accuracy"
  else
    eval_method="BLEU"
  fi

  if [[ $f == *"_he_"* ]]; then
    lang="Hebrew"
  elif [[ $f == *"_de_"* ]]; then
    lang="German"
  elif [[ $f == *"_de_"* ]]; then
    lang="German"
  elif [[ $f == *"_es_"* ]]; then
    lang="Spanish"
  elif [[ $f == *"_ru_"* ]]; then
    lang="Russian"
  fi
  exec > ${output_path}/${f}.tex
  exec 2>&1
  python tably.py ${search_dir}/${f}.csv --label Tab:${f} -c "${eval_method} scores of ${lang} translations using Opus-MT, with gender debias applied in the Encoder and the Decoder"
done