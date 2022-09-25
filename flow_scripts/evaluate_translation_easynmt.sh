#!/bin/bash
set -e
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=bar.iluz@mail.huji.ac.il
#SBATCH --output=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/slurm/evaluate_translation-%j.out
echo "**************************************** in evaluate_translation.sh ****************************************"

SHORT=l:,d:,p,t,a,b,e,w:,h
LONG=language:,debias_method:,preprocess,translate,debias_encoder,beginning_decoder_debias,end_decoder_debias,words_to_debias,help
OPTS=$(getopt -a -n debias --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

preprocess=false
translate=false
debias_encoder=0
beginning_decoder_debias=0
end_decoder_debias=0
words_to_debias=""

while :
do
  case "$1" in
    -l | --language )
      language="$2"
      shift 2
      ;;
    -d | --debias_method )
      debias_method="$2"
      shift 2
      ;;
    -p | --preprocess )
      preprocess=true
      shift 1
      ;;
    -t | --translate )
      translate=true
      shift 1
      ;;
    -a | --debias_encoder )
      debias_encoder=1
      shift 1
      ;;
    -b | --beginning_decoder_debias )
      beginning_decoder_debias=1
      shift 1
      ;;
    -e | --end_decoder_debias )
      end_decoder_debias=1
      shift 1
      ;;
    -w | --words_to_debias )
      words_to_debias="$2"
      shift 2
      ;;
    -h | --help)
      echo "usage:
Mandatory arguments:
  -l, --language                  the destination translation language. RUSSIAN = 0, GERMAN = 1,HEBREW = 2,SPANISH = 3 .
  -d, --debias_method             the debias method. BOLUKBASY = 0, NULL_IT_OUT = 1 .
Optional arguments:
  -p, --preprocess                preprocess the anti dataset .
  -t, --translate                 translate the entire dataset .
  -a, --debias_encoder            debias the encoder .
  -b, --beginning_decoder_debias  debias the decoder inputs .
  -e, --end_decoder_debias        debias the decoder outputs .
  -w, --words_to_debias           set of words to debias. ALL_VOCAB = 0, ONE_TOKEN_PROFESSIONS = 1, ALL_PROFESSIONS = 2 .
  -h, --help                      help message .
if none of debias_encoder, beginning_decoder_debias, end_decoder_debias is selected, debias_encoder is selected defaultly
if words_to_debias is not given, ONE_TOKEN_PROFESSIONS = 1 is selected"
      exit 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      exit 1;;
  esac
done

scripts_dir=`pwd`
source ${scripts_dir}/consts.sh ${language} ${debias_method} 1

#################### translate some dataset to test translation quality ####################
outputh_path_debiased=${debias_outputs_dir}/${language_dir}/output/debiased_${debias_method}_${model_str}.out.tmp
outputh_path_non_debiased=${debias_outputs_dir}/${language_dir}/output/non_debiased_${debias_method}_${model_str}.out.tmp
config_debiased="{'USE_DEBIASED': 1, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': ${debias_method}, 'TRANSLATION_MODEL': 1, 'DEBIAS_ENCODER': ${debias_encoder}, 'BEGINNING_DECODER_DEBIAS': ${beginning_decoder_debias}, 'END_DECODER_DEBIAS': ${end_decoder_debias}, 'WORDS_TO_DEBIAS': ${words_to_debias}}"
config_non_debiased="{'USE_DEBIASED': 0, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': ${debias_method}, 'TRANSLATION_MODEL': 1, 'DEBIAS_ENCODER': ${debias_encoder}, 'BEGINNING_DECODER_DEBIAS': ${beginning_decoder_debias}, 'END_DECODER_DEBIAS': ${end_decoder_debias}, 'WORDS_TO_DEBIAS': ${words_to_debias}}"

if [ $translate = true ]; then
  echo "#################### translate debiased ####################"
#  echo "python ${nematus_dir}/nematus/translate.py -i $input_path -m  $model_dir -k 12 -n -o ${outputh_path_debiased} -c ${config_debiased}"
  python ${debias_files_dir}/translate_easynmt.py \
       -i "$input_path" \
       -o "${outputh_path_debiased}" \
       -c "${config_debiased}"

  echo "#################### translate non debiased ####################"
  python ${debias_files_dir}/translate_easynmt.py \
       -i "$input_path" \
       -o "${outputh_path_non_debiased}" \
       -c "${config_non_debiased}"
fi
echo "#################### evaluate translation quality ####################"
output_result_path=${debias_outputs_dir}/${language_dir}/debias/translation_evaluation_${dst_language}_${debias_method}_${model_str}.txt
exec > ${output_result_path}
exec 2>&1
python ${debias_files_dir}/evaluate_translation.py \
     -c "{'USE_DEBIASED': 0, 'LANGUAGE': ${language_num}, 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': ${debias_method}, 'TRANSLATION_MODEL': 1, 'DEBIAS_ENCODER': ${debias_encoder}, 'BEGINNING_DECODER_DEBIAS': ${beginning_decoder_debias}, 'END_DECODER_DEBIAS': ${end_decoder_debias}, 'WORDS_TO_DEBIAS': ${words_to_debias}}"