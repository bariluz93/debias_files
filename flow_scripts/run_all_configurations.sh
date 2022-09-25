#!/bin/bash
set -e


SHORT=c,p,t,a,b,e,w:,m:,h
LONG=collect_embedding_table,preprocess,translate,debias_encoder,beginning_decoder_debias,end_decoder_debias,words_to_debias,model:,help
OPTS=$(getopt -a -n debias --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

collect_embedding_table=""
preprocess=""
translate=""
debias_encoder=""
beginning_decoder_debias=""
end_decoder_debias=""
words_to_debias=""
model=""
while :
do
  case "$1" in
    -c | --collect_embedding_table )
      collect_embedding_table="-c"
      shift 1
      ;;
    -p | --preprocess )
      preprocess="-p"
      shift 1
      ;;
    -t | --translate )
      translate="-t"
      shift 1
      ;;
    -a | --debias_encoder )
      debias_encoder="-a"
      shift 1
      ;;
    -b | --beginning_decoder_debias )
      beginning_decoder_debias="-b"
      shift 1
      ;;
    -e | --end_decoder_debias )
      end_decoder_debias="-e"
      shift 1
      ;;
    -w | --words_to_debias )
      words_to_debias="$2"
      shift 2
      ;;
    -m | --model )
      model="$2"
      shift 2
      ;;
    -h | --help)
      echo "usage:
Mandatory arguments:
  -m, --model                     the translation model. 0=Nematus, 1=EasyNMT .
Optional arguments:
  -c, --collect_embedding_table   collect embedding table .
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
### correctness checks

#check model is given
if [ "$model" == "" ]; then
  echo missing argument model
  exit 1
fi
# check model has correct values
case $model in
    0|1) echo ;;
    *) echo "argument model can get only the values 0 for Nematus or 1 to easyNMT"
       exit 1;;
esac

#if words_to_debias is not given, ONE_TOKEN_PROFESSIONS = 1 is selected
if [ "$words_to_debias" == "" ]; then
  words_to_debias="1"
fi

# check words_to_debias has the correct values
case $words_to_debias in
    0|1|2) echo ;;
    *) echo "argument words_to_debias can get only the following values ALL_VOCAB = 0, ONE_TOKEN_PROFESSIONS = 1, ALL_PROFESSIONS = 2"
       exit 1;;
esac

# if none of debias_encoder, beginning_decoder_debias, end_decoder_debias is selected, debias_encoder is selected defaultly
if [ "$debias_encoder" == "" ] && [ "$beginning_decoder_debias" == "" ] && [ "$end_decoder_debias" == "" ]; then
  debias_encoder="-a"
fi

if [ "$model" == '1' ]; then
  if [[ "$collect_embedding_table" == "-c" || "$preprocess" == "-p" ]]; then
    echo cannot pass --collect_embedding_table and --preprocess with easyNMT
    exit 1
  fi
fi
### done correctness checks

scripts_dir=`pwd`
# the dst language here doesn't matter
source ${scripts_dir}/consts.sh ru 0 ${model}

echo "#################### cleanup ####################"
nematus_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus
python ${debias_files_dir}/cleanup.py ${collect_embedding_table} ${translate}


#if [ "${model_str}" == "EASY_NMT" ]; then
#  echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#  sh run_all_flows.sh -l es -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias}
#  echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ es 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#  sh run_all_flows.sh -l es -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias}
#fi
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}"
sh run_all_flows.sh -l de -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ de 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh run_all_flows.sh -l de -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh run_all_flows.sh -l he -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ he 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh run_all_flows.sh -l he -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh run_all_flows.sh -l ru -d 0 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ru 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
sh run_all_flows.sh -l ru -d 1 -m ${model} ${collect_embedding_table} ${preprocess} ${translate} ${debias_encoder} ${beginning_decoder_debias} ${end_decoder_debias} -w ${words_to_debias}



echo "#################### write results to table ####################"
source /cs/usr/bareluz/gabi_labs/nematus_clean/nematus_env3/bin/activate
python ${debias_files_dir}/write_results_to_table.py -m ${model}