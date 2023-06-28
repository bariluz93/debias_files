import pandas as pd
import glob
import os
from pathlib import Path
import numpy as np
import argparse
# import sys
# sys.path.append("../..")
# from testSignificanceNLP.testSignificance import main as testSignificanceMain
import sacrebleu

RESULTS_DIR="/cs/usr/bareluz/gabi_labs/nematus_clean/debias_outputs/results_for_statistic_significance/"
LANGUAGES=["he","de","ru"]
DATA_DIR="/cs/snapless/gabis/bareluz/debias_nmt_data/data/"
LANGUAGE_TO_DATA_DIR = {"he":DATA_DIR+"en_he_20.07.21/dev.he", "de":DATA_DIR+"en_de_5.8/newstest2012.de", "ru":DATA_DIR+"en_ru_30.11.20/newstest2019-enru.ru"}
def get_gender_statistical_significance(files):
    for file in files:
        filename = Path(file).stem
        df = pd.read_csv(file, header=[0])
        os.makedirs(os.path.dirname(RESULTS_DIR+"ready_for_statistical_significance_test/gender_scores/"), exist_ok=True)

        res = np.where((df['Target gender'] == df['Predicted gender']), "1", "0")
        with open(RESULTS_DIR+"ready_for_statistical_significance_test/gender_scores/"+filename+".txt",'w') as f:
            for r in res:
                f.write(r+"\n")
def get_file_pairs(score_type,category=""):
    category_dir=""
    if category:
        category_dir="per_"+category+"/"

    all_files = glob.glob(RESULTS_DIR+"ready_for_statistical_significance_test/"+score_type+"/"+category_dir+"*")
    non_debiased_files = [f for f in all_files if f.__contains__("non_debiased")]
    pairs = [(f,f.replace("non_debiased","debiased")) for f in non_debiased_files]
    with open(RESULTS_DIR+"ready_for_statistical_significance_test/pairs_"+category+"_"+score_type+".txt","w") as f:
        for pair in pairs:
            f.write(pair[0]+","+pair[1]+"\n")
    return pairs
def get_translation_statistical_significance(files,lang):
    with open(LANGUAGE_TO_DATA_DIR[lang],"r") as f:
        refs = f.readlines()
    dirname=RESULTS_DIR+"ready_for_statistical_significance_test/blue_scores/"
    os.makedirs(dirname, exist_ok=True)
    for file in files:
        with open(file) as f, open(dirname+lang+"-"+Path(file).stem+".txt","w") as f2:
            lines = f.readlines()
            for i in range(len(lines)):
                score = sacrebleu.sentence_bleu(lines[i], [refs[i]],  smooth_method='exp').score
                f2.write(str(score)+"\n")
def merge_scores(score_type,category):
    files_pairs = get_file_pairs(score_type)
    if category == "language":
        possible_representing_str =  ["he-","de-","ru-"]
    elif category == "debias_method":
        possible_representing_str = ["_0_", "_1_"]
    elif category == "debias_location":
        possible_representing_str = ["_A","_B","_C"]
    elif category == "all":
        possible_representing_str = [""]

    else:
        print("bad category")
        exit(1)
    for s in possible_representing_str:
        print("bar")
        # print("*****")
        # print(s)
        category_dir =RESULTS_DIR+"ready_for_statistical_significance_test/"+score_type+"/per_"+category+"/"
        os.makedirs(category_dir, exist_ok=True)
        filename_debiased=category_dir+"debiased_"+s+".txt"
        filename_non_debiased=category_dir+"non_debiased_"+s+".txt"
        data_debiased=""
        data_non_debiased=""
        for pair in files_pairs:
            if pair[0].__contains__(s):
                with open(pair[0],'r') as fnd, open(pair[1],'r') as fd:
                    data_debiased+= fd.read()
                    # data_debiased+="\n"
                    data_non_debiased+= fnd.read()
                    # data_non_debiased+="\n"
        with open(filename_debiased,'w') as fd,open(filename_non_debiased,'w') as fnd:
            fd.write(data_debiased[:-1])
            fnd.write(data_non_debiased[:-1])

def main(category):
    for lang in LANGUAGES:
        results_gender = glob.glob(RESULTS_DIR+lang+"/results/*")
        get_gender_statistical_significance(results_gender)
        results_translation = glob.glob(RESULTS_DIR+lang+"/translations/*")
        results_translation = [f for f in results_translation if not f.__contains__("anti") ]
        get_translation_statistical_significance(results_translation, lang)


    merge_scores("gender_scores",category)
    get_file_pairs("gender_scores",category)

    merge_scores("blue_scores",category)
    get_file_pairs("blue_scores",category)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--category',
        help="category for which to merge result files. can be 'language', 'debias_method', 'debias_location', 'all'")
    args = parser.parse_args()
    category=args.category
    main(category)