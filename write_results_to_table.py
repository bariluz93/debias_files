import argparse
import re
from consts import LANGUAGE_STR_MAP,DebiasMethod, DEBIAS_FILES_HOME, OUTPUTS_HOME, TranslationModels
import pandas as pd
import numpy as np
import json
import csv
import math
from datetime import datetime
from pathlib import Path
import ast

INLP = "INLP"

HARD_DEBIAS = "Hard Debias"

LANGUAGES =LANGUAGE_STR_MAP.values()
DEBIAS_METHODS = [d.value for d in DebiasMethod]

def get_translation_results(file_name):
    result = {}
    with open(file_name) as f:
        lines = f.readlines()
        debiased_line = lines.index("debiased\n")+1
        result["debiased"] = float(lines[debiased_line].split(" ")[2])
        non_debiased_line = lines.index("non debiased\n") + 1
        result["non_debiased"] = float(lines[non_debiased_line].split(" ")[2])
    return result
def get_gender_results(file_name):
    result = {}
    with open(file_name) as f:
        line = f.readline()
        line_num=1
        debiased_found = False
        non_debiased_found = False

        while line:
            if line.__contains__("*debiased results*"):
                debiased_found = True
            if line.__contains__("*non debiased results*"):
                non_debiased_found = True
            # get general accuracy
            match = re.search("{\"acc\": (.*), \"f1_male\": .*, \"f1_female\": .*, \"unk_male\": .*, \"unk_female\": .*, \"unk_neutral\": .*}",line)
            if match:
                accuracy = match.group(1)
                if debiased_found:
                    result["debiased"] = float(accuracy)
                elif non_debiased_found:
                    result["non_debiased"] = float(accuracy)
            #get professions accuracy
            if line.startswith("professions accuracies"):
                line = f.readline()
                line_num += 1
                if debiased_found:
                    professions_accuracies_debiased = ast.literal_eval(line)
                    debiased_found = False
                elif non_debiased_found:
                    professions_accuracies_non_debiased = ast.literal_eval(line)
                    non_debiased_found =False
            line = f.readline()
            line_num+=1
    return result,professions_accuracies_debiased,professions_accuracies_non_debiased

def get_all_results(files_dict):

    results = {}
    professions_results = {}
    for language in LANGUAGES:
        results[language] = {}
        for debias_method in DEBIAS_METHODS:
            results[language][debias_method] = {}
            results[language][debias_method]["translation"] = get_translation_results(result_files[language][debias_method]["translation"])
            results[language][debias_method]["gender"],professions_accuracies_debiased,professions_accuracies_non_debiased = get_gender_results(result_files[language][debias_method]["gender"])
            for p in professions_accuracies_debiased.keys():
                if p not in professions_results:
                    professions_results[p] = {}
                if language not in professions_results[p]:
                    professions_results[p][language] = {}

                professions_results[p][language][debias_method]=professions_accuracies_debiased[p]-professions_accuracies_non_debiased[p]
    print(json.dumps(results, sort_keys=True, indent=4))
    methods_results = []
    for debias_method in DEBIAS_METHODS:
        orig_results = []
        method_results = []
        for lang in LANGUAGES:
            orig_bleu = results[lang][0]["translation"]["non_debiased"]
            orig_anti_accuracy = results[lang][0]["gender"]["non_debiased"]
            orig_results += [orig_bleu, -math.inf, orig_anti_accuracy, -math.inf]
            method_bleu = results[lang][debias_method]["translation"]["debiased"]
            method_anti_accuracy = results[lang][debias_method]["gender"]["debiased"]

            method_results += [method_bleu, method_bleu-orig_bleu, method_anti_accuracy, method_anti_accuracy-orig_anti_accuracy]
        methods_results+=[method_results]
    results =np.around([orig_results, methods_results[0], methods_results[1]], 2)
    results[results==-math.inf] = None

    professions_results_table = {}
    for p in professions_results.keys():
        professions_results_table[p] = [professions_results[p][language][debias_method]for language in LANGUAGES for debias_method in DEBIAS_METHODS]

    return results,professions_results_table

def write_professions_results_to_csv(professions_results, model):
    headers = [None, "Russian", None, "German",None, "Hebrew", None]
    sub_headers = [None] + [HARD_DEBIAS, INLP] * 3
    index = [[p] for p in professions_results.keys()]
    data = np.append(index, list(professions_results.values()), axis=1)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    Path(DEBIAS_FILES_HOME+"results").mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_HOME+"results/professions_results_"+model+"_"+dt_string+".csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(sub_headers)
        writer.writerows(data)
def write_results_to_csv(results, model):
    headers = [None, "Russian", None, None, None, "German",None, None, None,  "Hebrew", None, None, None]
    sub_headers = [None]+["Bleu", "delta Bleu", "anti accuracy", "delta anti accuracy"]*3
    index = [["Original"], [HARD_DEBIAS], [INLP]]
    data = np.append(index, results, axis=1)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    Path(DEBIAS_FILES_HOME+"results").mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_HOME+"results/results_"+model+"_"+dt_string+".csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(sub_headers)
        writer.writerows(data)
def write_results_to_table(results, model):


    iterables = [["Russian", "German",  "Hebrew"], ["Bleu", "delta Bleu", "anti accuracy", "delta anti accuracy"]]
    index =pd.MultiIndex.from_product(iterables)
    df = pd.DataFrame(results, index=["Original", HARD_DEBIAS, INLP], columns=index)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    with open(OUTPUTS_HOME+"results/results_"+model+"_"+dt_string+".tex", 'w') as f:
        f.write(df.to_latex())
    pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-m', '--model', type=str, required=True,
            help="the translation model:\n"
                 "0 = Nematus\n1 = Easy NMT")
    args = parser.parse_args()
    model = TranslationModels[int(args.model)]
    if model=='NEMATUS':
        LANGUAGES = list(LANGUAGES)
        LANGUAGES.remove('es')
    if model=='EASY_NMT':
        LANGUAGES = list(LANGUAGES)
        LANGUAGES.remove('es')
    result_files = {}
    for language in LANGUAGES:
        result_files[language] = {}
        for debias_method in DEBIAS_METHODS:
            result_files[language][debias_method] = {}
            result_files[language][debias_method]["translation"] = OUTPUTS_HOME + "en-" + language + "/debias/translation_evaluation_"+ language + "_"+str(debias_method)+"_"+model+".txt"
            result_files[language][debias_method]["gender"] = OUTPUTS_HOME + "en-" + language + "/debias/gender_evaluation_" + language + "_"+str(debias_method)+"_"+model+".txt"
    res,professions_results_table = get_all_results(result_files)
    for i in res:
        print(i)
    write_results_to_csv(res, model)
    write_professions_results_to_csv(professions_results_table, model)