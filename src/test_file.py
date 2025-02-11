import sys
import glob
import csv
import pathlib

def check_results_are_good(results_dir):
    files =list(pathlib.Path(results_dir).glob('*.csv'))
    conditions = []
    for file in files:
        if (file.name).__contains__("_de_") and (file.name).__contains__("anti_accuracy"):
            with open(str(file)) as f:
                spamreader = csv.reader(f)
                rows = [row for row in spamreader]
                conditions.append(float((rows[1])[2])>=59)
                conditions.append(float((rows[1])[3])>=54)
                conditions.append(float((rows[2])[3])>=60)
                conditions.append(float((rows[3])[3])>=61)
        if (file.name).__contains__("_he_") and (file.name).__contains__("anti_accuracy"):
            with open(str(file)) as f:
                spamreader = csv.reader(f)
                rows = [row for row in spamreader]
                conditions.append(float((rows[1])[2])>=48)
                conditions.append(float((rows[1])[3])>=42)
                conditions.append(float((rows[2])[3])>=46)
                conditions.append(float((rows[3])[3])>=46)
    return all(conditions)

print(check_results_are_good(sys.argv[1]))
sys.exit(0)