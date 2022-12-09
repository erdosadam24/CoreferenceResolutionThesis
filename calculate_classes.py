import pandas as pd

he_pronouns = ["he", "him", "his"]
she_pronouns = ["she", "her", "hers"]

def calculate_classes(filename) :
    df = pd.read_csv(filename, delimiter="\t")

    a_count = b_count = n_count = she_count = he_count = 0

    for index, row in df.iterrows():
        if (row['A-coref']):
            a_count += 1
        elif (row['B-coref']):
            b_count += 1
        else:
            n_count += 1

        if (row['Pronoun'] in she_pronouns):
            she_count += 1
        else:
            he_count += 1

    print(f"A count: {a_count} - B count: {b_count} - NEITHER count: {n_count}")
    print(f"She count: {she_count} - He count: {he_count}")

calculate_classes("merged_10k.tsv")