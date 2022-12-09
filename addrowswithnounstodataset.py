import os
import pandas as pd
from utils import calculate_offsets, equal_words_with_offset, WordWithOffset
import nltk
from random import random
from calculate_classes import calculate_classes

df = pd.read_csv("gap-test.tsv", delimiter="\t")

new_rows = []

testid = 2001
count = 1

is_noun = lambda pos: pos[:2] == "NN"

def getbooltext(trueorfalse):
    return "TRUE" if trueorfalse else "FALSE"

for index, row in df.iterrows():
    toBeAdded = []
    text = row['Text']
    link = row['URL']
    pronoun = WordWithOffset(row['Pronoun'], row['Pronoun-offset'])
    a = WordWithOffset(row['A'], row['A-offset'])
    b = WordWithOffset(row['B'], row['B-offset'])

    truenouns = []

    if (row['A-coref'] == True):
        if (" " in a.word):
            split = a.word.split(" ")
            truenouns.append(WordWithOffset(split[0], a.offset))
            truenouns.append(WordWithOffset(split[1], a.offset+len(split[0])+1))
        else:
            truenouns.append(a)
    if (row['B-coref'] == True):
        if (" " in b.word):
            split = b.word.split(" ")
            truenouns.append(WordWithOffset(split[0], b.offset))
            truenouns.append(WordWithOffset(split[1], b.offset+len(split[0])+1))
        else:
            truenouns.append(b)
    
    without_pronoun = text[0 : pronoun.offset-1 :] + text[pronoun.offset+len(pronoun) : :]

    tokenized = nltk.word_tokenize(without_pronoun)
    nouns = list(set([word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]))

    all_nouns_with_offsets = calculate_offsets(nouns, text)

    for noun_with_offset in all_nouns_with_offsets:
        for noun_offset in noun_with_offset.offsets:
            for other_noun_with_offset in all_nouns_with_offsets:
                for other_noun_offset in other_noun_with_offset.offsets:
                    if (other_noun_with_offset.word != noun_with_offset.word and other_noun_offset != noun_offset):
                        noun = noun_with_offset.word
                        other_noun = other_noun_with_offset.word
                        noun_is_true = False
                        other_noun_is_true = False
                        for truenoun in truenouns:
                            if (equal_words_with_offset(truenoun, WordWithOffset(noun, noun_offset))):
                                noun_is_true = True
                            if (equal_words_with_offset(truenoun, WordWithOffset(other_noun, other_noun_offset))):
                                other_noun_is_true = True
                        neither = not noun_is_true and not other_noun_is_true
                        xor = (noun_is_true and not other_noun_is_true) or (not noun_is_true and other_noun_is_true)
                        if (neither and random() < 0.0035) or (xor and random() < 0.08):
                            new_rows.append(["test-" + str(testid), text, pronoun.word, pronoun.offset, noun, noun_offset, getbooltext(noun_is_true), other_noun, other_noun_offset, getbooltext(other_noun_is_true), link])
                            testid += 1
    
    os.system('cls')
    print("--- " + str(count) + "/2000 ---")
    count += 1

allTogether = pd.DataFrame(new_rows, columns=df.columns)

allTogether.to_csv('newRows.csv', index=False, sep='\t')

calculate_classes('newRows.csv')