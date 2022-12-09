import os
import torch
import nltk
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader

from gapbot import defaultbot
from gapdataset import GAPDataset
from utils import collate_examples, convert_to_results, calculate_offsets, equal_words_with_offset, WordWithOffset, WordWithOffsetAndScore, Coreference

is_noun = lambda pos: pos[:2] == "NN"

class Predict():

    def __init__(self):
        self.allpronouns = ["she", "her", "hers", "he", "him", "his"]
        self.columns = ["ID", "Text", "Pronoun", "Pronoun-offset", "A", "A-offset", "A-coref", "B", "B-offset", "B-coref", "URL"]
        os.environ["SEED"] = "33223"
        BERT_MODEL = "bert-large-uncased"
        UNCASED = True
        self.tokenizer = BertTokenizer.from_pretrained(
            BERT_MODEL,
            do_lower_case=UNCASED,
            never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[A]", "[B]", "[P]")
        )
        self.tokenizer.vocab["[A]"] = -1
        self.tokenizer.vocab["[B]"] = -1
        self.tokenizer.vocab["[P]"] = -1

        self.bot = defaultbot()

        # Load the best checkpoint
        self.bot.load_model("./cache/model_cache/best_10k_wd_lr_20-19e.pth")

    def predict(self, input_text):

        final_results = []

        full_text = input_text
        all_tokenized = list(set([word for word in nltk.word_tokenize(full_text)]))
        pronouns = list(set([word for word in all_tokenized if word.lower() in self.allpronouns]))

        all_pronouns_with_offsets = calculate_offsets(pronouns, full_text)

        print(all_pronouns_with_offsets)

        coreference = []
        predict_id = 1
        coref_id = 1

        for pronoun_with_offset in all_pronouns_with_offsets:
            pronoun = pronoun_with_offset.word
            pronoun_offsets = pronoun_with_offset.offsets
            for pronoun_offset in pronoun_offsets:

                without_pronoun = full_text[0 : pronoun_offset-1 :] + full_text[pronoun_offset+len(pronoun) : :]

                tokenized = nltk.word_tokenize(without_pronoun)
                words = list(set([word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]))

                print()

                all_nouns_with_offsets = calculate_offsets(words, full_text)

                print(all_nouns_with_offsets)

                print()

                for noun_with_offset in all_nouns_with_offsets:
                    for noun_offset in noun_with_offset.offsets:
                        for other_noun_with_offset in all_nouns_with_offsets:
                            for other_noun_offset in other_noun_with_offset.offsets:
                                if (other_noun_with_offset.word != noun_with_offset.word and other_noun_offset != noun_offset):
                                    noun = noun_with_offset.word
                                    other_noun = other_noun_with_offset.word
                                    coreference.append(["predict-" + str(predict_id), full_text, pronoun, pronoun_offset, noun, noun_offset, False, other_noun, other_noun_offset, False, "[Link]"])
                                    predict_id += 1

                if (len(coreference) == 0 and len(all_nouns_with_offsets) != 0):
                    coreference.append(["predict-" + str(predict_id), full_text, pronoun, pronoun_offset, all_nouns_with_offsets[0][0], all_nouns_with_offsets[0][1][0], False, all_nouns_with_offsets[0][0], all_nouns_with_offsets[0][1][0], False, "[Link]"])

                df_predict = pd.DataFrame(coreference, columns=self.columns)

                predict_ds = GAPDataset(df_predict, self.tokenizer)

                predict_loader = DataLoader(
                    predict_ds,
                    collate_fn = collate_examples,
                    batch_size=1,
                    pin_memory=True,
                    shuffle = False
                )

                # Extract predictions
                preds = self.bot.predict(predict_loader)

                # Create results file
                df_sub = pd.DataFrame(torch.softmax(preds, -1).cpu().numpy().clip(1e-3, 1-1e-3), columns=["A", "B", "NEITHER"])
                df_sub["ID"] = df_predict.ID
                df_sub.to_csv("prediction_results.csv", index=False)
                df_result = convert_to_results('prediction_results.csv')
                df_result.to_csv('prediction_converted.csv', index=False)
                
                df_sub["word1"] = df_predict.A
                df_sub["word2"] = df_predict.B
                df_sub["word1_offset"] = df_predict["A-offset"]
                df_sub["word2_offset"] = df_predict["B-offset"]
                df_result["word1"] = df_predict.A
                df_result["word2"] = df_predict.B
                df_result["word1_offset"] = df_predict["A-offset"]
                df_result["word2_offset"] = df_predict["B-offset"]
                print(df_sub)
                print()
                print(df_result)
                result = []

                for index,row in df_sub.iterrows():
                    word1_with_offset = WordWithOffset(row["word1"], row["word1_offset"])
                    word2_with_offset = WordWithOffset(row["word2"], row["word2_offset"])

                    contains_a = False
                    contains_b = False

                    for result_so_far in result:
                        if (equal_words_with_offset(result_so_far.word_with_offset, word1_with_offset)):
                            contains_a = True
                        if (equal_words_with_offset(result_so_far.word_with_offset, word2_with_offset)):
                            contains_b = True

                    if (not contains_a):
                        result.append(WordWithOffsetAndScore(word1_with_offset, 0))
                    if (not contains_b):
                        result.append(WordWithOffsetAndScore(word2_with_offset, 0))

                    mutable_result = result

                    for index,result_so_far in enumerate(result):
                        if (equal_words_with_offset(result_so_far.word_with_offset, word1_with_offset)):
                            new_score = result_so_far.score + row["A"]
                            mutable_result[index] = WordWithOffsetAndScore(word1_with_offset, new_score)
                        if (equal_words_with_offset(result_so_far.word_with_offset, word2_with_offset)):
                            new_score = result_so_far.score + row["B"]
                            mutable_result[index] = WordWithOffsetAndScore(word2_with_offset, new_score)

                    result = mutable_result

                sorted_list = sorted(result, key=lambda x: x.score, reverse=True)
                score_sum = 0
                for results in sorted_list:
                    score_sum += results.score
                print()
                final_results.append(Coreference(coref_id, WordWithOffset(pronoun, pronoun_offset)._asdict(), sorted_list[0].word_with_offset._asdict(), round(sorted_list[0].score/score_sum, 2))._asdict())
                coref_id += 1
                print("------ " + pronoun + " (" + str(pronoun_offset) + ") - " + str(score_sum) + "------")
                for final_result_nouns in sorted_list:
                    print(final_result_nouns.word_with_offset.word + " (" + str(final_result_nouns.word_with_offset.offset) + ") - " + str(final_result_nouns.score) + " - " + str(round(final_result_nouns.score/score_sum, 2)))

        return final_results