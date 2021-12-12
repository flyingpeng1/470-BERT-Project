from typing import List, Tuple
import nltk
import sklearn
import transformers
import numpy as np
import pandas as pd

from qanta.ProjectServer import *

VOCAB_FILE = "../data/QuizBERT.vocab"
MODEL_FILE = "../data/QuizBERT.model"
BUZZER_FILE = "../data/QuizBERTBuzzer.model"
LINK_FILE = "../data/wiki_links.csv"

class QuizBowlModel:

    def __init__(self):
        self.guesser = Project_Guesser(VOCAB_FILE, MODEL_FILE)
        self.buzzer = Project_Buzzer(BUZZER_FILE, VOCAB_FILE, LINK_FILE)
        
        self.guesser.wait_for_load()
        self.buzzer.wait_for_load()


    def guess_and_buzz(self, question_text: List[str]) -> List[Tuple[str, bool]]:
        results = []
        for question in question_text:
            guess, buzz, kguess, kguess_scores, confidence = guess_and_buzz(self.guesser, self.buzzer, question)
            results.append((guess, buzz))

        return results