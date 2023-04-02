import argparse
import sys
import os
import re
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, TFAutoModel, TFAutoModelForQuestionAnswering

# suppress tensorflow warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--qg_model_name', type=str, default='mrm8488/t5-base-finetuned-question-generation-ap')
parser.add_argument('--qg_model_type', type=str, default='t5')
# parser.add_argument('--qa_model_name', type=str, default='MaRiOrOsSi/t5-base-finetuned-question-answering')
# parser.add_argument('--qa_model_type', type=str, default='t5')
parser.add_argument('--qa_model_name', type=str, default='distilbert-base-cased-distilled-squad')
parser.add_argument('--qa_model_type', type=str, default='bert')
parser.add_argument('--qg_model_path', type=str, default=None)
parser.add_argument('--qa_model_path', type=str, default=None)
parser.add_argument('--num_wrong_answers', type=int, default=2)
parser.add_argument('--input', type=str, default='input.txt')
parser.add_argument('--output', type=str, default='output.txt')

def read_input(input_file=None):
    if input_file is None:
        return sys.stdin.read()
    with open(input_file, 'r') as f:
        return f.read()

def write_output(output, output_file=None):
    if output_file is None:
        print(output)
    else:
        with open(output_file, 'w') as f:
            f.write(output)

def get_model(**kwargs):
    """
    Get model from either model_path or model_name.
    At least one of model_path or model_name must be provided.
    If both are provided, model_path will be used if it is downloaded model,
    otherwise model_name will be used to do download first and then save to model_path.

    Keyword arguments:
    - model_path -- path to model
    - model_name -- name of model
    - load_method -- method to load mode. Default: TFAutoModel.from_pretrained

    """
    if kwargs['model_path'] is None and kwargs['model_name'] is None:
        raise ValueError('Either model_path or model_name must be provided')
    if kwargs['model_path'] is None:
        return kwargs.get('load_method', TFAutoModel.from_pretrained)(kwargs['model_name'])
    else:
        # check if model_path is downloaded model
        if os.path.exists(kwargs['model_path']):
            return kwargs.get('load_method', TFAutoModel.from_pretrained)(kwargs['model_path'])
        else:
            if kwargs['model_name'] is None:
                raise ValueError('model_name must be provided if model_path is not downloaded model')
            return kwargs.get('load_method', TFAutoModel.from_pretrained)(kwargs['model_name'])

def save_model(model, model_path):
    model.save_pretrained(model_path)

def permute_sentences(input_text, seed=42):
    np.random.seed(seed)
    sentences = input_text.split('.')
    np.random.shuffle(sentences)
    return '. '.join(sentences)

def permute_words(input_text, seed=42):
    np.random.seed(seed)
    words = input_text.split()
    np.random.shuffle(words)
    return ' '.join(words)

def remove_answer_ngrams(input_text, answer, seed=42):
    np.random.seed(seed)
    words = input_text.split()
    answer_words = answer.split()
    ngram_size = len(answer_words) % 3
    if ngram_size == 0:
        ngram_size = 1
    answer_ngrams = [' '.join(answer_words[i:i+ngram_size]) for i in range(len(answer_words)-ngram_size+1)]
    for ngram in answer_ngrams:
        if ngram in input_text and np.random.rand() < 0.7:
            input_text = input_text.replace(ngram, '')
    return input_text

def is_yes_no_answer(answer):
    yes_regex = re.compile(r'\b' + 'yes' + r'\b', re.IGNORECASE)
    no_regex = re.compile(r'\b' + 'no' + r'\b', re.IGNORECASE)
    if yes_regex.search(answer):
        return True, 'yes'
    elif no_regex.search(answer):
        return True, 'no'
    return False, None

def pretty_print(input_text, question, correct_answer, wrong_answers):
    output = f'{question}\n'
    letters = 'abcdefghijklmnopqrstuvwxyz'.upper()
    wrong_answers = [a for a in wrong_answers if a]
    choice_letters = letters[:len(wrong_answers) + 1]
    answers = [correct_answer] + wrong_answers
    np.random.shuffle(answers)
    for i, answer in enumerate(answers):
        output += f'{choice_letters[i]}) {answer}\n'
    output += f'Answer: {choice_letters[answers.index(correct_answer)]}) {correct_answer}\n'
    return output

def is_numeric_answer(answer):
    try:
        float(answer)
        return True
    except ValueError:
        return False

def random_wrong_numeric_answers(correct_answer, num=3, seed=42):
    correct_answer = float(correct_answer)
    wrong_answers = []
    wrong_answers.append(correct_answer + np.random.randint(1, 10))
    wrong_answers.append(correct_answer * 10)
    wrong_answers.append(correct_answer / 10)
    wrong_answers.append(correct_answer - np.random.randint(1, 10))
    wrong_answers.append(correct_answer * np.random.randint(1, 10))
    if len(wrong_answers) > num:
        # shuffle wrong answers
        np.random.seed(seed)
        np.random.shuffle(wrong_answers)
        wrong_answers = wrong_answers[:num]
    return wrong_answers

def process_yes_no_or_numeric_answer(answer, num=3, seed=42):
    is_yes_no = is_yes_no_answer(answer)
    if is_yes_no[0]:
        if is_yes_no[1] == 'yes':
            return True, ['no']
        else:
            return True, ['yes']
    is_numeric = is_numeric_answer(answer)
    if is_numeric:
        return True, random_wrong_numeric_answers(answer, num=num, seed=seed)
    return False, []

class QA(ABC):
    @abstractmethod
    def generate_answer(self, input_text, question):
        ...

    @abstractmethod
    def generate_wrong_answers(self, input_text, question, correct_answer=None):
        ...

class T5QG:
    def __init__(self, model_name, model_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = get_model(model_path=model_path, model_name=model_name, load_method=TFAutoModelForSeq2SeqLM.from_pretrained)

    def generate_question(self, input_text):
        _input = self.tokenizer('context: ' + input_text, return_tensors='tf', padding=True, truncation=True)
        output = self.model.generate(**_input)
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return output

class T5QA(QA):
    def __init__(self, model_name, model_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = get_model(model_path=model_path, model_name=model_name, load_method=TFAutoModelForSeq2SeqLM.from_pretrained)

    def generate_answer(self, question, input_text):
        _input = self.tokenizer(question + 'content: ' + input_text, return_tensors='tf', padding=True, truncation=True)
        output = self.model.generate(**_input)
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return output

    def generate_wrong_answers(self, question, input_text, correct_answer=None, seed=42, num=3):
        if correct_answer is None:
            correct_answer = self.generate_answer(question, input_text)
        wrong_answers = []
        is_yes_no_or_numeric, wrong_answers = process_yes_no_or_numeric_answer(correct_answer, num=num, seed=seed)
        if is_yes_no_or_numeric:
            return wrong_answers
        for i in range(num):
            wrong_answers.append(self.generate_wrong_answer(question, input_text, correct_answer, seed=seed+i))
        wrong_answers = list(set(wrong_answers))
        return wrong_answers

    def generate_wrong_answer(self, question, input_text, correct_answer, seed=42):
        if len(correct_answer.split()) == 1:
            input_text = permute_words(input_text, seed=seed)
        else:
            input_text = remove_answer_ngrams(input_text, correct_answer, seed=seed)
        _input = self.tokenizer(question + 'content: ' + input_text, return_tensors='tf', padding=True, truncation=True)
        output = self.model.generate(**_input)
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return output

class BERTQA(QA):
    def __init__(self, model_name, model_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = get_model(model_path=model_path, model_name=model_name, load_method=TFAutoModelForQuestionAnswering.from_pretrained)
        self.output = None
        self._input = None
    
    def generate_answer(self, question, input_text):
        self._input = self.tokenizer(question, input_text, return_tensors='tf', padding=True, truncation=True)
        self.output = self.model(**self._input)
        answer_start_index = int(tf.math.argmax(self.output.start_logits, axis=-1)[0])
        answer_end_index = int(tf.math.argmax(self.output.end_logits, axis=-1)[0])
        predict_answer_tokens = self._input.input_ids[0, answer_start_index : answer_end_index + 1]
        answer = self.tokenizer.decode(predict_answer_tokens)
        return answer

    def generate_wrong_answers(self, question, input_text, correct_answer=None, seed=42, num=3):
        if self.output is None:
            self.generate_answer(question, input_text)
        wrong_answers = []
        is_yes_no_or_numeric, wrong_answers = process_yes_no_or_numeric_answer(correct_answer, num=num, seed=seed)
        if is_yes_no_or_numeric:
            return wrong_answers
        else:
            input_text = input_text.replace(correct_answer, '')
            for _ in range(num):
                wrong_answers.append(self.generate_answer(question, input_text))
                input_text = input_text.replace(wrong_answers[-1], '')
            wrong_answers = [a for a in wrong_answers if not re.match(r'[ ]|\.', a.strip())]
            return wrong_answers

def init_models(args):
    if args.qa_model_type == 't5':
        qa_model = T5QA(args.qa_model_name, args.qa_model_path)
    elif args.qa_model_type == 'bert':
        qa_model = BERTQA(args.qa_model_name, args.qa_model_path)
    else:
        raise ValueError('qa_model_type must be t5')
    if args.qg_model_type == 't5':
        qg_model = T5QG(args.qg_model_name, args.qg_model_path)
    else:
        raise ValueError('qg_model_type must be t5')
    return qa_model, qg_model
    


def main(args: argparse.Namespace, input_text=None):
    qa_model, qg_model = init_models(args)
    if input_text is None:
        input_text = read_input(args.input)
    question = qg_model.generate_question(input_text)
    correct_answer = qa_model.generate_answer(question, input_text)
    wrong_answers = qa_model.generate_wrong_answers(question, input_text, correct_answer, num=args.num_wrong_answers, seed=args.seed)
    output = pretty_print(input_text, question, correct_answer, wrong_answers)
    write_output(output, args.output)
    return output

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

