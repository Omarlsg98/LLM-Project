import json
import random

PUNCTUATION = [".", ","]

def load_questions_answers(filepath):
    f = open(filepath, "r")
    questions = []
    answers = []
    for line in f.readlines():
        parsed = json.loads(line)
        questions.append(parsed["question"])
        answer = parsed["answer"].split("\n")[-1].split(" ")[-1]
        answers.append(answer)
    f.close()
    return questions, answers

# removes the first phrase that begins with ", which" and ends with a punctuation
def remove_which_phrase(s):
    start_index = s.find(", which")
    if start_index == -1:
        return s, False
    post = s[start_index + 1:]
    next_punctuation = 100000000
    for punc in PUNCTUATION:
        ind = post.find(punc)
        if ind != -1 and ind < next_punctuation:
            next_punctuation = ind
    if next_punctuation == 100000000:
        return s, False
    processed = s[:start_index] + s[start_index + next_punctuation + 1:]
    return processed, True

# replace a random number with the word "some"
def replace_number(s):
    tokens = s.split(" ") # doesn't catch numbers followed by punctuation
    number_indices = []
    for (i, token) in enumerate(tokens):
        if token.isdigit():
            number_indices.append(i)
    
    if len(number_indices) == 0:
        return s, False

    random_index = random.randint(0, len(number_indices) - 1)
    tokens[number_indices[random_index]] = "some"
    return " ".join(tokens), True

random.seed(11667)
train_questions, train_answers = load_questions_answers("data/gsm8k/train.jsonl")
which_file = open("data/gsm8k/which.json", "w")
replace_file = open("data/gsm8k/replace.json", "w")
base_file = open("data/gsm8k/base.json", "w")
which_list = []
replace_list = []
base_list = []

for (i, question) in enumerate(train_questions):
    s, success = remove_which_phrase(question)
    if success:
        which_list.append({"question": s, "type": "not enough information", "base answer": train_answers[i]})
    else:
        if random.random() < 0.5:
            s, success = replace_number(question)
            if success:
                replace_list.append({"question": s, "type": "not enough information", "base answer": train_answers[i]})
        else:
            base_list.append({"question": question, "type": "enough information", "base answer": train_answers[i]})
json.dump(which_list, which_file)
json.dump(replace_list, replace_file)
json.dump(base_list, base_file)