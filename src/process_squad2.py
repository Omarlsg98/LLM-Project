import json
import spacy
import random

"""
Format
top level has two keys: "version" and "data"
only one that matters is "data"
"data" contains a list, each element is a dictionary with keys "title" and "paragraphs"
each paragraph is also a list, each element is a dictionary with keys "qas" and "context"
"context" contains the context paragraph for all the questions
"qas" is a list, each element is a dictionary with keys "question" "id" "answers" and "is_impossible"
"answers" is a list, each element is a dictionary with keys "text" and "answer_start"
"text" contains an accepted answer to the question, "answer_start" contains the position in the "context" that the answer is from
"""

def get_named_entities():
    squad2 = json.load(open("data/squad2/train-v2.0.json", "r"))
    data = squad2["data"]
    num_data = len(data)
    entity_dict = {}

    # perform NER on every context to get a list of all named entities
    # organize them in entity_dict by label
    for i in range(num_data):
        num_paragraphs = len(data[i]["paragraphs"])
        for j in range(num_paragraphs):
            entities = NER(data[i]["paragraphs"][j]["context"])
            for entity in entities.ents:
                label = entity.label
                text = entity.text
                if label not in entity_dict:
                    entity_dict[label] = set()
                entity_dict[label].add(text)

    for k in entity_dict.keys():
        entity_dict[k] = list(entity_dict[k])

    outfile = open("data/squad2/NER.json", "w")
    json.dump(entity_dict, outfile)

"""
modify a paragraph's context to make every answerable question unanswerable

for each answerable question, pick one named entity from it that is also in the context
and replace all instances in the context with a random one from the same category

the replacement named entity cannot be in any of the other questions to avoid
randomly messing up 
"""
def modify_context(data, index, paragraph, all_entities, NER):
    entities_to_modify = set()
    entities_in_questions = set()
    qas = data[index]["paragraphs"][paragraph]["qas"]
    context = data[index]["paragraphs"][paragraph]["context"]
    new_qas = []

    for qa in qas:
        entities = NER(qa["question"])
        entities_in_this_question = set()
        successful_modification = False
        for entity in entities.ents:
            entities_in_questions.add(entity.text)
            entities_in_this_question.add(entity)
        if not qa["is_impossible"]:
            shared_entities = set()
            for entity in entities_in_this_question:
                if entity.text in context:
                    shared_entities.add(entity)
            if len(shared_entities) > 0:
                entities_to_modify.add(random.choice(list(shared_entities)))
                successful_modification = True
                new_qas.append({
                    "question": qa["question"],
                    "id": qa["id"],
                    "answer": [],
                    "is_impossible": True,
                    "modified": "context NER"
                })

        if not successful_modification:
            new_qas.append(qa)
            new_qas[-1]["modified"] = "N/A"

    entities_to_modify = list(entities_to_modify)
    new_context = context
    for entity in entities_to_modify:
        label = str(entity.label)
        possible_replacements = set(all_entities[label])
        possible_replacements_no_overlap = list(possible_replacements - entities_in_questions)
        replacement = random.choice(possible_replacements_no_overlap)
        new_context = new_context.replace(entity.text, replacement)

    data[index]["paragraphs"][paragraph]["qas"] = new_qas
    data[index]["paragraphs"][paragraph]["context"] = new_context

    return new_qas, new_context

NER = spacy.load("en_core_web_sm")
squad2 = json.load(open("data/squad2/train-v2.0.json", "r"))
data = squad2["data"]
all_entities = json.load(open("data/squad2/NER.json", "r"))

num_data = len(data)
for i in range(num_data):
    num_paragraphs = len(data[i]["paragraphs"])
    for j in range(num_paragraphs):
        modified_qas, modified_context = modify_context(data, i, j, all_entities, NER)
        data[i]["paragraphs"][j]["qas"] = modified_qas
        data[i]["paragraphs"][j]["context"] = modified_context
modified_context_NER = open("data/squad2/modified_context_NER.json", "w")
json.dump(data, modified_context_NER)