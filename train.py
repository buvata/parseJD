import spacy
import random
from spacy.util import minibatch, compounding
from handle_data import normal_text
import pickle

nlp = spacy.load('vi_spacy_model-0.2.1/vi_spacy_model/vi_spacy_model-0.2.1/')

VALUE_ENTITIES = ["NAME", "EMAIL", "LOC", "TIME", "PHONE", "1-COM", "2-DESIG", "3-NUM", "4-SAL"]

def train_model(train_data):
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True) 
    else:
        ner = nlp.get_pipe("ner")
        
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimize = nlp.begin_training()
        
        n_iter = 200
        for itn in range(n_iter):
            print("starting:", itn)
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)
            

    nlp.to_disk("nlp_jd_model")
    


def convert_entities_list_to_dict(entities):
    info = {label: [] for label in VALUE_ENTITIES}
    for label, value in entities:
        if label not in info:
            info[label] = []
        info[label].append(value)
    
    return info


class JDParser():
    def __init__(self, model_dir):
        nlp_model = spacy.load(model_dir)
        self.nlp_model = nlp_model

    def preprocess_text(self, text):
        text = normal_text(text)
        return text.strip()

    def parse(self, text):
        text = self.preprocess_text(text)
        text_lower = text.lower()

        doc = self.nlp_model(text_lower)
        info = []
        for ent in doc.ents:
            info.append((ent.label_, ent.text))
            # infos.append((ent.label_, ent.text, ent.start_char, ent.end_char))
        
        info = convert_entities_list_to_dict(info)

        return info

    
def test_model(model_dir, file_dir):
    testfile = open(file_dir, 'rb')     
    data_test = pickle.load(testfile)

    list_text_test = []
    list_label_test = []
    for txt, label in data_test:
        list_text_test.append(txt)
        list_label_test.append(label)

    list_label_predict = []
    JD = JDParser(model_dir)
    for text in list_text_test:
        info = JD.parse(text)
        list_label_predict.append(info)
        
    dict_predict = {label: [] for label in VALUE_ENTITIES}
    for i in list_label_predict:
        for tag in VALUE_ENTITIES:
            dict_predict[tag].append(i[tag])
            
    dict_label = {label: [] for label in VALUE_ENTITIES}
    for i in list_label_test:
        for tag in VALUE_ENTITIES:
            dict_label[tag].append(i[tag])
    
    dict_score = {label: [] for label in VALUE_ENTITIES}
    for tag in dict_predict:
        cnt = 0
        for i, val in enumerate(dict_predict[tag]):  
            if val == dict_label[tag][i]:
                cnt+=1

        scr = "{0:.2f}".format(cnt/len(dict_predict[tag]))

        dict_score[tag].append(scr)

    return list_label_predict, dict_score
