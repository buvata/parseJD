import pandas as pd
import demoji
import re
import string
import json
import unicodedata2
import pyvi
from pyvi import ViTokenizer
import pickle


set_punctuations = set(string.punctuation)
list_punctuations_out = ['”', '”', "›", "“", '"']
for e_punc in list_punctuations_out:
    set_punctuations.add(e_punc)

set_punctuations.remove(".")
set_punctuations.remove("@")
print(set_punctuations)



def remove_multi_space(text):
    text = text.replace("\n", " . ")
    text = re.sub("\s\s+", " ", text)
    # handle exception when line just all of punctuation
    if len(text) == 0:
        return text
    if text[0] == " ":
        text = text[1:]
    if len(text) == 0:
        pass
    else:
        if text[-1] == " ":
            text = text[:-1]

    return "".join(text)


def handle_punctuation(text):
    l_new_char = []
    for e_char in text:
        if e_char not in list(set_punctuations):
            l_new_char.append(e_char)
        else:
            l_new_char.append(" {} ".format(e_char))

    text = "".join(l_new_char)

    return text


def handle_unicode(text):
    text = re.sub(r"òa", "oà", text)
    text = re.sub(r"óa", "oá", text)
    text = re.sub(r"ỏa", "oả", text)
    text = re.sub(r"õa", "oã", text)
    text = re.sub(r"ọa", "oạ", text)
    text = re.sub(r"òe", "oè", text)
    text = re.sub(r"óe", "oé", text)
    text = re.sub(r"ỏe", "oẻ", text)
    text = re.sub(r"õe", "oẽ", text)
    text = re.sub(r"ọe", "oẹ", text)
    text = re.sub(r"ùy", "uỳ", text)
    text = re.sub(r"úy", "uý", text)
    text = re.sub(r"ủy", "uỷ", text)
    text = re.sub(r"ũy", "uỹ", text)
    text = re.sub(r"ụy", "uỵ", text)
    text = re.sub(r"Ủy", "Uỷ", text)
    return text


def normal_text(txt):
    txt = demoji.replace(txt, "")
    txt = unicodedata2.normalize('NFKC', txt)
    txt = handle_unicode(txt)
    txt = handle_punctuation(txt)
    txt = remove_multi_space(txt)
    txt = txt.strip()
    return txt

