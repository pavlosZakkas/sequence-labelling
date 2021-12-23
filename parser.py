import nltk
from custom_dataclasses import Document, Sentence, WordInfo

def line_is_empty(line):
    return line.rstrip() == ""

def read_file(path):
    sents = []
    curr_sent = []
    with open(path) as my_file:
        for line in my_file:
            if(line_is_empty(line)):
                sents.append(curr_sent)
                curr_sent = []
                continue
            curr_sent.append(line)

    words, bios = [], []
    doc = Document(sentences=[])
    for sent in sents:
        sentence = Sentence(words=[])
        words, bios = [], []
        for record in sent:
            word, bio = record.rstrip().split()
            words.append(word)
            bios.append(bio)
        pos_tags = nltk.pos_tag(words)
        for index, val in enumerate(words):
            sentence.words.append(WordInfo(
                                    word=words[index], 
                                    pos=pos_tags[index][1], 
                                    bio=bios[index]
                                ))
        doc.sentences.append(sentence)
    return doc
