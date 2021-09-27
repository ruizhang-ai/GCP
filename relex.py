from dateutil import parser
import spacy,en_core_web_lg
nlp = en_core_web_lg.load()
from spacy import displacy
nlp.remove_pipe("parser")
nlp.remove_pipe("ner")
nlp.remove_pipe("tagger")

import datefinder

from kitchen.text.converters import getwriter, to_bytes, to_unicode
from kitchen.i18n import get_translation_object
translations = get_translation_object('example')
_ = translations.ugettext
b_ = translations.lgettext

import cPickle

import re

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

regex = re.compile(".*?\((.*?)\)")
def removeBracket(string):
    result = re.findall(regex, string)
    for r in result:
        remove = '('+r+')'
        string = string.replace(remove, '')
    return string.replace('  ', ' ').strip()

def containBracket(string):
    result = re.findall(regex, string)
    returned = None
    for r in result:
        returned = r
        break
    if returned == None:
        return returned
    else:
        return ' '.join(camel_case_split(returned)).lower()

def getSentenceFromAllTriple(triples):
    sentence = ''
    for triple in triples:
        for part in triple:
            key, value, labels = part
            label = ' '.join(labels)
            if(label=='UNKNOWN'):
                label2 = containBracket(key)
                if label2 != None:
                    label = label2
            if label=='UNKNOWN':
                print key
            sentence += _(value) + " " + _(label.lower()) + " "
        sentence += "TTRRPP "
    sentence = sentence[:-7].strip()
    return sentence

# GROUP PER SUBJECT
def getSentenceFromAllTripleGrouped(triples):
    sentence = ''
    #Get all subject
    all_subject = list()
    for triple in triples:
        s,p,o = triple
        key, value, label = s
        if key not in all_subject:
            all_subject.append(key)

    for subject in all_subject:
        for triple in triples:
            s,p,o = triple
            key_s, value_s, labels_s = s
            key_p, value_p, labels_p = p
            key_o, value_o, labels_o = o
            if key_s == subject:
                label2 = None
                #SUBJECT
                label = ' '.join(labels_s)
                if(label=='UNKNOWN'):
                    label2 = containBracket(key_s)
                    if label2 != None:
                        label = label2
                sentence += _(value_s) + " " + _(label.lower()) + " "
                #PREDICATE
                label = ' '.join(labels_p)
                if(label=='UNKNOWN'):
                    label2 = containBracket(key_p)
                    if label2 != None:
                        label = label2
                sentence += _(value_p) + " " + _(label.lower()) + " "
                #OBJECT
                label = ' '.join(labels_o)
                if(label=='UNKNOWN'):
                    label2 = containBracket(key_o)
                    if label2 != None:
                        label = label2
                sentence += _(value_o) + " " + _(label.lower()) + " "
                
        ### DELIMITER ###
            #sentence += "TTRRPP " ### PER TRIPLE
        sentence += "TTRRPP " ### PER SUBJECT
        #################
            
    sentence = sentence[:-7].strip() #Remove last separator
    return sentence

def getSentenceFromUniqueTriple(triples):
    
    #Populate subject & object
    unique_triple = list()
    inputted_entity = list()
    for triple in triples:
        s,p,o = triple
        replace = (s,p,o)
        for data in replace:
            key, value, label = data
            key = _(key)
            k = removeBracket(_(value))
            v = ' '.join(label)
            v = _(v)
            if(v=='UNKNOWN'):
                label2 = containBracket(_(key))
                if label2 != None:
                    v = label2
            t = value + " " + v.lower()
            if t not in unique_triple and key not in inputted_entity:
                unique_triple.append(_(t))
                inputted_entity.append(key)
                
    sentence = ' '.join(unique_triple)
    return sentence.strip()


from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer
def detokenizer2(the_str):
    matches = datefinder.find_dates(the_str, source=True, strict=False, index=True)
    for m in matches:
        try:
            d = parser.parse(str(m[0]))
            d = d.strftime("%Y-%m-%d")
            #d = d.strftime("%d %B, %Y")
            the_str = the_str.replace(str(m[1]), d).lower()
            #print the_str
        except:
            continue
    doc = nlp(_(the_str))
    the_str = ' '.join([token.text.strip() for token in doc])
    m_detokenizer = MosesDetokenizer()
    the_str = the_str.replace('- LRB-', '(')
    the_str = the_str.replace('- RRB-', ')')
    the_str = the_str.replace('-LRB-', '(')
    the_str = the_str.replace('-RRB-', ')')
    the_str = the_str.replace('`', "'")
    the_str = the_str.replace("''", "'")
    list_sent = the_str.split('.')
    for i in range(len(list_sent)-1,1,-1):
        if list_sent[i] == list_sent[i-1]:
            del list_sent[i]
    the_str = '.'.join(list_sent)
    return the_str.lower()
    '''tokens = the_str.split()
    result = m_detokenizer.detokenize(tokens, return_str=True)
    tokens = result.split()
    result = "".join([" "+i if (not i.startswith("'s") and not i.startswith("' ") and not i.startswith("'.")) else i for i in tokens]).strip()
    result = result.replace('( ', '(')
    return result.lower()'''

def camel(s):
    return (s[0] != s[0].upper() and s != s.lower() and s != s.upper())

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return ' '.join([m.group(0) for m in matches])
    
import sys

if 'test_unseen' in sys.argv[1]:
    dataset = cPickle.load(open('data/webnlg/test_unseen_RELEX.pickle', 'rb'))
elif 'test_gkb' in sys.argv[1]:
    dataset = cPickle.load(open('data/gkb/test_data_RELEX.pickle', 'rb'))
else:
    dataset = cPickle.load(open('data/webnlg/test_data_RELEX.pickle', 'rb'))

fo = open(sys.argv[1], 'r')
fw = open(sys.argv[1]+'.final', 'w')

for idx, line in enumerate(fo):
	triples, text = dataset[idx]
	#Get all entities
	entities = dict()
	#print text
	for t in triples:
	    #print t
	    for elem in t:
		value, name, entity_type = elem
		entities[name] = value
	result = line
	#Replace entites in text    
	for e in entities:
	    if b_(e) in result:
		result = result.replace(b_(e),removeBracket(camel_case_split(b_(entities[b_(e)]))))
	#print line.strip()
	#print result.strip()
	#print detokenizer2(result)
	#print
	fw.write(b_(detokenizer2(result))+"\n")
	#break

fo.close()
fw.close()