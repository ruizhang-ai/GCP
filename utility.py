from dateutil.parser import parse
import re
import string
import requests
import json
import editdistance
import sys
import datetime
from nltk.corpus import wordnet as wn
from nltk import ngrams
from itertools import groupby
from nltk.tree import Tree
import urllib
from datefinder import DateFinder
import datefinder
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
from kitchen.text.converters import getwriter, to_bytes, to_unicode
from kitchen.i18n import get_translation_object
from random import choice
import networkx as nx
import itertools
translations = get_translation_object('example')
_ = translations.ugettext
b_ = translations.lgettext

import gensim
from gensim.models import word2vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from gensim.matutils import softcossim
embeddings = gensim.models.KeyedVectors.load_word2vec_format("data/combined.vector", binary=False)
normalizer = gensim.models.KeyedVectors.load_word2vec_format("data/combined.vector.norm", binary=False)

import nltk
import numpy as np

regex = re.compile(".*?\((.*?)\)")
def removeBracket(string):
    result = re.findall(regex, string)
    for r in result:
        remove = '('+r+')'
        string = string.replace(remove, '')
    return string.replace('  ', ' ').strip()
    
def containBracket(string):
    if string == '<unk>':
        return string
    result = re.findall(regex, string)
    returned = None
    for r in result:
        returned = r
        break
    if returned == None:
        return returned
    else:
        return ' '.join(camel_case_split(returned)).lower()

url_regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
def contains_URL(string):
    found_url = False
    string = string.split()
    for s in string:
        result = re.findall(url_regex, s)
        for r in result:
            return True
    return False

def normalizeDate(input_string):
    finder = DateFinder()
    x = finder.extract_date_strings(input_string)
    matches = list(datefinder.find_dates(input_string))
    if len(matches) > 0:
        for m in matches:
            txt_str = x.next()
            find = txt_str[0]
            replacement = str(m).split(' ')[0]
            today = str(datetime.date.today())
            if replacement.strip() == today.strip():
                continue
            if(len(find) > 4):
                input_string = input_string.replace(find, replacement)
    else:
        pass
    return input_string

def getSPO(triple):
    s = triple.s.replace("_", " ").replace('"', '').strip()
    p = triple.p.replace("_", " ").replace('"', '').strip()
    o = triple.o.replace("_", " ").replace('"', '').strip()
    return s,p,o

def replaceDuplicateSequence(string):
    data = string.split(' ')
    output = list()
    prev = ''
    for d in data:
        if d!=prev:
            prev = d
            output.append(d)
    return ' '.join(output)

def printData(data):
    triples, texts = data
    for t in triples:
        print t
    print texts

def getLabelFullTries(entity, position, dictionary=None, advance=True):
    label = getLabel(entity, position, dictionary, advance)
    if label == 'UNKOWN':
        label = position
    if label == position and ',' in entity:
        entities = entity.split(',')
        for e in entities:
            tmp = getLabel(e, position, dictionary, advance)
            label = tmp
            if label != position:
                break
    if label == position and ' and ' in entity:
        entities = entity.split(' and ')
        for e in entities:
            tmp = getLabel(e, position, dictionary, advance)
            label = tmp
            if label != position:
                break
    if label == position:
        label = getLabel(entity, position, dictionary, advance)
    return label
    
def getLabel(entity, position, dictionary=None, advance=True):
    label = position
    if dictionary and entity in dictionary:
        return dictionary[entity]
    #CammelCase on bracket
    insideBracket = getBracketContent(entity)
    if insideBracket != entity[:-1] and camel(insideBracket):
        label = insideBracket
    else:
        entity = entity.split('(')[0]
        #Number only
        if entity.isdigit() or isFloat(entity):
            label = "NUMBER"

        #DATE format
        elif isDate(entity):
            label = "DATE"
        
        #FROM DICTIONARY or DBPEDIA
        else:
            #FROM DICT
            if dictionary:
                try:
                    label = dictionary[entity].strip()
                except:
                    pass
            #FROM ONLINE
            else:
                #print entity
                #print "FROM DBPEDIA"
                fromDBpedia = getDBpedia(entity, advance)
                if fromDBpedia:
                    label = fromDBpedia
                else:
                    #print "SEARCH", entity, "FROM GOOGLE"
                    fromGoogle = getGoogle(entity, advance)
                    if fromGoogle:
                        label = fromGoogle
    return label

def camel(s):
    return (s[0] != s[0].upper() and s != s.lower() and s != s.upper())

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def getBracketContent(s):
    return s[s.find("(")+1:s.find(")")]

def removeNonEntityBracket(s):
    s = s.replace("-LRB-", "(").replace("-RRB-", ")")
    content = s[s.find("(")+1:s.find(")")]
    if "ENTITIES_" not in content and "PREDICATE_" not in content:
        return s.replace("(" +content+")", "")
    else:
        return s

def isDate(string):
    try: 
        parse(string)
        return True
    except ValueError:
        return False
    
def isFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]

def getNameFromURI(uri):
    return uri.split('/')[-1].split('#')[-1].strip()

def getTypeDeep(classes):
    gen_type = 'ENTITY'
    gen_depth = sys.maxint
    all_type = set()
    for t in classes:
        if t['label'] == 'agent':
            continue
        synsets = wn.synsets(t['label'].replace(' ', '_'))
        if len(synsets) == 0 and ' ' in t['label']:
            synsets = wn.synsets(find_head_of_np(english_parser.raw_parse(t['label'].lower()).next()))
        for synset in synsets:
            if synset.name().split('.')[1] != 'n':
                continue

            if synset.min_depth() < gen_depth:
                gen_depth = synset.min_depth()
                gen_type = t['label']
            all_type.add(t['label'])
    max_score = 0
    choosen_type = ''
    for t in all_type:
        t_original = t
        if ' ' in t:
            t = find_head_of_np(english_parser.raw_parse(t.lower()).next())
        t = t.lower()
        if t != gen_type.lower():
            if gen_type.lower() in w2v_model.vocab and t in w2v_model.vocab:
                score = w2v_model.similarity(t, gen_type.lower())
                if score > max_score:
                    max_score = score
                    choosen_type = t_original
                    
    if gen_type!='ENTITY' and choosen_type != '':
        return gen_type.upper() + "[%%%]" + choosen_type.upper()
    elif gen_type != 'ENTITY':
        return gen_type.upper()
    else:
        return False

def getDBpedia(entity, advance=True):
    qstring = entity
    url = 'http://lookup.dbpedia.org/api/search.asmx/KeywordSearch?QueryString='+urllib.quote_plus(qstring.encode('utf8'))
    headers = {'accept': 'application/json'}

    r = requests.get(url, headers=headers)
    content = json.loads(r.text)['results']
    
    found = False
    max_distance = -1
    if len(content) > 0:
        choosen_content = None
        for c in content:
            if len(c['classes'])==0 and len(c['categories'])==0:
                continue
            tmp = SequenceMatcher(None, c['label'], qstring).ratio()
            if tmp > max_distance:
                max_distance = tmp
                choosen_content = c
        
        if choosen_content == None:
            return False
        
        label_class = 'ENTITY'
        if advance:
            if len(choosen_content['classes']) > 0:
                label_class = getTypeDeep(choosen_content['classes'])
                found = True
            else:
                label_class = getTypeDeep(choosen_content['categories'])
                found = True
        else:
            if len(choosen_content['classes']) > 0:
                    for kelas in choosen_content['classes']:
                        if (not kelas['label'].upper().startswith("OWL#")) and (not contains_URL(kelas['label'])):
                            label_class = kelas['label'].upper()
                            found = True
                            break
            else:
                if len(choosen_content['categories']) > 0:
                    for category in choosen_content['categories']:
                        if not category['label'].upper().startswith("OWL#"): 
                            label_class = category['label'].upper()
                            found = True
                            break
    if found:
        return label_class
    else:
        return False
    
def getEntityName(entity):
    url = "https://www.google.com.au/search?q="+urllib.quote_plus(entity.encode('utf8'))+" Wikipedia&start=0&num=1"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    response = requests.get(url, headers=headers)
    content = response.content
    soup = BeautifulSoup(content, "lxml")
    urlList = list()
    for result in soup.findAll('h3', class_="r"):
        for url in result.findAll('a',href=True):
            urlList.append(url['href'].split('/')[-1].replace('_', ' '))
    return urlList[0].strip()

def getGoogle(entity, advance=True):
    try:
        qstring = entity
        url = 'https://kgsearch.googleapis.com/v1/entities:search?query='+urllib.quote_plus(qstring.encode('utf8'))+'&key=AIzaSyBKuNx-gfX106kWNAO1QFibFhEe6I69AJg&limit=10&indent=True'
        print url
        r = requests.get(url)
        content = json.loads(r.text)['itemListElement']
        
        #try normalize name by using google search#
        if len(content)==0:
            qstring = getEntityName(entity)
            url = 'https://kgsearch.googleapis.com/v1/entities:search?query='+urllib.quote_plus(qstring.encode('utf8'))+'&key=AIzaSyCSWIDmgxih5nX8ijMsT_QUdwtwnW0NyOo&limit=10&indent=True'
            r = requests.get(url)
            content = json.loads(r.text)['itemListElement']
            
        found = False
        max_distance = 0
        if len(content) > 0:
            choosen_content = None
            for c in content:
                if 'name' not in c['result']:
                    continue
                tmp = SequenceMatcher(None, c['result']['name'], qstring).ratio()
                if tmp > max_distance:
                    max_distance = tmp
                    choosen_content = c
                ## Re-search again if match schore is low ##
                if max_distance < 0.5:
                    if len(content)==0:
                        qstring = getEntityName(entity)
                        url = 'https://kgsearch.googleapis.com/v1/entities:search?query='+urllib.quote_plus(qstring.encode('utf8'))+'&key=AIzaSyCSWIDmgxih5nX8ijMsT_QUdwtwnW0NyOo&limit=10&indent=True'
                        r = requests.get(url)
                        content = json.loads(r.text)['itemListElement']
                    max_distance = 0
                    if len(content) > 0:
                        choosen_content = None
                        for c in content:
                            if 'name' not in c['result']:
                                continue
                            tmp = SequenceMatcher(None, c['result']['name'], qstring).ratio()
                            if tmp > max_distance:
                                max_distance = tmp
                                choosen_content = c
                #END try normalize name by using google search#

            if choosen_content == None:
                return False
            
            label_class = 'ENTITY'
            classes = choosen_content['result']['@type']
            
            if advance:
                norm_classes = list()
                for c in classes:
                    if c == 'Thing':
                        continue
                    data = dict()
                    data['label'] = ' '.join(camel_case_split(c)).lower()
                    norm_classes.append(data)
                
                if len(classes) > 0:
                    label_class = getTypeDeep(norm_classes)
                    found = True
                else:
                    if 'description' in c['result']:
                        label_class = ' '.join(camel_case_split(choosen_content['result']['description'])).upper()
                        found = True
            else:
                if len(classes) > 0:
                        for kelas in classes:
                            label_class = ' '.join(camel_case_split(kelas)).upper()
                            found = True
                            break
                else:
                    if 'description' in c['result']:
                        label_class = ' '.join(camel_case_split(choosen_content['result']['description'])).upper()
                        found = True
            if (label_class == 'THING' or label_class == False) and 'description' in choosen_content['result']:
                label_class = ' '.join(camel_case_split(choosen_content['result']['description'])).upper()
                found = True
        if found:
            return label_class
        else:
            return False
    except:
        raise
        return False

def find_head_of_np(np):
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    top_level_trees = [np[i] for i in range(len(np)) if type(np[i]) is Tree]
    ## search for a top-level noun
    top_level_nouns = [t for t in top_level_trees if t.label() in noun_tags]
    if len(top_level_nouns) > 0:
        ## if you find some, pick the rightmost one, just 'cause
        return top_level_nouns[-1][0]
    else:
        ## search for a top-level np
        top_level_nps = [t for t in top_level_trees if t.label()=='NP']
        if len(top_level_nps) > 0:
            ## if you find some, pick the head of the rightmost one, just 'cause
            return find_head_of_np(top_level_nps[-1])
        else:
            ## search for any noun
            nouns = [p[0] for p in np.pos() if p[1] in noun_tags]
            if len(nouns) > 0:
                ## if you find some, pick the rightmost one, just 'cause
                return nouns[-1]
            else:
                ## return the rightmost word, just 'cause
                return np.leaves()[-1]

def getTypeSubject(subject, predicate):
    subject = subject.lower()
    predicate = predicate.lower()
    
    if ' ' in predicate or predicate not in w2v_model.vocab:
        return subject
    if ' ' in subject or subject not in w2v_model.vocab:
        return subject
    
    synsets = wn.synsets(subject.replace(' ', '_'))
    if len(synsets) == 0 and ' ' in subject:
        synsets = wn.synsets(find_head_of_np(english_parser.raw_parse(subject.lower()).next()))
        
    possible_words = set()
    possible_words.add(subject)
    for synset in synsets:
        if synset.name().split('.')[1] != 'n':
            continue
        for hyper in synset.hypernyms():
            possible_words.add(hyper.name().split('.')[0])
    
    max_score = 0
    choosen_type = ''
    for word in possible_words:
        if word in w2v_model.vocab:
            score = w2v_model.similarity(word, predicate)
            if score > max_score:
                max_score = score
                choosen_type = word
    return choosen_type

def replaceSequence(sequence, replacement, lst, expand=False):
    out = list(lst)
    cont = True
    while cont:
        tmp_list = out
        out = list(tmp_list)
        for i, e in enumerate(tmp_list):
            if e.lower() == sequence[0].lower():
                i1 = i
                f = 1
                for e1, e2 in zip(sequence, tmp_list[i:]):
                    if e1.lower() != e2.lower():
                        f = 0
                        break
                    i1 += 1
                if f == 1:
                    del out[i:i1]
                    if expand:
                        for x in list(replacement):
                            out.insert(i, x)
                    else:
                        out.insert(i, replacement)
                    break
            if i >= len(out)-1:
                cont = False
    return out

def replaceAbbreviation(text, triples, abbr):
    regex_abbr = r"\b[A-Z][a-zA-Z\.]*[A-Z]\b\.?"
    regex_punct = re.compile('[%s]' % re.escape(string.punctuation))
    
    #REPLACE TEXT#
    all_abbrev = re.findall(regex_abbr, text)
    all_abbrev2 = re.findall(regex_abbr, regex_punct.sub('',text))
    all_abbrev += all_abbrev2
    
    for abbrev in all_abbrev:
        abbrev = _(abbrev)
        if abbrev in abbr:
            for triple in triples:
                for t in triple:
                    longest_abbr = abbrev
                    for a in abbr[abbrev]:
                        a = _(a)
                        if a in _(t[0]) and (len(a) > len(longest_abbr) and ' ' in a):
                            longest_abbr = a
                    text = text.replace(abbrev, longest_abbr)
    #END REPLACE TEXT#
    
    #REPLACE TRIPLE#
    for triple in triples:
        for t in triple:
            all_abbrev = re.findall(regex_abbr, t[0])
            for abbrev in all_abbrev[:1]:#Take only the first occurence
                abbrev = _(abbrev)
                if abbrev in abbr:
                    longest_abbr = abbrev
                    for a in abbr[abbrev]:
                        a = _(a)
                        if a in _(text) and (len(a) > len(longest_abbr) and ' ' in a):
                            longest_abbr = a
                    t[0] = t[0].replace(abbrev, longest_abbr)
    #END REPLACE TRIPLE#
    return text, triples

def getSentenceFromAllTriple(triples, triple_max_len, includeLabel = True):
    sentence = ''
    for triple in triples:
        triple_part = ''
        for part in triple:
            key, value, labels = part
            label = ' '.join(labels)
            if(label=='UNKNOWN'):
                label2 = containBracket(key)
                if label2 != None:
                    label = label2
            if includeLabel:
                triple_part += _(value) + " " + _(label.lower()) + " "
            else:
                triple_part += _(key) + " "
        triple_part = triple_part.split(" ")
        while len(triple_part) < triple_max_len:
            triple_part.append("pad")
        triple_part = triple_part[:triple_max_len]
        sentence += ' '.join(triple_part)
        sentence += ' '
    return sentence


def getSentenceFromGraph(triples, triple_max_len, includeLabel = True):
    G = nx.DiGraph()
    first = True
    start_node = 0
    sentence = ''
    for triple in triples:
        s, p, o = triple
        key_s, value_s, label_s = s
        key_p, value_p, label_p = p
        key_o, value_o, label_o = o
        label_s = ' '.join(label_s[0:2])
        #label_s = ' '.join(label_s[1:]+ [label_s[0]])
        #label_s = ' '.join(label_s)
        
        '''label2=None
        if(label_s=='UNKNOWN'):
            label2 = containBracket(key_s)
            if label2 != None:
                label_s = label2.lower()'''
        if label_p != '<unk>':
            label_p = ' '.join(label_p[0:1])
        label2=None
        if(label_p.lower()=='unknown'):
            label2 = containBracket(key_p)
            print label2
            if label2 != None:
                label_p = label2.lower()
        label_o = ' '.join(label_o[0:2])
        #label_o = ' '.join(label_o[1:]+ [label_o[0]])
        #label_o = ' '.join(label_o)
        '''label2=None
        if(label_o=='UNKNOWN'):
            label2 = containBracket(key_o)
            if label2 != None:
                label_o = label2.lower()'''
        if includeLabel:
            sub = _(_(value_s) + ' ' + _(label_s))
            pre = _(_(label_p) + ' ' + _(value_p))
            obj = _(_(value_o) + ' ' + _(label_o))
        else:
            sub = _(key_s)
            pre = _(label_p +' '+value_p)
            obj = _(key_o)
        if first:
            start_node = sub
            first = False
        #print sub
        #print obj
        #print
        G.add_edge(sub, obj, pred=pre)
    
    max_now = -1
    for n in nx.nodes(G):
        count = 0
        for node in nx.dfs_preorder_nodes(G, n):
            count += 1
        if count > max_now:
            start_node = n
            max_now = count
    
    #start_node = choice(G.nodes()) ### RANDOM NODE CHOICE
    
    path_length = 0
    prev_ent = ''
    pads = []
    traversal = getTopoSort(G.copy(), start_node)

    for x in traversal:
        #print "RES:",x
        if prev_ent != _(x[0]):
            subject_string = _(x[0]).split(" ")
            subject_string = subject_string[:triple_max_len]
            while len(subject_string) < triple_max_len:
                subject_string.append('pad')
            sentence += ' '.join(subject_string)
            sentence += ' '
            sentence += ' '. join(pads)
            sentence += ' '
        prev_ent = _(x[1])
        
        object_string = _(x[1]).split(" ")
        object_string = object_string[:triple_max_len]
        while len(object_string) < triple_max_len:
            object_string.append('pad')
        sentence += ' '.join(object_string)
        sentence += ' '
        
        predicate_string = _(G[x[0]][x[1]]['pred']).split(' ')
        predicate_string = predicate_string[:triple_max_len]
        while len(predicate_string) < triple_max_len:
            predicate_string.append('pad')
        sentence += ' '.join(predicate_string)
        sentence += ' '
        path_length+=1
    return sentence


def getMostLeftNode(node_list):
    if len(node_list) == 1:
        return node_list[0]
        
    node_dict = dict()
    node_count = dict()
    for e in node_list:
        node_dict[e] = e
        node_count[e] = 0
        
    node_group = dict() #(xy -> [(x,y), (y,x)])
    for e in list(itertools.permutations(node_dict.keys(),2)):
        e_sorted = sorted(e) #(xy)
        key = ' '.join(e_sorted)
        if key not in node_group:
            node_group[key] = [None,None]
        if e[0] == e_sorted[0]:
            node_group[key][0] = e
        else:
            node_group[key][1] = e
    
    for key in node_group:
        left_sim = 0 #entity0 in the left of entity1
        right_sim = 0 ##entity1 in the left of entity0
        
        left_vector = embeddings[node_group[key][0][0].split(" ")[-1]] - embeddings[node_group[key][0][1].split(" ")[-1]]
        right_vector = embeddings[node_group[key][1][0].split(" ")[-1]] - embeddings[node_group[key][1][1].split(" ")[-1]]
        
        norm = normalizer["http://predicate.com/is_left_entity_of"].reshape(1, -1)
        norm = preprocessing.normalize(norm, norm='l2')
        
        left_vector = left_vector - np.sum(np.multiply(left_vector, norm)) * norm
        right_vector = right_vector - np.sum(np.multiply(right_vector, norm)) * norm
        
        left_sim = cosine_similarity(left_vector.reshape(1,-1), embeddings["http://predicate.com/is_left_entity_of"].reshape(1,-1))
        right_sim = cosine_similarity(right_vector.reshape(1,-1), embeddings["http://predicate.com/is_left_entity_of"].reshape(1,-1))
        
        if left_sim > right_sim:
            node_count[node_group[key][0][0]] += 1
        else:
            node_count[node_group[key][1][0]] += 1
    
    max_count = 0
    choosen_ent = None
    for e in node_count:
        if node_count[e] > max_count:
            max_count = node_count[e]
            choosen_ent = e
    return node_dict[choosen_ent]

def getTopoSort(G, start):    
    result = list()
    
    removed_node = start
    for neighbor in sorted(G.neighbors(removed_node)):
        res = (removed_node, neighbor, G[removed_node][neighbor]['pred'])
        result.append(res)
    G.remove_node(removed_node)
    
    all_nodes = G.nodes()
    while len(all_nodes) > 0:
        has_0_in_degree = list()
        removed_node = None
        for node in all_nodes:
            if G.in_degree(node) == 0:
                has_0_in_degree.append(node)
        if len(has_0_in_degree) == 0:
            #removed_node = choice(all_nodes) ### RANDOM NODE CHOICE
            
            ### CHOOSE BY EMBEDDINGS
            removed_node = getMostLeftNode(all_nodes)
            ###
            all_nodes.remove(removed_node)
        elif len(has_0_in_degree) == 1:
            removed_node = has_0_in_degree[0]
            all_nodes.remove(removed_node)
        else:
            for n in has_0_in_degree:
                if n == start:
                    removed_node = n
                    all_nodes.remove(removed_node)
                    break
                else:
                    #removed_node = choice(has_0_in_degree) ### RANDOM NODE CHOICE
                    
                    ### CHOOSE BY EMBEDDINGS
                    removed_node = getMostLeftNode(has_0_in_degree)
                    ###
                    
                    all_nodes.remove(removed_node)
                    break
        if removed_node != None:
            
            unsorted_neighbor = list()
            sorted_neighbor = list()
            for neighbor in G.neighbors(removed_node):
                unsorted_neighbor.append(neighbor)
            while len(unsorted_neighbor) > 0:
                removed_neighbor = getMostLeftNode(unsorted_neighbor)
                sorted_neighbor.append(removed_neighbor)
                unsorted_neighbor.remove(removed_neighbor)
            
            for neighbor in sorted_neighbor:
                res = (removed_node, neighbor, G[removed_node][neighbor]['pred'])
                result.append(res)
            G.remove_node(removed_node)
                
    return result
    
def createDataset(dataset, filename, triple_mode, includeLabel, triple_max_len, mode_vocab):
    src_file = open('data/'+filename+'.src', 'w')
    tar_file = open('data/'+filename+'.tar', 'w')
    vocab_file = open('data/vocab.all', mode_vocab)

    #i = 0
    for data in dataset:
        #i+=1
        #print i
        #if i < 19:
        #    continue
            
        triples, text = data
        
        #for t in triples:
        #    print t
        #print
        
        if triple_mode == 'single':
            sentence = getSentenceFromAllTriple(triples, triple_max_len, includeLabel=includeLabel) 
        elif triple_mode == 'graph':
            sentence = getSentenceFromGraph(triples, triple_max_len, includeLabel=includeLabel)
        src_file.write(b_(sentence)+"\n")
        tar_file.write(b_(text)+'\n')
        vocab_file.write(b_(sentence)+"\n")
        vocab_file.write(b_(text)+'\n')
        
        #break
        
    src_file.close()
    tar_file.close()
    vocab_file.close()