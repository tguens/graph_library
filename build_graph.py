#%%
from collections import OrderedDict
import os
import os.path as osp
import pickle
import random
import requests
import sys
from tqdm import tqdm

import numpy as np
import spacy

sys.path.insert(0, 'conceptnet5')
sys.path.insert(0, '/mnt/data_b/theo/graph_dl')
sys.path.insert(0, '/mnt/data_b/theo/graph_dl/graph_library')
sys.path.insert(0, '/DL2-Target1/theo')
sys.path.insert(0, '/DL2-Target1/theo/graph_library')
from conceptnet5.nodes import standardized_concept_uri
from conceptnet5.uri import uri_to_label
from graph_library.my_utils.tokenizer import reform_text
import my_utils as my_utils

#%%
# Part-Of-Speech tags corresponding to "concepts": 
# Nouns (NN, NNP, NNPS, NNS), 
# Verbs (MD, VB, VBD, VBG, VBN, VBP, VBZ), 
# Adjectives (JJ, JJR, JJS)
# See details at https://spacy.io/api/annotation#pos-tagging
POS_CONCEPTS = [8, 37, 38, 27, 16, 12, 20, 17, 21, 19, 4, 6, 36, 9]

# List of indices corresponding to the beginning of an entity. 
# Ex: United in United States of America.
ENTITY_BEGINNING = [5, 6, 8, 9, 12, 13, 18, 20, 25, 26, 
                    28, 29, 30, 31, 32, 33, 37, 38]
# List of indices corresponding to a part inside an entity
# Ex: States in United States of America.
ENTITY_INSIDE = [7, 10, 11, 14, 15, 16, 17, 19, 21, 22, 
                 23, 24, 27, 34, 35, 36, 39, 40]

# ConceptNet relations considered to build the graphs.
RELATIONS = ["None", "RelatedTo", "ExternalURL", "FormOf", "IsA", 
             "PartOf", "HasA", "UsedFor", "CapableOf", "AtLocation", 
             "Causes", "HasSubevent", "HasFirstSubevent", 
             "HasLastSubevent", "HasPrerequisite", "HasProperty", 
             "MotivatedByGoal", "ObstructedBy", "Desires", 
             "CreatedBy", "Synonym", "Antonym", "DistinctFrom", 
             "DerivedFrom", "SymbolOf", "DefinedAs", "Entails", 
             "MannerOf", "LocatedNear", "HasContext", "SimilarTo", 
             "EtymologicallyRelatedTo", "EtymologicallyDerivedFrom", 
             "CausesDesire", "MadeOf", "ReceivesAction", "InstanceOf", 
             "NotDesires", "NotUsedFor", "NotCapableOf", "NotHasProperty"]

# ConceptNet relations not considered to build the graphs. (they are specific)
DISCARDED_RELATIONS = ["ExternalURL", "genre", "influencedBy", 
                       "knownFor", "occupation", "language", 
                       "field", "product", "capital", "leader"]

def generate_graph(toks, postags, nertags, conceptnet_dict=None, verbose=False):
    SEED = 1234
    random.seed(SEED)
    
    primary_vocab_list = []
    total_vocab_list = []
    edges_list = []
    
    concepts = []
    toks_len = len(toks)
    
    i = 0
    while i < toks_len:
        # Adds entities (n_grams)
        if nertags[i] in ENTITY_BEGINNING:
            ngram_list = [toks[i].text]
            j = i+1
            while j < toks_len and nertags[j] in ENTITY_INSIDE:
                ngram_list.append(toks[j].text)
                j += 1
            ngram = ' '.join(ngram_list)
            concepts.append(ngram)
            i = j
            
        # Adds tokens that are considered as concepts under the Part Of Speech conditions
        elif postags[i] in POS_CONCEPTS and not toks[i].is_stop:
            concepts.append(toks[i].lemma_) # May change lemma by text...
            i += 1
        else:
            i += 1
            
    concepts_uri =[standardized_concept_uri('en', ngram) for ngram in concepts]
    unique_concepts_uri = list(OrderedDict.fromkeys(concepts_uri))
    
    if verbose:
        print("")
        print("Preprocessing generate_graph")
        print("concepts_uri:", concepts_uri)
        print("unique_concepts_uri:", unique_concepts_uri)
        print("")
        
    for uri_word in tqdm(unique_concepts_uri, total=len(unique_concepts_uri)):
        normalized_word = uri_to_label(uri_word)
        primary_vocab_list.append(normalized_word)

        if normalized_word not in total_vocab_list:
            total_vocab_list.append(normalized_word)

        if conceptnet_dict == None:
            try:
                obj = requests.get('{}{}'.format("http://api.conceptnet.io", 
                                                 uri_word)).json()
                edges = obj["edges"]
            except Exception:
                print("Request error")
                edges = []
                
        else:
            if normalized_word not in conceptnet_dict:
                try:
                    obj = requests.get('{}{}'.format("http://api.conceptnet.io", 
                                                     uri_word)).json()
                    edges = obj["edges"]
                    conceptnet_dict[normalized_word] = edges
                except Exception:
                    print("Request error")
                    edges = []
            else:
                edges = conceptnet_dict[normalized_word]

        at_least_one_edge = False
        for edge in edges:
            start_node = edge["start"]["term"]
            end_node = edge["end"]["term"]
            rel = edge["rel"]["label"]
            normalized_start_word = uri_to_label(start_node)  # uri_to_label("/c/en/movies") -> "movies"
            normalized_end_word = uri_to_label(end_node)
            
            if (rel in DISCARDED_RELATIONS or rel not in RELATIONS 
                or edge["start"]["language"] != "en" 
                or edge["end"]["language"] != "en"
                or (normalized_start_word in total_vocab_list 
                    and normalized_end_word in total_vocab_list)): 
                continue
            at_least_one_edge = True
                 
            if normalized_start_word not in total_vocab_list:
                total_vocab_list.append(normalized_start_word)

            if normalized_end_word not in total_vocab_list:
                total_vocab_list.append(normalized_end_word)
                        
            normalized_start_word_idx = total_vocab_list.index(normalized_start_word)
            normalized_end_word_idx = total_vocab_list.index(normalized_end_word)
            edges_list.append(((normalized_start_word_idx,normalized_end_word_idx), 
                               RELATIONS.index(rel)))
            
    total_vocab_list = [concept.replace(" ", "_") for concept in total_vocab_list]
    
    return (total_vocab_list, edges_list)

def postag_func(toks, vocab):
    return [vocab[w.tag_] for w in toks if len(w.text) > 0]

def nertag_func(toks, vocab):
    return [vocab['{}_{}'.format(w.ent_type_, w.ent_iob_)] 
            for w in toks if len(w.text) > 0]

def generate_graph_and_align(toks, postags, nertags, 
conceptnet_dict=None, verbose=False):

    def _get_concepts(uri_word, associated_nodes=[], edges_list=[], conceptnet_dict=None): 
        #What about duplicates ??-Don't care, just keep track.
        #associated_nodes = []
        normalized_word = uri_to_label(uri_word)
        normalized_word = uri_to_label(uri_word)
        associated_nodes.append(normalized_word)
        if conceptnet_dict == None:
            try:
                obj = requests.get('{}{}'.format("http://api.conceptnet.io", 
                                                 uri_word)).json()
                edges = obj["edges"]
            except Exception:
                print("Request error")
                edges = []
                
        else:
            if normalized_word not in conceptnet_dict:
                try:
                    obj = requests.get('{}{}'.format("http://api.conceptnet.io", 
                                                     uri_word)).json()
                    edges = obj["edges"]
                    conceptnet_dict[normalized_word] = edges
                except Exception:
                    print("Request error")
                    edges = []
            else:
                edges = conceptnet_dict[normalized_word]

        at_least_one_edge = False
        for edge in edges:
            start_node = edge["start"]["term"]
            end_node = edge["end"]["term"]
            rel = edge["rel"]["label"]
            normalized_start_word = uri_to_label(start_node)  # uri_to_label("/c/en/movies") -> "movies"
            normalized_end_word = uri_to_label(end_node)
            
            if (rel in DISCARDED_RELATIONS or rel not in RELATIONS 
                or edge["start"]["language"] != "en" 
                or edge["end"]["language"] != "en"
                or (normalized_start_word in associated_nodes 
                    and normalized_end_word in associated_nodes)): 
                continue
                #total_vocab_list-->associated_nodes
            at_least_one_edge = True
                 
            if normalized_start_word not in associated_nodes:
                associated_nodes.append(normalized_start_word)
            #total_vocab_list-->associated_node

            if normalized_end_word not in associated_nodes:
                associated_nodes.append(normalized_end_word)
            
            normalized_start_word_idx = associated_nodes.index(normalized_start_word)
            normalized_end_word_idx = associated_nodes.index(normalized_end_word)
            edges_list.append(((normalized_start_word_idx,normalized_end_word_idx), 
                               RELATIONS.index(rel)))
            
            #total_vocab_list = [concept.replace(" ", "_") for concept in total_vocab_list]
            #total_vocab_
        return associated_nodes, edges_list

    SEED = 1234
    random.seed(SEED)
    
    primary_vocab_list = []
    total_vocab_list = []
    edges_list = []
    
    SEQ_LIST = []
    concepts = []
    toks_len = len(toks)
    
    i = 0
    while i < toks_len:
        # Adds entities (n_grams)
        if nertags[i] in ENTITY_BEGINNING:
            ngram_list = [toks[i].text]
            j = i+1
            while j < toks_len and nertags[j] in ENTITY_INSIDE:
                ngram_list.append(toks[j].text)
                j += 1
            ngram = ' '.join(ngram_list)
            concepts.append(ngram)
            #modif
            uri_word = standardized_concept_uri('en', ngram)
            norm_word =  uri_to_label(uri_word)
            if norm_word in total_vocab_list:
                SEQ_LIST.append((i, norm_word, total_vocab_list.index(norm_word)))
            else:
                total_vocab_list, edges_list = _get_concepts(uri_word,total_vocab_list, edges_list, conceptnet_dict)
                SEQ_LIST.append((i, norm_word, total_vocab_list.index(norm_word)))
            i = j
            
        # Adds tokens that are considered as concepts under the Part Of Speech conditions
        elif postags[i] in POS_CONCEPTS and not toks[i].is_stop:
            concepts.append(toks[i].lemma_) # May change lemma by text...
            lemma = toks[i].lemma_
            uri_word = standardized_concept_uri('en', lemma)
            norm_word = uri_to_label(uri_word)
            if norm_word in total_vocab_list:
                SEQ_LIST.append((i, norm_word, total_vocab_list.index(norm_word)))
            else:
                total_vocab_list, edges_list = _get_concepts(uri_word,total_vocab_list, edges_list, conceptnet_dict)
                SEQ_LIST.append((i, norm_word, total_vocab_list.index(norm_word)))
            i += 1
        else:
            i += 1
            
    return (total_vocab_list, edges_list), SEQ_LIST


def generate_graph_for_concept_list(concepts, conceptnet_dict=None, verbose=False):
    """Given a list of concepts, and not a sequence, build the KG from 
    ConceptNet based on this concept list."""
    primary_vocab_list = []
    total_vocab_list = []
    edges_list = []
    concepts_uri =[standardized_concept_uri('en', ngram) for ngram in concepts]
    unique_concepts_uri = list(OrderedDict.fromkeys(concepts_uri))
    for uri_word in tqdm(unique_concepts_uri, total=len(unique_concepts_uri)):
        normalized_word = uri_to_label(uri_word)
        primary_vocab_list.append(normalized_word)

        if normalized_word not in total_vocab_list:
            total_vocab_list.append(normalized_word)

        if conceptnet_dict == None:
            try:
                obj = requests.get('{}{}'.format("http://api.conceptnet.io", 
                                                 uri_word)).json()
                edges = obj["edges"]
            except Exception:
                print("Request error")
                edges = []
                
        else:
            if normalized_word not in conceptnet_dict:
                try:
                    obj = requests.get('{}{}'.format("http://api.conceptnet.io", 
                                                     uri_word)).json()
                    edges = obj["edges"]
                    conceptnet_dict[normalized_word] = edges
                except Exception:
                    print("Request error")
                    edges = []
            else:
                edges = conceptnet_dict[normalized_word]

        at_least_one_edge = False
        for edge in edges:
            start_node = edge["start"]["term"]
            end_node = edge["end"]["term"]
            rel = edge["rel"]["label"]
            normalized_start_word = uri_to_label(start_node)  # uri_to_label("/c/en/movies") -> "movies"
            normalized_end_word = uri_to_label(end_node)
            
            if (rel in DISCARDED_RELATIONS or rel not in RELATIONS 
                or edge["start"]["language"] != "en" 
                or edge["end"]["language"] != "en"
                or (normalized_start_word in total_vocab_list 
                    and normalized_end_word in total_vocab_list)): 
                continue
            at_least_one_edge = True
                 
            if normalized_start_word not in total_vocab_list:
                total_vocab_list.append(normalized_start_word)

            if normalized_end_word not in total_vocab_list:
                total_vocab_list.append(normalized_end_word)
                        
            normalized_start_word_idx = total_vocab_list.index(normalized_start_word)
            normalized_end_word_idx = total_vocab_list.index(normalized_end_word)
            edges_list.append(((normalized_start_word_idx,normalized_end_word_idx), 
                               RELATIONS.index(rel)))
            
    total_vocab_list = [concept.replace(" ", "_") for concept in total_vocab_list]
    
    return (total_vocab_list, edges_list)
#%%
def _get_concepts(uri_word, conceptnet_dict=None): 
        #What about duplicates ??-Don't care, just keep track.
    associated_nodes = []
    normalized_word = uri_to_label(uri_word)
    normalized_word = uri_to_label(uri_word)
    associated_nodes.append(normalized_word)
    if conceptnet_dict == None:
        try:
            obj = requests.get('{}{}'.format("http://api.conceptnet.io", 
                                                 uri_word)).json()
            edges = obj["edges"]
        except Exception:
            print("Request error")
            edges = []
                
    else:
        if normalized_word not in conceptnet_dict:
            try:
                obj = requests.get('{}{}'.format("http://api.conceptnet.io", 
                                                     uri_word)).json()
                edges = obj["edges"]
                conceptnet_dict[normalized_word] = edges
            except Exception:
                print("Request error")
                edges = []
        else:
            edges = conceptnet_dict[normalized_word]

    at_least_one_edge = False
    for edge in edges:
        start_node = edge["start"]["term"]
        end_node = edge["end"]["term"]
        rel = edge["rel"]["label"]
        normalized_start_word = uri_to_label(start_node)  # uri_to_label("/c/en/movies") -> "movies"
        normalized_end_word = uri_to_label(end_node)
            
        if (rel in DISCARDED_RELATIONS or rel not in RELATIONS 
            or edge["start"]["language"] != "en" 
            or edge["end"]["language"] != "en"
            or (normalized_start_word in associated_nodes 
                and normalized_end_word in associated_nodes)): 
            continue
                #total_vocab_list-->associated_nodes
            at_least_one_edge = True
                 
        if normalized_start_word not in associated_nodes:
            associated_nodes.append(normalized_start_word)
            #total_vocab_list-->associated_node

        if normalized_end_word not in associated_nodes:
            associated_nodes.append(normalized_end_word)
            #total_vocab_list-->associated_node
    return associated_nodes


def test_generate_graph():
    nlp = spacy.load('en', disable=['vectors', 'textcat', 'parser'])
    
    resource_path = 'resource'
    
    with open(os.path.join(resource_path, 'vocab_tag.pick'),'rb') as f:
        vocab_tag = pickle.load(f)
    with open(os.path.join(resource_path,'vocab_ner.pick'),'rb') as f:
        vocab_ner = pickle.load(f)

    paragraph = "The Super Bowl is the annual championship game of the National Football League (NFL). The game is the culmination of a regular season that begins in the late summer of the previous calendar year. Normally, Roman numerals are used to identify each game, rather than the year in which it is held. For example, Super Bowl I was played on January 15, 1967, following the 1966 regular season. The sole exception to this naming convention tradition occurred with Super Bowl 50, which was played on February 7, 2016, following the 2015 regular season, and the following year, the nomenclature returned to Roman numerals for Super Bowl LI, following the 2016 regular season. The most recent Super Bowl was Super Bowl LII, on February 4, 2018, following the 2017 regular season."
    
    paragraph_tokend = nlp(reform_text(paragraph))
    postagsm = postag_func(paragraph_tokend, vocab_tag)
    nertags = nertag_func(paragraph_tokend, vocab_ner)
    
    print("paragraph_tokend:", paragraph_tokend)
    print("postagsm:", postagsm)
    print("nertags:", nertags)
    
    total_vocab_list, edges_list = generate_graph(paragraph_tokend, 
                                                  postagsm, 
                                                  nertags,    
                                                  verbose=True)
    
    print(total_vocab_list, edges_list)
    return(total_vocab_list, edges_list)
#%%
"""   
nlp = spacy.load('en', disable=['vectors', 'textcat', 'parser'])

resource_path = '/mnt/data_b/theo/graph_dl/graph_library/resource'
with open(osp.join(resource_path, 'vocab_tag.pick'),'rb') as f:
    vocab_tag = pickle.load(f)
with open(osp.join(resource_path,'vocab_ner.pick'),'rb') as f:
    vocab_ner = pickle.load(f)
#%%
paragraph = "The Super Bowl is the annual championship game of the National Football League (NFL). The game is the culmination of a regular season that begins in the late summer of the previous calendar year. Normally, Roman numerals are used to identify each game, rather than the year in which it is held. For example, Super Bowl I was played on January 15, 1967, following the 1966 regular season. The sole exception to this naming convention tradition occurred with Super Bowl 50, which was played on February 7, 2016, following the 2015 regular season, and the following year, the nomenclature returned to Roman numerals for Super Bowl LI, following the 2016 regular season. The most recent Super Bowl was Super Bowl LII, on February 4, 2018, following the 2017 regular season."    
paragraph_tokend = nlp(reform_text(paragraph))
postagsm = postag_func(paragraph_tokend, vocab_tag)
nertags = nertag_func(paragraph_tokend, vocab_ner)
"""
#%%
#Get numberbatch and see if 'mational football league has an embedding in nb vs check 
#that 'national_football_league' does not.
#%%
if __name__ == "__main__":
    test_generate_graph() 


#%%
#/graph_dl/graph_library/resource/vocab_ner.pick