# -*- coding: utf-8 -*-

import pandas as pd
from quicksectx import IntervalNode, IntervalTree, Interval
import spacy


def overlaps(doc1_ents, doc2_ents,labels=1):
    '''Calculates overlapping entities between two spacy documents. Also checks for matching labels if label=1.
    
    Return:
        Dictionaries with the mapping of matching entity indices:
            keys: entity index from one annotation
            value: matched entity index from other annotation
        
        Ex: "{1 : [2] , 3 : [4,5]}" means that entity 1 from doc1 matches entity 1 in doc2, and entity 3 in doc1 matches 
        entity 4 and 5 from doc2.
    '''
    
    doc1_matches = dict()
    doc2_matches = dict()
    
    tree = IntervalTree()
    for index2,ent2 in enumerate(doc2_ents):
        tree.add(ent2.start_char,ent2.end_char,index2)
    
    for index1,ent1 in enumerate(doc1_ents):
        matches = tree.search(ent1.start_char,ent1.end_char)
        for match in matches:
            index2 = match.data #match.data is the index of doc2_ents
            if ((labels == 0) | (doc2_ents[index2].label_ == ent1.label_)):
                if index1 not in doc1_matches.keys():
                    doc1_matches[index1] = [index2]
                else:
                    doc1_matches[index1].append(index2)
                if index2 not in doc2_matches.keys():
                    doc2_matches[index2] = [index1]
                else:
                    doc2_matches[index2].append(index1)
                
    return doc1_matches, doc2_matches

def df_overlaps(docs1_df, docs2_df,labels=1):
    '''Calculates overlapping entities between two spacy documents. Also checks for matching labels if label=1.
    
    Return:
        Dictionaries with the mapping of matching entity indices:
            keys: entity index from one annotation
            value: matched entity index from other annotation
        
        Ex: "{1 : [2] , 3 : [4,5]}" means that entity 1 from doc1 matches entity 1 in doc2, and entity 3 in doc1 matches 
        entity 4 and 5 from doc2.
    '''
    
    doc1_matches = dict()
    doc2_matches = dict()
    
    tree = IntervalTree()
    for index2,row2 in docs2_df.iterrows():
        tree.add(row2['start loc'],row2['end loc'],index2)
    
    for index1,row1 in docs1_df.iterrows():
        matches = tree.search(row1['start loc'],row1['end loc'])
        for match in matches:
            index2 = match.data #match.data is the index of doc2_ents
            if ((labels == 0) | (docs2_df.loc[index2,'Concept Label'] == row1['Concept Label'])):
                if index1 not in doc1_matches.keys():
                    doc1_matches[index1] = [index2]
                else:
                    doc1_matches[index1].append(index2)
                if index2 not in doc2_matches.keys():
                    doc2_matches[index2] = [index1]
                else:
                    doc2_matches[index2].append(index1)
                
    return doc1_matches, doc2_matches

def exact_match(doc1_ents, doc2_ents, labels):
    '''calculate whether two ents have exact overlap
    returns bool
    '''
    
    doc1_matches = dict()
    doc2_matches = dict()

    doc1_ent_dict = dict()
    doc2_ent_dict = dict()
    
    for index1,ent1 in enumerate(doc1_ents):
        if labels == 1: #If checking for labels, then include this in the tuple's to-be-compared elements
            doc1_ent_dict[(ent1.start_char,ent1.end_char,ent1.label_)] = index1
        else:
            doc1_ent_dict[(ent1.start_char,ent1.end_char)] = index1
            
    for index2,ent2 in enumerate(doc2_ents):
        if labels == 1:    
            doc2_ent_dict[(ent2.start_char,ent2.end_char,ent2.label_)] = index2
        else:
            doc2_ent_dict[(ent2.start_char,ent2.end_char)] = index2
        
    doc1_ent_set = set(doc1_ent_dict.keys())
    doc2_ent_set = set(doc2_ent_dict.keys())
    
    matched_ents = doc1_ent_set.intersection(doc2_ent_set)
    
    for match in matched_ents:
        index1 = doc1_ent_dict[match]
        index2 = doc2_ent_dict[match]
        doc1_matches[index1] = [index2]
        doc2_matches[index2] = [index1]
        
    return doc1_matches, doc2_matches

def agreement(doc1, doc2, loose=1, labels=1, ent_or_span = 'ent'):
    '''Calculates confusion matrix for agreement between two documents.
    
       returns true positive, false positive, and false negative
    '''
    if (type(doc1) is tuple) or (type(doc1) is spacy.tokens.span_group.SpanGroup) and \
    (type(doc2) is tuple) or (type(doc2) is spacy.tokens.span_group.SpanGroup):
        doc1_ents = doc1
        doc2_ents = doc2
    elif (type(doc1) is spacy.tokens.doc.Doc) and (type(doc2) is spacy.tokens.doc.Doc):
        if ent_or_span == 'ent':
            doc1_ents = doc1.ents
            doc2_ents = doc2.ents
        elif ent_or_span == 'span':
            if len(doc1.spans) > 1:
                #raise error
                print("Error: cannot distinquish which span group to use from doc1.")
                return
            else:
                span_group = list(doc1.spans.keys())[0]
                doc1_ents = doc1.spans[span_group]
                doc2_ents = doc2.spans[span_group]
        else:
            #raise error
            print("Error: Must select 'span' or 'ent' for ent_or_span option.")
            return
    else:
        #raise error
        print("Error: Input must be of type 'tuples', 'spacy.tokens.span_group.SpanGroup', or 'spacy.tokens.doc.Doc'")
        return
        
    if loose:
        doc1_matches, doc2_matches = overlaps(doc1_ents, doc2_ents, labels)
    else:
        doc1_matches, doc2_matches = exact_match(doc1_ents, doc2_ents, labels)
    
    return conf_matrix(doc1_matches,doc2_matches,len(doc1_ents),len(doc2_ents))


def corpus_agreement(docs1, docs2, loose=1, labels=1,ent_or_span='ent'):
    '''calculate f1 over an entire corpus of documents'''
    corpus_tp, corpus_fp, corpus_fn = (0,0,0)
    
    if isinstance(docs1, pd.DataFrame):
        for doc_name in docs1['doc name'].unique():
            docs1_df = docs1[docs1['doc name'] == doc_name]
            docs2_df = docs2[docs2['doc name'] == doc_name]
            doc1_matches,doc2_matches = df_overlaps(docs1_df,docs2_df)
            tp,fp,fn = conf_matrix(doc1_matches,doc2_matches,docs1_df.shape[0],docs2_df.shape[0])
            corpus_tp += tp
            corpus_fp += fp
            corpus_fn += fn
    elif type(docs1[0]) is spacy.tokens.doc.Doc:
        for i, doc1 in enumerate(docs1):
            tp,fp,fn = agreement(doc1, docs2[i],loose,labels,ent_or_span)
            corpus_tp += tp
            corpus_fp += fp
            corpus_fn += fn
    else:
        #raise error
        print('Input Error: Input must be iterable of spacy documents, or dataframe.')
        return
    
    data = {'IAA' : [pairwise_f1(corpus_tp,corpus_fp,corpus_fn)], 'Recall' : [corpus_tp/float(corpus_tp+corpus_fp)], 'Precision' : [corpus_tp/float(corpus_tp+corpus_fn)],\
           'True Positives' : [corpus_tp] , 'False Positives' : [corpus_fp], 'False Negative' : [corpus_fn]}
    
    return pd.DataFrame(data)

def pairwise_f1(tp,fp,fn):
    '''calculate f1 given true positive, false positive, and false negative values'''
    
    return (2*tp)/float(2*tp+fp+fn)

def conf_matrix(doc1_matches,doc2_matches,doc1_ent_num,doc2_ent_num):

    doc1_match_num = len(doc1_matches.keys())
    doc2_match_num = len(doc2_matches.keys())
    
    duplicate_matches = 0
    for value in doc2_matches.values():
        duplicate_matches += len(value) - 1
    
    tp = doc1_match_num - duplicate_matches #How many entity indices from doc1 matched, minus duplicated matches
    fp = doc2_ent_num - doc2_match_num #How many entities from doc2 that didn't match
    fn = doc1_ent_num - doc1_match_num #How many entities from doc1 that didn't match
    
    return (tp,fp,fn)


