# -*- coding: utf-8 -*-

import pandas as pd
from quicksectx import IntervalNode, IntervalTree, Interval
import spacy

#Column names (strings) for relevant columns in dataframes
df_start_char = 'start loc' #column containing starting positions of ents
df_end_char = 'end loc' #column containing ending positions of ents
df_concept_label = 'Concept Label' #column containing label of ent
df_doc_name = 'doc name' #column containing document name/file name

def overlaps(doc1_ents, doc2_ents,labels=1):
    '''Calculates overlapping entities between two spacy documents. Also checks for matching labels if label=1.
    Resultant dictionaries can be used to calculate true positive, false positive, and false negatives and can consider duplicate matches.
    
    Return:
        Dictionaries with the mapping of matching entity indices:
            keys: entity index from one annotation
            value: matched entity index from other annotation
        
        Ex: "{1 : [2] , 3 : [4,5]}" means that entity 1 from doc1 matches entity 2 in doc2, and entity 3 in doc1 matches 
        entity 4 and 5 from doc2.
    '''
    
    doc1_matches = dict()
    doc2_matches = dict()
    
    tree = IntervalTree() #Navigating a tree avoids needing to iterate and compare all ents in doc2 for each doc1 ent (ie. avoids a nested for-loop). This is faster.
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

def df_overlaps(docs1_df, docs2_df,labels=1,attributes=[]):
    '''Calculates overlapping entities between two dataframes, each containing entity information for a single document.
        Also checks for matching labels if label=1.
        Works the same as "overlaps" function, but uses dataframes instead of documents.
    
    Return:
        Dictionaries with the mapping of matching entity indices:
            keys: entity index from one dataframe
            value: matched entity index from other dataframe
        
            Ex: "{1 : [2] , 3 : [4,5]}" means that the entity at index 1 from docs1_df matches the entity indexed at 1 in docs2_df, and 
            entity indexed at 3 in docs1_df matches entity indexed at 4 and 5 from docs2_df.
    '''
    
    doc1_matches = dict()
    doc2_matches = dict()
    
    tree = IntervalTree()
    for index2,row2 in docs2_df.iterrows():
        tree.add(row2[df_start_char],row2[df_end_char],index2)
    
    for index1,row1 in docs1_df.iterrows():
        matches = tree.search(row1[df_start_char],row1[df_end_char])
        for match in matches:
            index2 = match.data #match.data is the index of doc2_ents
            attributes_match = 1
            for attribute in attributes: #check if attributes match
                if row1[attribute] != docs2_df.loc[index2,attribute]:
                    attributes_match = 0
                    break
            if ((labels == 0) | (docs2_df.loc[index2,df_concept_label] == row1[df_concept_label])) & (attributes_match):
                if index1 not in doc1_matches.keys():
                    doc1_matches[index1] = [index2]
                else:
                    doc1_matches[index1].append(index2)
                if index2 not in doc2_matches.keys():
                    doc2_matches[index2] = [index1]
                else:
                    doc2_matches[index2].append(index1)
                
    return doc1_matches, doc2_matches

def exact_match(doc1_ents, doc2_ents, labels=1):
    '''Calculates entities in exactly the same position between two spacy documents. Also checks for matching labels if label=1.
    
    Return:
        Dictionaries with the mapping of matching entity indices:
            keys: entity index from one document annotations
            value: matched entity index from other document annotations
        
        Ex: "{1 : [2] , 3 : [4,5]}" means that entity 1 from doc1 matches entity 2 in doc2, and entity 3 in doc1 matches 
        entity 4 and 5 from doc2.
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

def df_exact_match(docs1_df, docs2_df, labels=1,attributes=[]):
    '''Calculates entities in exactly the same position between two dataframes containing entity information.
    Also checks for matching labels if label=1.
    Same as "exact_match", but uses dataframes.
    
    Return:
        Dictionaries with the mapping of matching entity indices:
            keys: entity index from one document's dataframe
            value: matched entity index from other document's dataframe
        
        Ex: "{1 : [2] , 3 : [4,5]}" means that entity 1 from doc1's df matches entity 2 in doc2's df, and entity 3 in doc1's df matches 
        entity 4 and 5 from doc2's df.
    '''
    
    
    doc1_matches = dict()
    doc2_matches = dict()

    doc1_ent_dict = dict()
    doc2_ent_dict = dict()
    
    for index1,row1 in docs1_df.iterrows():
        key_array = []
        key_array.extend([row1[df_start_char],row1[df_end_char],row1[df_concept_label]])
        for attribute in attributes:
            key_array.append(row1[attribute])
        if labels == 1: #If checking for labels, then include this in the tuple's to-be-compared elements
            key_array.append(row1[df_concept_label])
        doc1_ent_dict[tuple(key_array)] = index1
            
    for index2,row2 in docs2_df.iterrows():
        key_array = []
        key_array.extend([row2[df_start_char],row2[df_end_char],row2[df_concept_label]])
        for attribute in attributes:
            key_array.append(row2[attribute])
        if labels == 1: #If checking for labels, then include this in the tuple's to-be-compared elements
            key_array.append(row2[df_concept_label])
        doc2_ent_dict[tuple(key_array)] = index2
        
    doc1_ent_set = set(doc1_ent_dict.keys())
    doc2_ent_set = set(doc2_ent_dict.keys())
    
    matched_ents = doc1_ent_set.intersection(doc2_ent_set)
    
    for match in matched_ents:
        index1 = doc1_ent_dict[match]
        index2 = doc2_ent_dict[match]
        doc1_matches[index1] = [index2]
        doc2_matches[index2] = [index1]
        
    return doc1_matches, doc2_matches

def extract_ents(doc1, doc2, loose=1, labels=1, ent_or_span = 'ent'):
    '''Establishes entities or span lists based on input. Then calls "overlaps" and "conf_matrix" using relevant arguments.
    You can also manually call "overlaps", then "conf_matrix" with relevant entities/span lists to get equivalent results to this function.
    This function is simply to get these arguments from spacy documents conveniently.
    
    Arguments:
        doc1: Either a spacy document, tuple/list of entities/spans, or spangroup. Considered the golden/correct annotation for fp,fn.
        doc2: Either a spacy document, tuple/list of entities/spans, or spangroup.
        loose: Boolean. 1 indicates to consider any overlap. 0 indicates to only consider exact matches.
        labels: Boolean. 1 indicates to consider labels as matching criteria.
        ent_or_span: String of either 'ent' or 'span'. 'ent' indicates to compare doc.ents between documents. 'span' indicates to
            compare doc1's only spangroup (note that doc1 must have only 1 spangroup) with doc2's equivalently named spangroup. This
            argument is only relevant if passing in spacy document (ie. can be ignored if passing in tuple/list of ents/spans)
    
    Return:
        Tuple of format "(true positive, false positive, false negative)".
    '''
    if (isinstance(doc1,tuple) or isinstance(doc1,list) or isinstance(doc1,spacy.tokens.span_group.SpanGroup)) and \
    (isinstance(doc2,tuple) or isinstance(doc2,list) or isinstance(doc2,spacy.tokens.span_group.SpanGroup)):
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
        print("Error: Input must be of type 'tuples','list', 'spacy.tokens.span_group.SpanGroup', or 'spacy.tokens.doc.Doc'")
        return
    
    return (doc1_ents,doc2_ents)


def corpus_agreement(docs1, docs2, loose=1, labels=1,ent_or_span='ent',attributes=[]):
    '''Calculates f1 over an entire corpus of documents.
    
    Arguments:
        docs1: Either a list of spacy documents, list containing inner tuples/lists of entities/spans, list of spangroups, or a dataframe. 
            Considered the golden/correct annotation for fp,fn.
        docs2: Either a list of spacy documents, list of tuples/lists of entities/spans, list of spangroups, or a dataframe. 
        loose: Boolean. 1 indicates to consider any overlap. 0 indicates to only consider exact matches.
        labels: Boolean. 1 indicates to consider labels as matching criteria.
        ent_or_span: String of either 'ent' or 'span'. 'ent' indicates to compare doc.ents between documents. 'span' indicates to
            compare doc1's only spangroup (note that doc1 must have only 1 spangroup) with doc2's equivalently named spangroup. This
            argument is only relevant if passing in a list of spacy documents (ie. can be ignored if passing in a list of
            tuple/list of ents/spans/spangroups or dataframe)
            
    Returns:
        Simple Dataframe containing IAA, Recall, Precision, true positive count, false positive count, and false negative count.
    '''
    corpus_tp, corpus_fp, corpus_fn = (0,0,0)
    agreement_df = pd.DataFrame(columns=["doc name","Annotation_1","Annotation_2", "Annot_1_label", "Annot_1_char", "Annot_2_label", "Annot_2_char", "Overall_start_char", "Exact Match?", "Duplicate Matches?", "Overlap?"])
    if isinstance(docs1, pd.DataFrame):
        docs1['Span Text'] = docs1['Span Text'].str.replace('\n',' ')
        docs2['Span Text'] = docs2['Span Text'].str.replace('\n',' ')
        docs1['Span Text'] = docs1['Span Text'].str.replace('\t',' ')
        docs2['Span Text'] = docs2['Span Text'].str.replace('\t',' ')
        for doc_name in docs1[df_doc_name].unique():
            docs1_df = docs1[docs1[df_doc_name] == doc_name]
            docs2_df = docs2[docs2[df_doc_name] == doc_name]
            if loose==1:
                doc1_matches,doc2_matches = df_overlaps(docs1_df,docs2_df,labels,attributes)
            else:
                doc1_matches,doc2_matches = df_exact_match(docs1_df,docs2_df,labels,attributes)
            tp,fp,fn = conf_matrix(doc1_matches,doc2_matches,docs1_df.shape[0],docs2_df.shape[0])
            corpus_tp += tp
            corpus_fp += fp
            corpus_fn += fn
            new_agreement_df = create_agreement_df(doc1_matches,doc2_matches,docs1_df,docs2_df)
            new_agreement_df.index += agreement_df.shape[0]
            agreement_df = pd.concat([agreement_df,new_agreement_df])
            
            
    elif (type(docs1[0]) is spacy.tokens.doc.Doc) | ((isinstance(docs1[0],tuple) or isinstance(docs1[0],list) or isinstance(docs1[0],spacy.tokens.span_group.SpanGroup)) and \
    (isinstance(docs2[0],tuple) or isinstance(docs2[0],list) or (isinstance(docs2[0],spacy.tokens.span_group.SpanGroup)))):
        for i, doc1 in enumerate(docs1):
            doc1_ents,doc2_ents = extract_ents(doc1, docs2[i],loose,labels,ent_or_span)
            if loose:
                doc1_matches, doc2_matches = overlaps(doc1_ents, doc2_ents, labels)
            else:
                doc1_matches, doc2_matches = exact_match(doc1_ents, doc2_ents, labels)
            tp,fp,fn = conf_matrix(doc1_matches,doc2_matches,len(doc1_ents),len(doc2_ents))
            corpus_tp += tp
            corpus_fp += fp
            corpus_fn += fn
    else:
        #raise error
        print('Input Error: Input must be iterable of spacy documents, or dataframe.')
        return
    
    data = {'IAA' : [pairwise_f1(corpus_tp,corpus_fp,corpus_fn)], 'Recall' : [corpus_tp/float(corpus_tp+corpus_fp)], 'Precision' : [corpus_tp/float(corpus_tp+corpus_fn)],\
           'True Positives' : [corpus_tp] , 'False Positives' : [corpus_fp], 'False Negative' : [corpus_fn]}
    
    return (pd.DataFrame(data),agreement_df)

def pairwise_f1(tp,fp,fn):
    '''Calculates f1 given true positive, false positive, and false negative values
    
    Returns:
        pairwise_f1/IAA float value'''
    
    return (2*tp)/float(2*tp+fp+fn)

def conf_matrix(doc1_matches,doc2_matches,doc1_ent_num,doc2_ent_num):
    '''Calculates true positive, false positive, and false negative counts given mapping dictionary from overlap functions. 
    Note that duplicate matches (eg. two ents from one doc match one ent in the other) only count as a single tp and do not affect fp,fn.
    
    Arguments:
        doc1_matches: dictionary containing mappings of matched entities/spans. This is the output of the overlap functions.
        doc2_matches: dictionary containing mappings of matched entities/spans. This is the output of the overlap functions.
        doc1_ent_num: number of total entities/spans in doc1.
        doc2_ent_num: number of total entities/spans in doc2.
    
    Returns:
        Tuple of format "(true positive, false positive, false negative)".
    '''
                                                     
                                                     
    doc1_match_num = len(doc1_matches.keys())
    doc2_match_num = len(doc2_matches.keys())
    
    duplicate_matches = 0
    for value in doc2_matches.values():
        duplicate_matches += len(value) - 1 #Duplicate matches are anytime an ent from doc2 matches more than 1 ent from doc1 (ie. doc2 dictionary value > 1)
    
    tp = doc1_match_num - duplicate_matches #How many entity indices from doc1 matched, minus duplicated matches
    fp = doc2_ent_num - doc2_match_num #How many entities from doc2 that didn't match
    fn = doc1_ent_num - doc1_match_num #How many entities from doc1 that didn't match
    #Note that for fp, it only considers ents from doc2 that had no match. Therefore if two or more ent from doc2 match one ent from doc1
    #than none of these ents will be considered a fp (these ents are only counted as a single tp). Same logic applies for fn.
    
    return (tp,fp,fn)

#fix \n error, fix adding index2's, fix index, get rid of 'magic' strings

def create_agreement_df(doc1_matches,doc2_matches,doc1_ents,doc2_ents):
    if "Concept Label" in list(doc1_ents.columns) and "Concept Label" in list(doc2_ents.columns):
        label=1
    else:
        label=0
    result_dict = {"doc name" : [],"Annotation_1" : [],"Annotation_2" : [], "Annot_1_label" : [], "Annot_1_char" : [], "Annot_2_label" : [], "Annot_2_char" : [], "Overall_start_char" : [], "Exact Match?" : [], "Duplicate Matches?" : [], "Overlap?" : [], "Index" : []}
    doc_name = doc1_ents['doc name'].tolist()[0]
    for index1 in doc1_ents.index: #iterate through all ents inset one
        if index1 in doc1_matches.keys(): #Cases where the ent has a match
            #if another index1 is in doc2_matches.values(), then add it to this row
            first_index2 = sorted(doc1_matches[index1])[0]
            first_index1 = sorted(doc2_matches[first_index2])[0]
            if first_index1 < index1:
                #Add to index: sorted(doc2_matches[first_index2])[0]
                duplicate_match_index = result_dict["Index"].index(first_index1)
                result_dict["Annotation_1"][duplicate_match_index] += " || " + doc1_ents.loc[index1,'Span Text']
                result_dict["Duplicate Matches?"][duplicate_match_index] = 1
                result_dict["Overlap?"][duplicate_match_index] = 1
                result_dict["Annot_1_char"][duplicate_match_index] += " || " + str(doc1_ents.loc[index1,'start loc']) + "-" + str(doc1_ents.loc[index1,'end loc'])
                if result_dict["Overall_start_char"][duplicate_match_index] > int(doc1_ents.loc[index1,'start loc']):
                    result_dict["Overall_start_char"][duplicate_match_index] = doc1_ents.loc[index1,'start loc']
                if label==1:
                    result_dict["Annot_1_label"][duplicate_match_index] += " || " + doc1_ents.loc[index1,'Concept Label']
                
                #take the index2's from doc1_matches[index1] that don't match doc1_matches[first_index1] (or previous index1) and add those
                
                
            else:
                result_dict["doc name"].append(doc_name)
                result_dict["Index"].append(index1)
                result_dict["Annotation_1"].append(doc1_ents.loc[index1,'Span Text'])
                annot_2 = ""
                annot_2_label = ""
                annot_2_char = ""
                overall_start_char = int(doc1_ents.loc[index1,'start loc'])
                first_time=1
                annot_2_start_char = -1
                annot_2_end_char = -1
                annot_1_start_char = int(doc1_ents.loc[index1,'start loc'])
                annot_1_end_char = int(doc1_ents.loc[index1,'end loc'])
                exact_match = 0
                for index2 in sorted(doc1_matches[index1]):
                    annot_2_start_char = int(doc2_ents.loc[index2,'start loc'])
                    annot_2_end_char = int(doc2_ents.loc[index2,'end loc'])
                    if (annot_1_start_char == annot_2_start_char) and (annot_1_end_char == annot_2_end_char):
                        exact_match = 1
                    if annot_2_start_char < overall_start_char:
                        overall_start_char = annot_2_start_char
                    if first_time ==1:
                        first_time=0
                        annot_2_char = str(annot_2_start_char) + "-" + str(annot_2_end_char)
                        annot_2 += doc2_ents.loc[index2,'Span Text']
                        if label==1:
                            annot_2_label = doc2_ents.loc[index2,'Concept Label']
                    else:
                        if label==1:
                            annot_2_label += " || " + doc2_ents.loc[index2,'Concept Label']
                        annot_2 += " || " + doc2_ents.loc[index2,'Span Text']
                        annot_2_char += " || " + str(annot_2_start_char) + "-" + str(annot_2_end_char)
                result_dict["Annotation_2"].append(annot_2)
                result_dict["Annot_2_label"].append(annot_2_label)
                result_dict["Annot_2_char"].append(annot_2_char)
                if label == 1:
                    result_dict["Annot_1_label"].append(doc1_ents.loc[index1,'Concept Label'])
                else:
                    result_dict["Annot_1_label"].append("")
                result_dict["Annot_1_char"].append(str(annot_1_start_char) + "-" + str(annot_1_end_char))
                result_dict["Overall_start_char"].append(overall_start_char)
                result_dict["Exact Match?"].append(exact_match)
                if len(doc1_matches[index1]) > 1:
                    result_dict["Duplicate Matches?"].append(1)
                else:
                    result_dict["Duplicate Matches?"].append(0)
                result_dict["Overlap?"].append(1)
        else: #Cases where an ent in doc1 doesn't have a match
            result_dict["doc name"].append(doc_name)
            result_dict["Index"].append(index1)
            result_dict["Annotation_1"].append(doc1_ents.loc[index1,'Span Text'])
            result_dict["Annotation_2"].append("")
            if label == 1:
                result_dict["Annot_1_label"].append(doc1_ents.loc[index1,'Concept Label'])
                result_dict["Annot_2_label"].append("")
            else:
                result_dict["Annot_1_label"].append("")
                result_dict["Annot_2_label"].append("")
            result_dict["Overall_start_char"].append(int(doc1_ents.loc[index1,'start loc']))
            result_dict["Annot_1_char"].append(str(doc1_ents.loc[index1,'start loc']) + "-" + str(doc1_ents.loc[index1,'end loc']))
            result_dict["Annot_2_char"].append("")
            result_dict["Exact Match?"].append(0)
            result_dict["Duplicate Matches?"].append(0)
            result_dict["Overlap?"].append(0)
    del result_dict["Index"] #Only needed to link new index1 with old index1 when there's a duplicate match with the same ent from doc2
    for index2 in doc2_ents.index: #Cases where an ent in doc2 doesn't have a match
        if index2 not in doc2_matches.keys():
            result_dict["doc name"].append(doc_name)
            result_dict["Annotation_1"].append("")
            result_dict["Annotation_2"].append(doc2_ents.loc[index2,'Span Text'])
            if label == 1:
                result_dict["Annot_1_label"].append("")
                result_dict["Annot_2_label"].append(doc2_ents.loc[index2,'Concept Label'])
            else:
                result_dict["Annot_1_label"].append("")
                result_dict["Annot_2_label"].append("")
            result_dict["Annot_1_char"].append("")
            result_dict["Annot_2_char"].append(str(doc2_ents.loc[index2,'start loc']) + "-" + str(doc2_ents.loc[index2,'end loc']))
            result_dict["Overall_start_char"].append(int(doc2_ents.loc[index2,'start loc']))
            result_dict["Exact Match?"].append(0)
            result_dict["Duplicate Matches?"].append(0)
            result_dict["Overlap?"].append(0)
    return pd.DataFrame.from_dict(result_dict).sort_values(by=['Overall_start_char']).reset_index(drop=True)


