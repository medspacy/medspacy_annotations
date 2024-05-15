import pandas as pd
from quicksectx import IntervalNode, IntervalTree, Interval
import numpy as np

class AnnotationAggregator(object):
    """
    AnnotationAggregator imports annotations as spaCy and dataframe objects, and calculates performance metrics and facilitates
        error analyses of the disagreed cases using the dataframe.
    """
    
    CONTEXT_CHAR_AMOUNT = 50 #For the span range to output in resultant table
    OUTPUT_DECIMAL_PLACES = 3
    NULL_RELATIONSHIP_VALUES = [np.nan,"",str(np.nan),"NaN","Null","NULL","nan","NA","na","Na"] #For checking null relationship values
    
    def __init__(
        self,
        annot = dict(),#:dict,
        doc_col = 'DocID',#:str,
        span_col = 'annotatedSpan',#:str,
        label_col = 'spanLabel',#:str,
        attr = None,#:object,
        label_ind_attr = None,
        label_dep_attr = None,
        rel_source_col = None,
        rel_target_col = None,
        span_start_col = 'spanStartChar',#:str,
        span_end_col = 'spanEndChar',#:str,
        text = None#:dict or filepath
    ):
        """
        Creates new AnnotationAggregator.

        Args:
            annot: Dict containing names as the key. The value is an accepted input type containing annotation information, 
                or a list of these accepted input types. Accepted input types currently include: a spaCy/medspaCy document, a list of spaCy
                entities or spans.
            doc_col: If using a dataframe, this specifies the name of the column containing the document names
            label_col: If using a dataframe, this specifies the name of the column containing the labels
            attr_col: If using a dataframe, this specifies the name of the column containing the labels
            span_start_col: If using a dataframe, this specifies the name of the column containing the 
                starting span location
            span_end_col: If using a dataframe, this specifies the name of the column containing the 
                ending span location
        """

        self.doc_col = doc_col
        self.span_col = span_col
        self.label_col = label_col
        self.spacy_docs = dict()
    
        
        if (attr == None):
            self.attr = []
        elif not isinstance(attr,list) and (attr != None):
            self.attr = [attr]
        elif isinstance(attr,list):
            self.attr = attr
            
        if not isinstance(label_ind_attr,list) and (label_ind_attr != None):
            self.label_ind_attr = [label_ind_attr]
        else:
            self.label_ind_attr = label_ind_attr
            
        if not isinstance(label_dep_attr,list) and (label_dep_attr != None):
            self.label_dep_attr = [label_dep_attr]
        else:
            self.label_dep_attr = label_dep_attr
        
        if self.label_ind_attr != None:
            for att in self.label_ind_attr:
                if (att != None) and (att not in self.attr):
                    self.attr.append(att)
                    
        if self.label_dep_attr != None:
            for att in self.label_dep_attr:
                if (att != None) and (att not in self.attr):
                    self.attr.append(att)
                    
        
        if not isinstance(rel_source_col,list) and (rel_source_col != None):
            self.rel_source_col = [rel_source_col]
        else:
            self.rel_source_col = rel_source_col
        if (rel_source_col == None):
            self.rel_source_col = []
            
        
        if not isinstance(rel_target_col,list) and (rel_target_col != None):
            self.rel_target_col = [rel_target_col]
        else:
            self.rel_target_col = rel_target_col
            
        if (rel_target_col == None):
            self.rel_target_col = []
                    
        self.span_start_col = span_start_col
        self.span_end_col = span_end_col
        
        self.match_dict = dict()
        self.agreement_metrics = dict()
    
        #annot_df_list = []
        #for doc_id,ann in enumerate(annot): #for each corpus
        #    if not isinstance(ann, pd.DataFrame):
        #        if (type(ann[0]) is spacy.tokens.doc.Doc) | ((isinstance(ann[0],tuple) or isinstance(ann[0],list) or\
        #        isinstance(ann[0],spacy.tokens.span_group.SpanGroup)): #also need to check if it's just a single ent
        #            annot_df_list.append(spacy_obj_to_df(ann,doc_id))
        #        else:
        #            #raise error
        #            print('Input Error: Input must be type or iterable of spacy documents, or dataframe.')
        #            break
        #    else:
        #        annot_df_list.append(ann)
                                                        
        #self.annot = annot_df_list
        
        if isinstance(annot,dict):
            self.annot = annot #For now lets just assume the input type is correctct

        try:
            self.text = self.extract_txt(text)
        except:
            print('Error importing txt files from file, will try to use dictionary or spacy text if provided.')
            if isinstance(text,dict):
                self.text = text
            
                
    def calculate_agreement(self):
        
        if (self.annot != None) & (isinstance(self.annot,dict)):
            all_docID = set()
            for key,item in self.annot.items():
                self.annot[key][self.doc_col] = item[self.doc_col].astype(str) #set all document values as strings so it matches txt file titles
                all_docID = (all_docID | set(item[self.doc_col].to_list()))
            try:
                if len(all_docID - set(self.text.keys())) > 0:
                    print("There are missing .txt files: " + str(all_docID - set(self.text.keys())))
                    self.text = None #Not usable if annotations have texts not seen in the text dict
            except:
                self.text = None
                                      
        
        key_list = list(self.annot.keys())
        index_list = range(len(key_list))
        if len(index_list) > 1:
            for i in index_list:
                for i2 in index_list[i+1:]:
                    key1 = key_list[i]
                    key2 = key_list[i2]
                    df1 = self.annot[key1]
                    df2 = self.annot[key2]
                    self.match_dict["".join((str(key1),"-",str(key2)))] = self.create_agreement_df_corpus(df1,df2)
                    self.agreement_metrics["".join((str(key1),"-",str(key2)))] = self.df_agreement_metrics(self.match_dict["".join((str(key1),"-",str(key2)))])
        else:
            print("To get agreement metrics and dataframe, you must input at least 2 annotations")
                
    def get_raw_df(self):
        return self.annot
    
    def get_spacy_docs(self):
        return self.spacy_docs
    
    def get_text(self):
        return self.text
                                                             
    def get_agreement_dict(self):
        if self.match_dict == dict():
            self.calculate_agreement()
        return self.match_dict
    
    def get_agreement_metrics(self):
        if self.agreement_metrics == dict():
            self.calculate_agreement()
        return self.agreement_metrics
                                                             
    def extract_txt(self,folder_path):                               
        import os
        def read_txt_file(file_path):
            with open(file_path, 'r') as file:#, encoding='utf-8'
                return file.read()
        
        def collect_documents(folder_path):
            documents = {}
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    document_name = os.path.splitext(file_name)[0]
                    documents[document_name] = read_txt_file(file_path)
            return documents
        
        return collect_documents(folder_path)
                                                             
    
    def add_ehost_files(self,annot_dirs,schema_file=None):
        try:
            import medspacy
            import ehost_reader
            from spacy.lang.en import English
            import os
        except:
            print('MedspaCy, ehost_reader, spacy.lang.en library failed to import. Pleast ensure your sys paths are set to the appropriate directories')
            raise RuntimeError("Error occurred with import")
        attr_list = set()
        if not isinstance(annot_dirs,list):
            annot_dirs = [annot_dirs]
        for annot_dir in annot_dirs:
            #if schema_file==None:
            schema_file = os.path.join(annot_dir,'config/projectschema.xml')
            if attr_list == set():
                attr_list = ehost_reader.EhostDocReader(txt_extension = 'txt', nlp=English(),recursive=True, support_overlap=True, schema_file = schema_file).attr_names #put this in for loop
            print("Starting ",annot_dir)
            print('attr list',attr_list)
            dir_reader = ehost_reader.EhostDirReader(nlp=English(),recursive=True, support_overlap=True, schema_file = schema_file)
            print("End of dir reader print out")
            docs = dir_reader.read(txt_dir = annot_dir)
            annot_dir_name = os.path.basename(annot_dir)
            self.spacy_docs[annot_dir_name] = dict()
            for doc in docs:
                self.spacy_docs[annot_dir_name][doc._.doc_name] = doc
            long_df = self.dir_extract(docs,attr_list)
            long_df.replace('NULL','NA',inplace=True)
            long_df.fillna('NA',inplace=True)
            attrs = long_df['attrName'].unique().tolist()
            try:
                attrs.remove('NA')
            except:
                pass
            flat1_df = long_df.pivot_table(index=['DocID','annotatedSpan','spanStartChar','spanEndChar','spanLabel','relLabel','relID','spanID'],columns='attrName',values='attrVal',aggfunc='first').reset_index()
            flat1_df.replace('NULL','NA',inplace=True)
            flat1_df.fillna('NA',inplace=True)
            if 'NA' in flat1_df.columns:
                flat1_df.drop('NA',axis=1,inplace=True)
            rels = flat1_df['relLabel'].unique().tolist()
            try:
                rels.remove('NA')
            except:
                pass
            if flat1_df['relLabel'].nunique() > 1: #If relationships are detected
                flat1_df = flat1_df.pivot_table(index=['DocID','annotatedSpan','spanStartChar','spanEndChar','spanLabel','spanID']+attrs,columns='relLabel',values='relID',aggfunc='first').reset_index()
                flat1_df.replace('NULL','NA',inplace=True)
                flat1_df.fillna('NA',inplace=True)
                if 'NA' in flat1_df.columns:
                    flat1_df.drop('NA',axis=1,inplace=True)
                self.rel_target_col = ['spanID']
                rel_update = set(self.rel_source_col)
                rel_update.update(rels)
                self.rel_source_col = list(rel_update)
            else:
                self.rel_target_col = []
                self.rel_source_col = []
                
            attr_update = set(self.attr)
            attr_update.update(attrs)
            self.attr = list(attr_update)
            self.annot[annot_dir_name] = flat1_df
        
        self.doc_col = 'DocID'
        self.span_col = 'annotatedSpan'
        self.label_col = 'spanLabel'
        self.span_start_col = 'spanStartChar'
        self.span_end_col = 'spanEndChar'
        try:
            if self.text == None:
                self.text = self.extract_txt(os.path.join(annot_dir,'corpus'))
            elif isinstance(self.text,dict):
                self.text.update(self.extract_txt(os.path.join(annot_dir,'corpus')))
        except:
            print('Error trying to import txt files from ehost \'corpus\' directory.')    
        
    
    def dir_extract(self,docs,attribute_list):
        doc_names = []
        span_texts = []
        span_start = []
        span_end = []
        span_label = []
        span_attr_name = []#one class may have multiple attributes
        span_attr_value = []
        span_id = [] #current span Ehost ID (relation source ID)
        relSpan_id=[] #relation target Ehost ID
        relLabel = [] # relation label
        for doc in docs:
            classes = doc._.concepts.keys() #get classes list
            relations = doc._.relations #get relation list
            #print(doc._.doc_name)
            for cls in doc._.concepts.keys():
                for splist in doc._.concepts[cls]:
                    sp = splist[0]
                    spID = splist[1]
                    # check if the current span(sp) is the source of a relation
                    ssp = None
                    tsp = None
                    rn = None
                    if relations:
                        source = []
                        target = []
                        rLabel = []
                        for (sourceSP, sourceID, targetSP, targetID, relname) in relations:
                            if spID == sourceID:
                                source.append([sourceSP, sourceID])
                                target.append([targetSP, targetID])
                                rLabel.append(relname)
                                ssp = source
                                tsp = target
                                rn = rLabel

                    #get the attributes;
                    # For current ehost reader all the attributes are included in each span with difference value
                    has_attr = False
                    for attr in attribute_list:
                        #exec('attr_val=sp._.'+attr) #cannot process attribute like KPS_1-100 witll treat 100 as -100
                        attr_val = getattr(sp._,attr,None) #more reliable than above line
                        if not (attr_val==None):
                            has_attr = True

                            # append both relation and attributes
                            if rn:
                                for s,t,r in zip(ssp,tsp,rn):
                                    doc_names.append((doc._.doc_name).replace('.txt',''))
                                    span_texts.append(sp.text)
                                    span_label.append(sp.label_)
                                    span_start.append(sp.start_char)
                                    span_end.append(sp.end_char)
                                    span_attr_name.append(attr.replace('ANNOT_',''))
                                    span_attr_value.append(attr_val)
                                    span_id.append(s[1])
                                    relSpan_id.append(t[1])
                                    relLabel.append(r)
                            else:
                                doc_names.append((doc._.doc_name).replace('.txt',''))
                                span_texts.append(sp.text)
                                span_label.append(sp.label_)
                                span_start.append(sp.start_char)
                                span_end.append(sp.end_char)
                                span_attr_name.append(attr.replace('ANNOT_',''))
                                span_attr_value.append(attr_val)
                                span_id.append(sp._.annotation_id)
                                relSpan_id.append("NULL")
                                relLabel.append("NULL")

                    if has_attr == False:
                        if rn:
                            for s,t,r in zip(ssp,tsp,rn):
                                doc_names.append((doc._.doc_name).replace('.txt',''))
                                span_texts.append(sp.text)
                                span_label.append(sp.label_)
                                span_start.append(sp.start_char)
                                span_end.append(sp.end_char)
                                span_attr_name.append("NULL")
                                span_attr_value.append("NULL")
                                span_id.append(s[1])
                                relSpan_id.append(t[1])
                                relLabel.append(r)
                        else:
                            doc_names.append((doc._.doc_name).replace('.txt',''))
                            span_texts.append(sp.text)
                            span_label.append(sp.label_)
                            span_start.append(sp.start_char)
                            span_end.append(sp.end_char)
                            span_attr_name.append("NULL")
                            span_attr_value.append("NULL")
                            span_id.append(sp._.annotation_id)
                            relSpan_id.append("NULL")
                            relLabel.append("NULL")

        d = {
            "DocID": doc_names,
            "annotatedSpan": span_texts, 
            "spanStartChar": span_start,
            "spanEndChar": span_end,
            "spanLabel": span_label,
            "attrName": span_attr_name,
            "attrVal": span_attr_value,
            "spanID": span_id,
            "relID": relSpan_id,
            "relLabel": relLabel
         }

        return(pd.DataFrame(data=d, ))
          
        
    def add_spacyDocs(self,docs1,name=None,id_list=[], labels=1, ent_or_spangroup = 'ent',attributes=[]):
        '''Establishes type of input and creates corresponding dataframe for entities.

        Arguments:
            docs1: Either a spacy document, tuple/list of entities/spans, or spangroup.
            ent_or_spangroup: String of either 'ent' or 'spangroup'. 'ent' indicates to compare doc.ents between documents. 'spangroup' 
                indicates to compare doc1's only spangroup (note that doc1 must have only 1 spangroup) with doc2's equivalently named spangroup.
                This argument is only relevant if passing in spacy document (ie. can be ignored if passing in tuple/list of ents/spans)
                attributes: list of attributes to be extracted from ents/spans

        Return:
            Dataframe of entities for document lists
        '''
        try:
            import spacy
        except:
            print("Error: Could not import spacy")
        if (isinstance(docs1[0],tuple) or isinstance(docs1[0],list) or isinstance(docs1[0],spacy.tokens.span_group.SpanGroup)):
            ent_type = 1
        elif (type(docs1[0]) is spacy.tokens.doc.Doc):
            if ent_or_spangroup == 'ent':
                ent_type = 2
            elif ent_or_spangroup == 'spangroup':
                ent_type = 3
            else:
                #raise error
                print("Error: Must select 'spangroup' or 'ent' for ent_or_spangroup option.")
                return
        else:
            #raise error
            print("Error: Input must contain list with type 'tuples','list', 'spacy.tokens.span_group.SpanGroup', or 'spacy.tokens.doc.Doc'")
            return

        df_doc_name = 'DocID'
        df_span_text = 'annotatedSpan'
        df_start_char = 'spanStartChar'
        df_end_char = 'spanEndChar'
        df_concept_label = 'spanLabel'
        
        ent_dict_1 = {df_doc_name:[],df_span_text:[],df_start_char:[],df_end_char:[],df_concept_label:[]}
        
        if name == None:
            name = 'spacyDoc_set1'
            num = 1
            while name in (self.annot.keys()):
                num+=1
                name = 'spacyDoc_set' + str(num)
                
        self.spacy_docs[name] = dict()

        for attr in attributes:
            ent_dict_1[attr] = []
        attr_update = set(self.attr)
        attr_update.update(attributes)
        self.attr = list(attr_update)

        if ent_type == 1:
            for index,document in enumerate(docs1):
                for ents in document:
                    try:
                        ent_dict_1[df_doc_name].append(str(id_list[index]))
                    except:
                        ent_dict_1[df_doc_name].append(str(index))
                    ent_dict_1[df_span_text].append(ents.text)
                    ent_dict_1[df_start_char].append(ents.start_char)
                    ent_dict_1[df_end_char].append(ents.end_char)
                    for attr in attributes:
                        try:
                            val = getattr(ents, attr)
                        except AttributeError:
                            val = getattr(ents._, attr)
                        ent_dict_1[attr].append(val)
                    if labels == 1:
                        ent_dict_1[df_concept_label].append(ents.label_)
                    else:
                        ent_dict_1[df_concept_label].append("")
                try:
                    self.spacy_docs[name][str(id_list[index])] = document
                except:
                    self.spacy_docs[name][str(index)] = document
        if ent_type == 2:
            for index,document in enumerate(docs1):
                for ents in document.ents:
                    try:
                        ent_dict_1[df_doc_name].append(str(id_list[index]))
                    except:
                        ent_dict_1[df_doc_name].append(str(index))
                    ent_dict_1[df_span_text].append(ents.text)
                    ent_dict_1[df_start_char].append(ents.start_char)
                    ent_dict_1[df_end_char].append(ents.end_char)
                    for attr in attributes:
                        try:
                            val = getattr(ents, attr)
                        except AttributeError:
                            val = getattr(ents._, attr)
                        ent_dict_1[attr].append(val)
                    if labels == 1:
                        ent_dict_1[df_concept_label].append(ents.label_)
                    else:
                        ent_dict_1[df_concept_label].append("")
                try:
                    self.text[str(id_list[index])] = document.text
                    self.spacy_docs[name][str(id_list[index])] = document
                except:
                    self.text[str(index)] = document.text
                    self.spacy_docs[name][str(index)] = document
        if ent_type == 3:
            ent_dict_1['Span Group key'] = []
            for index,document in enumerate(docs1):
                for span_key in list(document.spans.keys()):
                    for span in document.spans[span_key]:
                        try:
                            ent_dict_1[df_doc_name].append(str(id_list[index]))
                        except:
                            ent_dict_1[df_doc_name].append(str(index))
                        ent_dict_1[df_span_text].append(span.text)
                        ent_dict_1[df_start_char].append(span.start_char)
                        ent_dict_1[df_end_char].append(span.end_char)
                        for attr in attributes:
                            try:
                                val = getattr(ents, attr)
                            except AttributeError:
                                val = getattr(ents._, attr)
                            ent_dict_1[attr].append(val)
                        if labels == 1:
                            ent_dict_1[df_concept_label].append(span.label_)
                        else:
                            ent_dict_1[df_concept_label].append("")
                        ent_dict_1['Span Group key'].append(span_key)
                try:
                    self.text[str(id_list[index])] = document.text
                    self.spacy_docs[name][str(id_list[index])] = document
                except:
                    self.text[str(index)] = document.text
                    self.spacy_docs[name][str(index)] = document
        
        self.annot[name] = pd.DataFrame.from_dict(ent_dict_1)
        self.doc_col = 'DocID'
        self.span_col = 'annotatedSpan'
        self.label_col = 'spanLabel'
        self.span_start_col = 'spanStartChar'
        self.span_end_col = 'spanEndChar'
    
    def add_dataframe(self,df_dict):
        for key in df_dict.keys():
            self.annot[key] = df_dict[key]
            
    def remove_annotation_set(self,name):
        if not isinstance(name,list):
            name = [name]
        for n in name:
            try:
                del self.annot[n]
                print('Removed dataframe: ',n)
            except:
                pass
            try:
                del self.spacy_docs[n]
                print('Removed set of spacy documents: ',n)
            except:
                pass
        
                                                             
    def overlaps(self,df1, df2, labels:bool,match_on_attr:bool):
        '''Calculates overlapping entities between two dataframes, each containing entity information for a single document.
            Also checks for matching labels if label=1.
            Works the same as "overlaps" function, but uses dataframes instead of documents.

        Return:
            Dictionaries with the mapping of matching entity indices:
                keys: entity index from one dataframe
                value: matched entity index from other dataframe

                Ex: "{1 : [2] , 3 : [4,5]}" means that the entity at index 1 from df1 matches the entity indexed at 1 in df2, and 
                entity indexed at 3 in df1 matches entity indexed at 4 and 5 from df2.
        '''

        df1_matches = dict()
        df2_matches = dict()

        tree = IntervalTree()
        for index2,row2 in df2.iterrows():
            tree.add(row2[self.span_start_col],row2[self.span_end_col],index2)

        for index1,row1 in df1.iterrows():
            matches = tree.search(row1[self.span_start_col],row1[self.span_end_col])
            for match in matches:
                index2 = match.data #match.data is the index of doc2_ents
                attributes_match = 1
                if match_on_attr:
                    for attribute in self.attr: #check if attributes match
                        if row1[attribute] != df2.loc[index2,attribute]:
                            attributes_match = 0
                            break
                if ((labels == 0) | (df2.loc[index2,self.label_col] == row1[self.label_col])) & (attributes_match):
                    if index1 not in df1_matches.keys():
                        df1_matches[index1] = [index2]
                    else:
                        df1_matches[index1].append(index2)
                    if index2 not in df2_matches.keys():
                        df2_matches[index2] = [index1]
                    else:
                        df2_matches[index2].append(index1)

        return df1_matches, df2_matches
                                                             
                                                             
    def exact_match(self,df1, df2, labels:bool,match_on_attr:bool):
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

        df1_matches = dict()
        df2_matches = dict()

        df1_ent_dict = dict()
        df2_ent_dict = dict()

        for index1,row1 in df1.iterrows():
            key_array = []
            key_array.extend([row1[self.span_start_col],row1[self.span_end_col],row1[self.label_col]])
            if match_on_attr:
                for attribute in self.attr:
                    key_array.append(row1[attribute])
            if labels == 1: #If checking for labels, then include this in the tuple's to-be-compared elements
                key_array.append(row1[self.label_col])
            df1_ent_dict[tuple(key_array)] = index1

        for index2,row2 in df2.iterrows():
            key_array = []
            key_array.extend([row2[self.span_start_col],row2[self.span_end_col],row2[self.label_col]])
            if match_on_attributes:
                for attribute in self.attr:
                    key_array.append(row2[attribute])
            if labels == 1: #If checking for labels, then include this in the tuple's to-be-compared elements
                key_array.append(row2[self.label_col])
            df2_ent_dict[tuple(key_array)] = index2

        df1_ent_set = set(doc1_ent_dict.keys())
        df2_ent_set = set(doc2_ent_dict.keys())

        matched_ents = df1_ent_set.intersection(df2_ent_set)

        for match in matched_ents:
            index1 = df1_ent_dict[match]
            index2 = df2_ent_dict[match]
            df1_matches[index1] = [index2]
            df2_matches[index2] = [index1]

        return df1_matches, df2_matches
                                                             
                                                            
    def generate_report(self):
        #'span_metrics''token_level_metrics''label_metrics''overall_label_metrics''attr_metrics''overall_attr_metrics''attr_metrics_hier''overall_attr_metrics_hier'
        if len(self.agreement_metrics.keys()) == 0:
            self.calculate_agreement()
        import matplotlib.pyplot as plt
        font = {'family':'normal','weight':'bold','size':50}
        plt.rcParams.update({'font.size':50})#'ytick.labelsize'
        body = ""
        
        def round_float(value):
            if isinstance(value,float):
                return round(value,self.OUTPUT_DECIMAL_PLACES)
            return value
        
        for key in self.match_dict.keys(): #these are also keys for the agreement_metrics
            title_text = key
            df_export = self.match_dict[key]
            df_export['Annotation_1'] = df_export['Annotation_1'].str.replace('\n',' ')
            df_export['Annotation_2'] = df_export['Annotation_2'].str.replace('\n',' ')
            if self.text != None:
                df_export['context'] = df_export['context'].str.replace('\n',' ')
                df_export['context'] = df_export['context'].str.replace('\t',' ')
            df_export['Annotation_1'] = df_export['Annotation_1'].str.replace('\t',' ')
            df_export['Annotation_2'] = df_export['Annotation_2'].str.replace('\t',' ')
            df_export.to_csv(key + ".csv",index=False)
            agreement_dict = self.agreement_metrics[key]
            
            plt.close()
            annot2_mismatch = self.match_dict[key][self.match_dict[key]['Annotation_1'] == ""]['Annotation_2'].str.lower().value_counts()
            annot2_mismatch.plot(kind='barh',figsize=(50,max(10,int(annot2_mismatch.shape[0]*1.3))),xticks=range(annot2_mismatch.max()+1))
            plt.tight_layout()
            plt.savefig(key + 'Annotator2_mismatches.png')
            
            plt.close()
            annot1_mismatch = self.match_dict[key][self.match_dict[key]['Annotation_2'] == ""]['Annotation_1'].str.lower().value_counts()
            annot1_mismatch.plot(kind='barh',figsize=(50,max(10,int(annot1_mismatch.shape[0]*1.3))),xticks=range(annot1_mismatch.max()+1))
            plt.tight_layout()
            plt.savefig(key + 'Annotator1_mismatches.png')
            
            plt.close()
            plt.figure(figsize=(15,8))
            pi_names = list(self.label_count_dict.keys())
            pi_quantities = [round(float(x)/sum(list(self.label_count_dict.values()))*100,2) for x in list(self.label_count_dict.values())]
            plt.pie(pi_quantities,labels=None,autopct='',startangle=140, textprops={'fontsize':10},pctdistance=1.3)
            legend_labels = [f'{name}, ({quantity}%)' for name,quantity in zip(pi_names,pi_quantities)]
            plt.legend(legend_labels,loc='best',fontsize=18,bbox_to_anchor=(.95,.9))#,title='Label (token density)'
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(key + '_pi.png')
            plt.close()
            
            body += f'''
            <h1>{title_text}</h1>
            <h2>Span Matching</h2>
            <p>Explanation here</p>
            {agreement_dict['span_metrics'].applymap(round_float).to_html()}
            <h2>Token level Matching</h2>
            <p>Explanation here</p>
            {agreement_dict['token_level_metrics'].applymap(round_float).to_html()}
            <h2>Token-level Density of annotations</h2>
            <img src={key + '_pi.png'} width="700">
            <h2>Label Matching</h2>
            <p>Explanation here</p>
            {agreement_dict['label_metrics'].applymap(round_float).to_html()}
            {agreement_dict['overall_label_metrics'].applymap(round_float).to_html()}
            <h2>Attribute Matching (Label Dependent)</h2>
            <p>Explanation here</p>
            {agreement_dict['attr_metrics_hier'].applymap(round_float).to_html()}
            {agreement_dict['overall_attr_metrics_hier'].applymap(round_float).to_html()}
            <h2>Attribute Matching (Label Independent)</h2>
            <p>Explanation here</p>
            {agreement_dict['attr_metrics'].applymap(round_float).to_html()}
            {agreement_dict['overall_attr_metrics'].applymap(round_float).to_html()}
            <h2>Relationship Matching</h2>
            '''
            if self.rel_source_col != []:
                body += f'''
                <p>Explanation here</p>
                {agreement_dict['rel_metrics'].applymap(round_float).to_html()}
                {agreement_dict['overall_rel_metrics'].applymap(round_float).to_html()}
                '''
            body += f'''
            <h2>Mismatched spans</h2>
            <h3>Annotator 1 Misses</h3>
            <img src={key + 'Annotator2_mismatches.png'} width="1000">
            <h3>Annotator 2 Misses</h3>
            <img src={key + 'Annotator1_mismatches.png'} width="1000">
            '''
        
        page_title = "Annotation Report"

        html = f'''
            <html>
                <head>
                    <title>{page_title}</title>
                </head>
                <body>''' + body + f'''
                </body>
            </html>
            '''
        with open('html_report.html','w') as f:
            f.write(html)          
            
            
    def corpus_agreement(self,df1,df2,loose:bool,labels:bool,match_on_attr:bool):
        '''Calculates f1 over an entire corpus of documents.

        Arguments:
            df1: Either a list of spacy documents, list containing inner tuples/lists of entities/spans, list of spangroups, or a dataframe. 
                Considered the golden/correct annotation for fp,fn.
            df2: Either a list of spacy documents, list of tuples/lists of entities/spans, list of spangroups, or a dataframe. 
            loose: Boolean. 1 indicates to consider any overlap. 0 indicates to only consider exact matches.
            labels: Boolean. 1 indicates to consider labels as matching criteria.
            ent_or_spangroup: String of either 'ent' or 'spangroup'. 'ent' indicates to compare doc.ents between documents. 'spangroup' 
            indicates to extract doc1's and doc2's spangroups, but will not be compared. This argument is only relevant if passing in a list of
            spacy documents (ie. can be ignored if passing in a list of tuple/list of ents/spans/spangroups or dataframe)

        Returns:
            Tuple of two items:
            First item: Simple Dataframe containing IAA, Recall, Precision, true positive count, false positive count, and false negative count.
            Second item: Dataframe containing all entities with information on entities and matches
        '''

        corpus_tp, corpus_fp, corpus_fn = (0,0,0)
        """
        df_columns=["doc name","Annotation_1","Annotation_2", "Annot_1_label", "Annot_1_char", "Annot_2_label", "Annot_2_char", "Overall_start_char", "Exact Match?", "Duplicate Matches?", "Overlap?","Matching_label?"]
        for attr in self.attr:
            df_columns.append("A1_" + attr)
            df_columns.append("A2_" + attr)
        agreement_df = pd.DataFrame(columns=df_columns)
        """
        #m_df_columns=[self.doc_col,self.span_col,self.span_start_col,self.span_end_col,self.label_col,"Match"]
        #for attr in self.attr:
        #    m_df_columns.append(attr)
        #complete_merge_df = pd.DataFrame(columns=m_df_columns)

        unique_doc_names = list(set(df1[self.doc_col].unique().tolist() + df2[self.doc_col].unique().tolist()))
        for doc_name in unique_doc_names:
            docs1_df = df1[df1[self.doc_col] == doc_name]
            docs2_df = df2[df2[self.doc_col] == doc_name]
            if loose==1:
                doc1_matches,doc2_matches = overlaps(docs1_df,docs2_df,labels,match_on_attr)
            else:
                doc1_matches,doc2_matches = exact_match(docs1_df,docs2_df,labels,match_on_attr)
            tp,fp,fn = conf_matrix(doc1_matches,doc2_matches,docs1_df.shape[0],docs2_df.shape[0])
            corpus_tp += tp
            corpus_fp += fp
            corpus_fn += fn
            #new_agreement_df = create_agreement_df(doc1_matches,doc2_matches,docs1_df,docs2_df)
            #new_agreement_df.index += agreement_df.shape[0]
            #agreement_df = pd.concat([agreement_df,new_agreement_df])
            #new_merge_df = merge_df(doc1_matches,doc2_matches,docs1_df,docs2_df)
            #new_merge_df.index += complete_merge_df.shape[0]
            #complete_merge_df = pd.concat([complete_merge_df,new_merge_df])

        if corpus_tp == 0 and corpus_fp == 0:
            #print("Both tp and fp count was 0. Not showing recall, precision, or f1.")
            data = {'IAA' : "Und", 'Recall' : "Und", 'Precision' : "Und",'True Positives' : [corpus_tp] , 'False Positives' : [corpus_fp], 'False Negative' : [corpus_fn]}  
        else:
            data = {'IAA' : [pairwise_f1(corpus_tp,corpus_fp,corpus_fn)], 'Recall' : [corpus_tp/float(corpus_tp+corpus_fn)], 'Precision' : [corpus_tp/float(corpus_tp+corpus_fp)],'True Positives' : [corpus_tp] , 'False Positives' : [corpus_fp], 'False Negative' : [corpus_fn]}

        return (pd.DataFrame(data))#,agreement_df)#,complete_merge_df)


    def pairwise_f1(self,tp,fp,fn):
        '''Calculates f1 given true positive, false positive, and false negative values

        Returns:
            pairwise_f1/IAA float value'''
        if tp+fn+fp == 0:
            return 'Und'
        else:
            return (2*tp)/float(2*tp+fp+fn)
    
    def recall(self,tp,fp,fn):
        if tp+fn == 0:
            return 'Und'
        else:
            return (tp/float(tp+fn))
        
    def precision(self,tp,fp,fn):
        if tp+fp == 0:
            return 'Und'
        else:
            return (tp/float(tp+fp))
        
    def cohens_kappa(self,df):
        try:
            import medspacy
            import spacy
            from medspacy.custom_tokenizer import create_medspacy_tokenizer
        except:
            print("Error: could not import mespacy.custom_tokenizer, medspacy, and/or spacy for cohen's kappa")
            return ["-" for i in range(6)]
        
        #build tokenizer
        nlp = spacy.blank("en")
        medspacy_tokenizer = create_medspacy_tokenizer(nlp)
        default_tokenizer = nlp.tokenizer
        
        label_count_dict = dict()
        label_count_dict['No Annotation'] = 0
        
        tp,fp,fn,tn = (0,0,0,0)
        for x in set(df['doc_name'].unique().tolist()):
            tree1 = IntervalTree()
            for index1,row1 in df[(df['doc_name'] == x) & (df['Annot_1_char'] != "")].iterrows():
                start_loc = [int(t.split("-")[0]) for t in row1["Annot_1_char"].split("||")]
                end_loc = [int(t.split("-")[1]) for t in row1["Annot_1_char"].split("||")]
                for s,e in zip(start_loc,end_loc):
                    tree1.add(s,e,index1)
            tree2 = IntervalTree()
            for index2,row2 in df[(df['doc_name'] == x) & (df['Annot_2_char'] != "")].iterrows():
                start_loc = [int(t.split("-")[0]) for t in row2["Annot_2_char"].split("||")]
                end_loc = [int(t.split("-")[1]) for t in row2["Annot_2_char"].split("||")]
                for s,e in zip(start_loc,end_loc):
                    tree2.add(s,e,index2)
            ents = [f for f in medspacy_tokenizer(self.text[x])]
            for ent in ents:
                matches1 = tree1.search(ent.idx,ent.idx + len(ent.text) - 1)
                matches2 = tree2.search(ent.idx,ent.idx + len(ent.text) - 1)
                if len(matches1) > 0:
                    if len(matches2) > 0:
                        tp+=1
                    else:
                        fn+=1
                else:
                    if len(matches2) > 0:
                        fp+=1
                    else:
                        label_count_dict['No Annotation'] += 1
                        tn+=1
                
                mat_label_set = set()
                for mat in matches1:
                    for lab in df.loc[mat.data,"Annot_1_label"].split(" || "):
                        if lab != None:
                            mat_label_set.add(lab)
                for mat in matches2:
                    for lab in df.loc[mat.data,"Annot_2_label"].split(" || "):
                        if lab != None:
                            mat_label_set.add(lab)
                for lab in mat_label_set:
                    if lab not in label_count_dict.keys():
                        label_count_dict[lab] = 1
                    else:
                        label_count_dict[lab] += 1
                self.label_count_dict = label_count_dict
        
        
                        
                        
        return [tp,fp,fn,tn,self.pairwise_f1(tp,fp,fn),((2 * (float(tp * tn) - float(fn * fp))) / float(float(tp + fp) * float(fp + tn) + float(tp + fn) * float(fn + tn)))] #wikipedia binary classification confusion matrix for Cohen's kappa
                  
                    
                  
                  
        #count entities that have a start or end span within their span from either annotation
        #tn = len(doc.ents) - count
        #return kappa using binary classification

                                                             
    def conf_matrix(self,doc1_matches,doc2_matches,doc1_ent_num,doc2_ent_num):
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
    
    
    def extract_context(self,row):
        document_id = row["doc_name"]
        if (row["Annot_1_char"] != "") & (row["Annot_2_char"] != ""):
            start_pos = min([int(t.split("-")[0]) for t in row["Annot_1_char"].split("||")] + [int(t.split("-")[0]) for t in row["Annot_2_char"].split("||")])
            end_pos = max([int(t.split("-")[1]) for t in row["Annot_1_char"].split("||")] + [int(t.split("-")[1]) for t in row["Annot_2_char"].split("||")])
        elif (row["Annot_1_char"] != "") & (row["Annot_2_char"] == ""):
            start_pos = min([int(t.split("-")[0]) for t in row["Annot_1_char"].split("||")])
            end_pos = max([int(t.split("-")[1]) for t in row["Annot_1_char"].split("||")])
        else:
            start_pos = min([int(t.split("-")[0]) for t in row["Annot_2_char"].split("||")])
            end_pos = max([int(t.split("-")[1]) for t in row["Annot_2_char"].split("||")])
        
        text = self.text.get(document_id,"")
        return "..." + text[max(0,int(start_pos)-AnnotationAggregator.CONTEXT_CHAR_AMOUNT):min(int(end_pos)+AnnotationAggregator.CONTEXT_CHAR_AMOUNT, len(text))] + "..."
    
    
    def create_agreement_df_corpus(self,df1,df2):
        df_columns=["doc_name","Annotation_1","Annotation_2", "Annot_1_label", "Annot_1_char", "Annot_2_label", "Annot_2_char", "Overall_start_char", "Exact_Match?", "Duplicate_Matches?", "Overlap?","Matching_label?"]
        for attr in (self.attr + self.rel_source_col + self.rel_target_col):
            df_columns.append("A1_" + attr)
            df_columns.append("A2_" + attr)
        agreement_df = pd.DataFrame(columns=df_columns)

        unique_doc_names = list(set(df1[self.doc_col].unique().tolist() + df2[self.doc_col].unique().tolist()))
        for doc_name in unique_doc_names:
            docs1_df = df1[df1[self.doc_col] == doc_name]
            docs2_df = df2[df2[self.doc_col] == doc_name]
            doc1_matches,doc2_matches = self.overlaps(docs1_df,docs2_df,labels=0,match_on_attr=0)

            new_agreement_df = self.create_agreement_df(doc1_matches,doc2_matches,docs1_df,docs2_df)
            new_agreement_df.index += agreement_df.shape[0]
            agreement_df = pd.concat([agreement_df,new_agreement_df])
            
        if self.text != None:
            agreement_df['context'] = agreement_df.apply(self.extract_context,axis=1)
            
        for attr in self.attr:
            agreement_df[attr + "_Match?"] = agreement_df.apply(lambda x: set(x["A1_" + attr].split(" || ")) == set(x["A2_" + attr].split(" || ")),axis=1)
        
        #Fix label matching
        agreement_df["Matching_label?"] = agreement_df.apply(lambda x: set(x["Annot_1_label"].split(" || ")) == set(x["Annot_2_label"].split(" || ")),axis=1) #This overrides what was there
        
        #Fix the relationship ID's to be interoperable between both annotations
        null_values = set(AnnotationAggregator.NULL_RELATIONSHIP_VALUES)
        if self.rel_source_col != []:
            for rel in self.rel_source_col:
                rel_replace_col1 = []
                rel_replace_col2 = []
                for index,row in agreement_df.iterrows():
                    for rel_s in row['A1_' + rel].split(" || "):
                        if rel_s not in null_values:
                            newID = agreement_df[agreement_df['A1_' + self.rel_target_col[0]].apply(lambda x: rel_s in x.split(' || '))][['A1_' + self.rel_target_col[0],'A2_' + self.rel_target_col[0]]]
                            if newID.shape[0] > 0:
                                newID = newID.iloc[0,:].tolist()
                                rel_replace_col1.append("_".join([str(x) for d in newID for x in d.split(' || ')]))
                                break
                            else:
                                print("NewID No Match A1: ",row[['doc_name','Annotation_1']])
                                rel_replace_col1.append(str(np.nan))
                                break
                        else:
                            rel_replace_col1.append(str(np.nan))
                            break
                    
                    for rel_s in row['A2_' + rel].split(" || "):
                        if rel_s not in null_values:
                            newID = agreement_df[agreement_df['A2_' + self.rel_target_col[0]].apply(lambda x: rel_s in x.split(' || '))][['A1_' + self.rel_target_col[0],'A2_' + self.rel_target_col[0]]]
                            if newID.shape[0] > 0:
                                newID = newID.iloc[0,:].tolist()
                                rel_replace_col2.append("_".join([str(x) for d in newID for x in d.split(' || ')]))
                                break
                            else:
                                print("NewID No Match A2: ",row[['doc_name','Annotation_2']])
                                rel_replace_col2.append(str(np.nan))
                                break
                        else:
                            rel_replace_col2.append(str(np.nan))
                            break
                agreement_df['A1_' + rel] = rel_replace_col1
                agreement_df['A2_' + rel] = rel_replace_col2
                        
                        
                        
        
        #Add relationship matching
        def rel_match_check(row,rel): #Add attr agreement logic here, since you may have duplicates with multiple labels
            rel1_set = set(row["A1_" + rel].split(" || "))
            rel2_set = set(row["A2_" + rel].split(" || "))
            null_values = set(AnnotationAggregator.NULL_RELATIONSHIP_VALUES)
            if (rel1_set == rel2_set) and not (rel1_set - null_values):
                return "TN"
            elif (rel1_set == rel2_set) and (rel1_set - null_values):
                return "TP"
            elif not (rel2_set - null_values) and (rel1_set - null_values):
                return "FN"
            else:
                return "FP"
            
        if self.rel_source_col != []:
            for rel in self.rel_source_col:
                agreement_df[rel + "_Match?"] = agreement_df.apply(lambda x: rel_match_check(x,rel),axis=1)
            
        return agreement_df
    
    def earliest_match(self,start_key,dict_1,dict_2,visited_1=None,visited_2=None):
            if visited_1 is None:
                visited_1 = set()
            if visited_2 is None:
                visited_2 = set()

            if start_key in visited_1:
                return None

            visited_1.add(start_key)

            if start_key in dict_1:
                for value in sorted(dict_1[start_key]):
                    if value in dict_2:
                        for next_key in sorted(dict_2[value]):
                            if next_key in visited_2:
                                continue
                            result = self.earliest_match(next_key, dict_1, dict_2, visited_1,visited_2)
                            if result is not None:
                                visited_1.add(result)
                                return min({x for x in visited_1 if x is not None})
            return start_key
        
                                                             
    def create_agreement_df(self,doc1_matches,doc2_matches,doc1_ents,doc2_ents):

        if self.label_col in list(doc1_ents.columns) and self.label_col in list(doc2_ents.columns):
            label=1
        else:
            label=0
        result_dict = {"doc_name" : [],"Annotation_1" : [],"Annotation_2" : [], "Annot_1_label" : [], "Annot_1_char" : [], "Annot_2_label" : [], "Annot_2_char" : [], "Overall_start_char" : [], "Exact_Match?" : [], "Duplicate_Matches?" : [], "Overlap?" : [],"Matching_label?": [], "Index" : []}
        for attr in (self.attr + self.rel_source_col + self.rel_target_col):
            result_dict["A1_" + attr] = []
            result_dict["A2_" + attr] = []
        if len(doc1_ents[self.doc_col].tolist()) > 0:
            doc_name = doc1_ents[self.doc_col].tolist()[0]
        elif len(doc2_ents[self.doc_col].tolist()) > 0:
            doc_name = doc2_ents[self.doc_col].tolist()[0]
        for index1 in doc1_ents.index: #iterate through all ents inset one
            if index1 in doc1_matches.keys(): #Cases where the ent has a match
                #if another index1 is in doc2_matches.values(), then add it to this row
                first_index2 = sorted(doc1_matches[index1])[0]
                first_index1 = sorted(doc2_matches[first_index2])[0]
                if first_index1 < index1: #Cases where the entity in doc2 already matched a previous doc1 ent (ie. duplicate match)
                    #Add to index: sorted(doc2_matches[first_index2])[0]
                    try:
                        duplicate_match_index = result_dict["Index"].index(first_index1)
                        result_dict["Annotation_1"][duplicate_match_index] += " || " + doc1_ents.loc[index1,self.span_col]
                    except:
                        early_match = self.earliest_match(first_index1,doc1_matches,doc2_matches)
                        duplicate_match_index = result_dict["Index"].index(early_match)
                        result_dict["Annotation_1"][duplicate_match_index] += " || " + doc1_ents.loc[index1,self.span_col]

                    
                    #duplicate_match_index = result_dict["Index"].index(first_index1)
                    
                    result_dict["Duplicate_Matches?"][duplicate_match_index] = True
                    result_dict["Overlap?"][duplicate_match_index] = True
                    result_dict["Annot_1_char"][duplicate_match_index] += " || " + str(doc1_ents.loc[index1,self.span_start_col]) + "-" + str(doc1_ents.loc[index1,self.span_end_col])
                    if result_dict["Overall_start_char"][duplicate_match_index] > int(doc1_ents.loc[index1,self.span_start_col]):
                        result_dict["Overall_start_char"][duplicate_match_index] = doc1_ents.loc[index1,self.span_start_col]
                    if label==1:
                        if result_dict["Matching_label?"][duplicate_match_index] == 0:
                            result_dict["Matching_label?"][duplicate_match_index] = (doc1_ents.loc[index1,self.label_col] in result_dict["Annot_2_label"][duplicate_match_index].split(" || "))
                        try:
                            result_dict["Annot_1_label"][duplicate_match_index] += " || " + doc1_ents.loc[index1,self.label_col]
                        except TypeError:
                            if type(doc1_ents.loc[index1,self.label_col]) is not str and type(result_dict["Annot_1_label"][duplicate_match_index]) is not str:
                                result_dict["Annot_1_label"][duplicate_match_index] = ""
                            elif type(doc1_ents.loc[index1,self.label_col]) is not str:
                                result_dict["Annot_1_label"][duplicate_match_index] += " || " + ' '
                            elif type(result_dict["Annot_1_label"][duplicate_match_index]) is not str:
                                result_dict["Annot_1_label"][duplicate_match_index] = " " + " || " + doc1_ents.loc[index1,self.label_col]
                    for attr in (self.attr + self.rel_source_col + self.rel_target_col):
                        result_dict["A1_" + attr][duplicate_match_index] += " || " + str(doc1_ents.loc[index1,attr])

                    #take the index2's from doc1_matches[index1] that don't match doc1_matches[first_index1] (or previous index1) and add those


                else:
                    result_dict["doc_name"].append(doc_name)
                    result_dict["Index"].append(index1)
                    #result_dict["Match"].append("TP")
                    result_dict["Annotation_1"].append(doc1_ents.loc[index1,self.span_col])
                    annot_2 = ""
                    annot_2_label = ""
                    annot_2_char = ""
                    overall_start_char = int(doc1_ents.loc[index1,self.span_start_col])
                    first_time=1
                    annot_2_start_char = -1
                    annot_2_end_char = -1
                    annot_1_start_char = int(doc1_ents.loc[index1,self.span_start_col])
                    annot_1_end_char = int(doc1_ents.loc[index1,self.span_end_col])
                    exact_match = False
                    annot_2_attr_dict = dict()
                    for index2 in sorted(doc1_matches[index1]):
                        annot_2_start_char = int(doc2_ents.loc[index2,self.span_start_col])
                        annot_2_end_char = int(doc2_ents.loc[index2,self.span_end_col])
                        if (annot_1_start_char == annot_2_start_char) and (annot_1_end_char == annot_2_end_char):
                            exact_match = True
                        if annot_2_start_char < overall_start_char:
                            overall_start_char = annot_2_start_char
                        if first_time ==1:
                            first_time=0
                            annot_2_char = str(annot_2_start_char) + "-" + str(annot_2_end_char)
                            annot_2 += doc2_ents.loc[index2,self.span_col]
                            if label==1:
                                if type(doc2_ents.loc[index2,self.label_col]) is str:
                                    annot_2_label = doc2_ents.loc[index2,self.label_col]
                            for attr in (self.attr + self.rel_source_col + self.rel_target_col):
                                annot_2_attr_dict[attr] = str(doc2_ents.loc[index2,attr])
                        else:
                            if label==1:
                                try:
                                    annot_2_label += " || " + doc2_ents.loc[index2,self.label_col]
                                except TypeError:
                                    annot_2_label += " ||  "
                            annot_2 += " || " + doc2_ents.loc[index2,self.span_col]
                            annot_2_char += " || " + str(annot_2_start_char) + "-" + str(annot_2_end_char)
                            for attr in (self.attr + self.rel_source_col + self.rel_target_col):
                                try:
                                    annot_2_attr_dict[attr] += " || " + str(doc2_ents.loc[index2,attr])
                                except TypeError:
                                    annot_2_attr_dict[attr] += " || "
                    result_dict["Annotation_2"].append(annot_2)
                    result_dict["Annot_2_label"].append(annot_2_label)
                    result_dict["Matching_label?"].append(doc1_ents.loc[index1,self.label_col] in annot_2_label.split(' || '))
                    result_dict["Annot_2_char"].append(annot_2_char)
                    if label == 1:
                        result_dict["Annot_1_label"].append(doc1_ents.loc[index1,self.label_col])
                    else:
                        result_dict["Annot_1_label"].append("")
                    result_dict["Annot_1_char"].append(str(annot_1_start_char) + "-" + str(annot_1_end_char))
                    result_dict["Overall_start_char"].append(overall_start_char)
                    result_dict["Exact_Match?"].append(exact_match)
                    if len(doc1_matches[index1]) > 1:
                        result_dict["Duplicate_Matches?"].append(True)
                    else:
                        result_dict["Duplicate_Matches?"].append(False)
                    result_dict["Overlap?"].append(True)
                    for attr in (self.attr + self.rel_source_col + self.rel_target_col):
                        result_dict["A1_" + attr].append(str(doc1_ents.loc[index1,attr]))
                        result_dict["A2_" + attr].append(annot_2_attr_dict[attr])
            else: #Cases where an ent in doc1 doesn't have a match
                result_dict["doc_name"].append(doc_name)
                result_dict["Index"].append(index1)
                #result_dict["Match"].append("FN")
                result_dict["Annotation_1"].append(doc1_ents.loc[index1,self.span_col])
                result_dict["Annotation_2"].append("")
                if label == 1:
                    result_dict["Annot_1_label"].append(doc1_ents.loc[index1,self.label_col])
                    result_dict["Annot_2_label"].append("")
                else:
                    result_dict["Annot_1_label"].append("")
                    result_dict["Annot_2_label"].append("")
                result_dict["Matching_label?"].append(False)
                result_dict["Overall_start_char"].append(int(doc1_ents.loc[index1,self.span_start_col]))
                result_dict["Annot_1_char"].append(str(doc1_ents.loc[index1,self.span_start_col]) + "-" + str(doc1_ents.loc[index1,self.span_end_col]))
                result_dict["Annot_2_char"].append("")
                result_dict["Exact_Match?"].append(False)
                result_dict["Duplicate_Matches?"].append(False)
                result_dict["Overlap?"].append(False)
                for attr in (self.attr + self.rel_source_col + self.rel_target_col):
                    result_dict["A1_" + attr].append(str(doc1_ents.loc[index1,attr]))
                    result_dict["A2_" + attr].append("")
        del result_dict["Index"] #Only needed to link new index1 with old index1 when there's a duplicate match with the same ent from doc2
        for index2 in doc2_ents.index: #Cases where an ent in doc2 doesn't have a match
            if index2 not in doc2_matches.keys():
                result_dict["doc_name"].append(doc_name)
                #result_dict["Match"].append("FP")
                result_dict["Annotation_1"].append("")
                result_dict["Annotation_2"].append(doc2_ents.loc[index2,self.span_col])
                result_dict["Matching_label?"].append(False)
                if label == 1:
                    result_dict["Annot_1_label"].append("")
                    result_dict["Annot_2_label"].append(doc2_ents.loc[index2,self.label_col])
                else:
                    result_dict["Annot_1_label"].append("")
                    result_dict["Annot_2_label"].append("")
                result_dict["Annot_1_char"].append("")
                result_dict["Annot_2_char"].append(str(doc2_ents.loc[index2,self.span_start_col]) + "-" + str(doc2_ents.loc[index2,self.span_end_col]))
                result_dict["Overall_start_char"].append(int(doc2_ents.loc[index2,self.span_start_col]))
                result_dict["Exact_Match?"].append(False)
                result_dict["Duplicate_Matches?"].append(False)
                result_dict["Overlap?"].append(False)
                for attr in (self.attr + self.rel_source_col + self.rel_target_col):
                    result_dict["A1_" + attr].append("")
                    result_dict["A2_" + attr].append(str(doc2_ents.loc[index2,attr]))
        return pd.DataFrame.from_dict(result_dict).sort_values(by=['Overall_start_char']).reset_index(drop=True)

    
    def df_agreement_metrics(self,df):
        agreement_type_dict = dict()
        #kripendorf alpha?
        
        #span matching
        tp = df[df["Overlap?"] == True].shape[0]
        fp = df[(df["Overlap?"] == False) & (df['Annotation_1'] == "")].shape[0]
        fn = df[(df["Overlap?"] == False) & (df['Annotation_2'] == "")].shape[0]
        recall = self.recall(tp,fp,fn)
        prec = self.precision(tp,fp,fn)
        f1 = self.pairwise_f1(tp,fp,fn)
        agreement_type_dict['span_metrics'] = pd.DataFrame.from_dict({'Span Matching':[tp,fp,fn,recall,prec,f1]},columns=['TP','FP','FN','Recall','Precision',"F1"],orient='index')
        
        if self.text != None:
            cohens = self.cohens_kappa(df)
        else:
            cohens = ["-" for x in range(6)]
        agreement_type_dict['token_level_metrics'] = pd.DataFrame.from_dict({'Token Level Matching':[*cohens]},columns=['TP','FP','FN','TN',"F1","Token-level Cohens Kappa"],orient='index')
        
        #label matching
        #df_o = df[(df["Overlap?"] == True)]
        df_o = df #use the above line if you would like to filter out span mismatches
        multicls_score = dict()
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        overall_f1 = 0
        count = 0
        
        dup_lab_val = [x for xs in set(df_o[df_o["Duplicate_Matches?"] == True]["Annot_1_label"].to_list() + df_o[df_o["Duplicate_Matches?"] == True]["Annot_2_label"].to_list()) for x in xs.split(' || ')]
        for lab in set(df_o[df_o["Duplicate_Matches?"] == False]["Annot_1_label"].to_list() + df_o[df_o["Duplicate_Matches?"] == False]["Annot_2_label"].to_list() + dup_lab_val):
            if lab in self.NULL_RELATIONSHIP_VALUES:
                continue
            df_lab = df_o[((df_o["Annot_1_label"] == lab) | (df_o["Annot_2_label"] == lab)) & (df_o["Duplicate_Matches?"] == False)]
            tp = df_lab[df_lab["Annot_1_label"] == df_lab["Annot_2_label"]].shape[0]
            fp = df_lab[(df_lab["Annot_1_label"] != lab) & (df_lab["Annot_2_label"] == lab)].shape[0]
            fn = df_lab[(df_lab["Annot_1_label"] == lab) & (df_lab["Annot_2_label"] != lab)].shape[0]
            
            #Cases of duplicate matches
            for index,row in df_o[df_o["Duplicate_Matches?"] == True].iterrows():
                annot1_list = set(row["Annot_1_label"].split(" || "))
                annot2_list = set(row["Annot_2_label"].split(" || "))
                if (lab in annot1_list) and (lab in annot2_list):
                    tp += 1
                    continue
                for annot1_lab in annot1_list:
                    for annot2_lab in annot2_list:
                        if (annot1_lab == lab) and (annot2_lab != lab) and (annot2_lab not in annot1_list):
                            fn += 1
                        elif (annot1_lab != lab) and (annot2_lab == lab) and (annot1_lab not in annot2_list):
                            fp += 1
            
            recall = self.recall(tp,fp,fn)
            prec = self.precision(tp,fp,fn)
            f1 = self.pairwise_f1(tp,fp,fn)
            multicls_score[lab] = [tp,fp,fn,recall,prec,f1]
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
            if not isinstance(f1, str):
                overall_f1 += f1
                count += 1
        agreement_type_dict['label_metrics'] = pd.DataFrame.from_dict(multicls_score,columns=['TP','FP','FN','Recall','Precision',"F1"],orient='index')
        
        #label_overall
        
        agreement_type_dict['overall_label_metrics'] = pd.DataFrame.from_dict({"Overall Label Metrics":[overall_tp,overall_fp,overall_fn,self.recall(overall_tp,overall_fp,overall_fn),self.precision(overall_tp,overall_fp,overall_fn),self.pairwise_f1(overall_tp,overall_fp,overall_fn),overall_f1/count]},columns=['TP','FP','FN','Recall','Precision',"F1 (Micro)", "F1 (Macro)"],orient='index')
        
        
        if (self.label_ind_attr == None) and (self.label_dep_attr == None) and (self.attr == None):
            agreement_type_dict['attr_metrics'] = pd.DataFrame()
            agreement_type_dict['overall_attr_metrics'] = pd.DataFrame()
            agreement_type_dict['attr_metrics_hier'] = pd.DataFrame()
            agreement_type_dict['overall_attr_metrics_hier'] = pd.DataFrame()
            return agreement_type_dict  
        
        #attr matching (independent of label matching)
        multicls_score = dict()
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        overall_f1 = 0
        count = 0

        if self.label_ind_attr == None:
            attr_list = self.attr
        else:
            attr_list = self.label_ind_attr  
            
        if (attr_list != []) and (attr_list != None):
            for attr in attr_list:
                dup_attr_val = [x for xs in set(df_o[df_o["Duplicate_Matches?"] == True]["A1_" + attr].to_list() + df_o[df_o["Duplicate_Matches?"] == True]["A2_" + attr].to_list()) for x in xs.split(' || ')]
                for attr_val in set((df_o[df_o["Duplicate_Matches?"] == False]["A1_" + attr]).to_list() + (df_o[df_o["Duplicate_Matches?"] == False]["A2_" + attr]).to_list() + dup_attr_val):
                    df_attr = df_o[((df_o["A1_" + attr] == attr_val) | (df_o["A2_" + attr] == attr_val)) & (df_o["Duplicate_Matches?"] == False)]
                    tp = df_attr[df_attr["A1_" + attr] == df_attr["A2_" + attr]].shape[0]
                    fp = df_attr[(df_attr["A1_" + attr] != attr_val) & (df_attr["A2_" + attr] == attr_val)].shape[0]
                    fn = df_attr[(df_attr["A1_" + attr] == attr_val) & (df_attr["A2_" + attr] != attr_val)].shape[0]

                    #Cases of duplicate matches
                    for index,row in df_o[df_o["Duplicate_Matches?"] == True].iterrows():
                        annot1_list = set(row["A1_" + attr].split(" || "))
                        annot2_list = set(row["A2_" + attr].split(" || "))
                        if (attr_val in annot1_list) and (attr_val in annot2_list):
                            tp += 1
                            continue
                        for annot1_attr in annot1_list:
                            for annot2_attr in annot2_list:
                                if (annot1_attr == attr_val) and (annot2_attr != attr_val) and (annot2_attr not in annot1_list):
                                    fn += 1
                                elif (annot1_attr != attr_val) and (annot2_attr == attr_val) and (annot1_attr not in annot2_list):
                                    fp += 1


                    recall = self.recall(tp,fp,fn)
                    prec = self.precision(tp,fp,fn)
                    f1 = self.pairwise_f1(tp,fp,fn)
                    multicls_score[(attr,attr_val)] = [tp,fp,fn,recall,prec,f1]
                    overall_tp += tp
                    overall_fp += fp
                    overall_fn += fn
                    if not isinstance(f1, str):
                        overall_f1 += f1
                        count += 1
            output_df = pd.DataFrame.from_dict(multicls_score,columns=['TP','FP','FN','Recall','Precision',"F1"],orient='index')
            output_df.index = pd.MultiIndex.from_tuples(output_df.index)
            agreement_type_dict['attr_metrics'] = output_df

            #attr overall

            agreement_type_dict['overall_attr_metrics'] = pd.DataFrame.from_dict({"Overall Attribute Metrics (label independent)":[overall_tp,overall_fp,overall_fn,self.recall(overall_tp,overall_fp,overall_fn),self.precision(overall_tp,overall_fp,overall_fn),self.pairwise_f1(overall_tp,overall_fp,overall_fn),overall_f1/count]},columns=['TP','FP','FN','Recall','Precision',"F1 (Micro)", "F1 (Macro)"],orient='index')
        else:
            agreement_type_dict['attr_metrics'] = pd.DataFrame()
            agreement_type_dict['overall_attr_metrics'] = pd.DataFrame()
                        
        
        #attr matching (label dependent)
        df_ol = df_o[(df_o["Matching_label?"] == True) & (df["Overlap?"] == True)]
        multicls_score = dict()
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        overall_f1 = 0
        count = 0

        if self.label_dep_attr == None:
            attr_list = self.attr
        else:
            attr_list = self.label_dep_attr
        if (attr_list != []) and (attr_list != None):
            for lab in set(df_ol[df_ol["Duplicate_Matches?"] == False]["Annot_1_label"].to_list() + df_ol[df_ol["Duplicate_Matches?"] == False]["Annot_2_label"].to_list()):
                df_lab = df_ol[((df_ol["Annot_1_label"] == lab) & (df_ol["Annot_2_label"] == lab)) & (df_ol["Duplicate_Matches?"] == False)] #df_lab is where there is overlap, labels match, and labels are whatever lab is
                for attr in attr_list:
                    for attr_val in set((df_lab["A1_" + attr]).to_list() + (df_lab["A2_" + attr]).to_list()):
                        if attr_val in self.NULL_RELATIONSHIP_VALUES:
                            continue
                        df_attr = df_lab[(df_lab["A1_" + attr] == attr_val) | (df_lab["A2_" + attr] == attr_val)]
                        tp = df_attr[df_attr["A1_" + attr] == df_attr["A2_" + attr]].shape[0]
                        fp = df_attr[(df_attr["A1_" + attr] != attr_val) & (df_attr["A2_" + attr] == attr_val)].shape[0]
                        fn = df_attr[(df_attr["A1_" + attr] == attr_val) & (df_attr["A2_" + attr] != attr_val)].shape[0]
                        recall = self.recall(tp,fp,fn)
                        prec = self.precision(tp,fp,fn)
                        f1 = self.pairwise_f1(tp,fp,fn)
                        multicls_score[(lab,attr,attr_val)] = [tp,fp,fn,recall,prec,f1]
                        overall_tp += tp
                        overall_fp += fp
                        overall_fn += fn
                        if not isinstance(f1, str):
                            overall_f1 += f1
                            count += 1

            output_df = pd.DataFrame.from_dict(multicls_score,columns=['TP','FP','FN','Recall','Precision',"F1"],orient='index')
            output_df.index = pd.MultiIndex.from_tuples(output_df.index)
            agreement_type_dict['attr_metrics_hier'] = output_df

            #attr overall hier

            agreement_type_dict['overall_attr_metrics_hier'] = pd.DataFrame.from_dict({"Overall Attribute Metrics (label dependent)":[overall_tp,overall_fp,overall_fn,self.recall(overall_tp,overall_fp,overall_fn),self.precision(overall_tp,overall_fp,overall_fn),self.pairwise_f1(overall_tp,overall_fp,overall_fn),overall_f1/count]},columns=['TP','FP','FN','Recall','Precision',"F1 (Micro)", "F1 (Macro)"],orient='index')
        
        #Relationships
        
        multicls_score = dict()
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        overall_f1 = 0
        count = 0
        
        if self.rel_source_col != []:
            for rel in self.rel_source_col:
                TP = df[df[rel + "_Match?"] == "TP"].shape[0]
                FP = df[df[rel + "_Match?"] == "FP"].shape[0]
                FN = df[df[rel + "_Match?"] == "FN"].shape[0]
                recall = self.recall(TP,FP,FN)
                prec = self.precision(TP,FP,FN)
                f1 = self.pairwise_f1(TP,FP,FN)
                overall_tp += TP
                overall_fp += FP
                overall_fn += FN
                if not isinstance(f1, str):
                    overall_f1 += f1
                    count += 1
                multicls_score[rel] = [TP,FP,FN,recall,prec,f1]
            output_df = pd.DataFrame.from_dict(multicls_score,columns=['TP','FP','FN','Recall','Precision',"F1"],orient='index')
            agreement_type_dict['rel_metrics'] = output_df
            
            agreement_type_dict['overall_rel_metrics'] = pd.DataFrame.from_dict({"Overall Relationship Metrics":[TP,FP,FN,self.recall(TP,FP,FN),self.precision(TP,FP,FN),self.pairwise_f1(TP,FP,FN),overall_f1/count]},columns=['TP','FP','FN','Recall','Precision',"F1 (Micro)","F1 (Macro)"],orient='index')
        else:
            agreement_type_dict['rel_metrics'] = pd.DataFrame()
            agreement_type_dict['overall_rel_metrics'] = pd.DataFrame()
        
        return agreement_type_dict
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
'''
ent_dict_1[df_doc_name].append(index)
                ent_dict_1[df_span_text].append(ents.text)
                ent_dict_1[df_start_char].append(ents.start_char)
                ent_dict_1[df_end_char].append(ents.end_char)
                for attr in attributes:
                    try:
                        val = getattr(ents, attr)
                    except AttributeError:
                        val = getattr(ents._, attr)
                    ent_dict_1[attr].append(val)
                if labels == 1:
                    ent_dict_1[df_concept_label].append(ents.label_)
                else:
                    ent_dict_1[df_concept_label].append("")
                    
'''
"""
def merge_df(doc1_matches,doc2_matches,doc1_ents,doc2_ents):
    if self.label_col in list(doc1_ents.columns) and self.label_col in list(doc2_ents.columns):
        label=1
    else:
        label=0
        
    merge_dict = {self.doc_col:[],self.span_col:[],self.span_start_col:[],self.span_end_col:[],self.label_col:[],"Index" : [],"Match":[]}
    for attr in self.attr:
        merge_dict[attr] = []
        
    if len(doc1_ents['doc name'].tolist()) > 0:
        doc_name = doc1_ents['doc name'].tolist()[0]
    elif len(doc2_ents['doc name'].tolist()) > 0:
        doc_name = doc2_ents['doc name'].tolist()[0]
    for index1 in doc1_ents.index: #iterate through all ents in set one
        if index1 in doc1_matches.keys(): #Cases where the ent has a match
            #if another index1 is in doc2_matches.values(), then add it to this row
            first_index2 = sorted(doc1_matches[index1])[0]
            first_index1 = sorted(doc2_matches[first_index2])[0]
            if first_index1 < index1: #Cases where the entity in doc2 already matched a previous doc1 ent
                #Add to index: sorted(doc2_matches[first_index2])[0]
                duplicate_match_index = merge_dict["Index"].index(first_index1)
                merge_dict[self.span_col][duplicate_match_index] = doc2_ents.loc[first_index2,'Span Text']
                merge_dict[self.span_start_col][duplicate_match_index] = int(doc2_ents.loc[first_index2,'start loc'])
                merge_dict[self.span_end_col][duplicate_match_index] = int(doc2_ents.loc[first_index2,'end loc'])
                for attr in self.attr:
                    merge_dict[attr][duplicate_match_index] = (doc2_ents.loc[first_index2,attr])
                if label==1:
                    merge_dict[self.label_col][duplicate_match_index] = doc2_ents.loc[first_index2,'Concept Label']
            else:
                if len(doc1_matches[index1]) == 1:
                    index2 = doc1_matches[index1][0]
                    if len(doc1_ents.loc[index1,'Span Text']) >= len(doc2_ents.loc[index2,'Span Text']):
                        merge_dict[self.doc_col].append(doc_name)
                        merge_dict["Index"].append(index1)
                        merge_dict["Match"].append("TP")
                        merge_dict[self.span_col].append(doc1_ents.loc[index1,'Span Text'])
                        merge_dict[self.span_start_col].append(int(doc1_ents.loc[index1,'start loc']))
                        merge_dict[self.span_end_col].append(int(doc1_ents.loc[index1,'end loc']))
                        for attr in self.attr:
                            merge_dict[attr].append(doc1_ents.loc[index1,attr])
                        if label == 1:
                            merge_dict[self.label_col].append(doc1_ents.loc[index1,self.label_col])
                        else:
                            merge_dict[self.label_col].append("")
                    else:
                        merge_dict[self.doc_col].append(doc_name)
                        merge_dict["Index"].append(index1)
                        merge_dict["Match"].append("TP")
                        merge_dict[self.span_col].append(doc2_ents.loc[index2,'Span Text'])
                        merge_dict[self.span_start_col].append(int(doc2_ents.loc[index2,'start loc']))
                        merge_dict[self.span_end_col].append(int(doc2_ents.loc[index2,'end loc']))
                        for attr in self.attr:
                            merge_dict[attr].append(doc2_ents.loc[index2,attr])
                        if label == 1:
                            merge_dict[self.label_col].append(doc2_ents.loc[index2,self.label_col])
                        else:
                            merge_dict[self.label_col].append("")
                else:
                    merge_dict[self.doc_col].append(doc_name)
                    merge_dict["Index"].append(index1)
                    merge_dict["Match"].append("TP")
                    merge_dict[self.span_col].append(doc1_ents.loc[index1,'Span Text'])
                    merge_dict[self.span_start_col].append(int(doc1_ents.loc[index1,'start loc']))
                    merge_dict[self.span_end_col].append(int(doc1_ents.loc[index1,'end loc']))
                    for attr in self.attr:
                        merge_dict[attr].append(doc1_ents.loc[index1,attr])
                    if label == 1:
                        merge_dict[self.label_col].append(doc1_ents.loc[index1,self.label_col])
                    else:
                        merge_dict[self.label_col].append("")
        else: #Cases where an ent in doc1 doesn't have a match
            merge_dict[self.doc_col].append(doc_name)
            merge_dict["Index"].append(index1)
            merge_dict["Match"].append("FN")
            merge_dict[self.span_col].append(doc1_ents.loc[index1,'Span Text'])
            merge_dict[self.span_start_col].append(int(doc1_ents.loc[index1,'start loc']))
            merge_dict[self.span_end_col].append(int(doc1_ents.loc[index1,'end loc']))
            for attr in self.attr:
                merge_dict[attr].append(doc1_ents.loc[index1,attr])
            if label == 1:
                merge_dict[self.label_col].append(doc1_ents.loc[index1,self.label_col])
            else:
                merge_dict[self.label_col].append("")

    del merge_dict["Index"] #Only needed to link new index1 with old index1 when there's a duplicate match with the same ent from doc2
    for index2 in doc2_ents.index: #Cases where an ent in doc2 doesn't have a match
        if index2 not in doc2_matches.keys():
            merge_dict[self.doc_col].append(doc_name)
            merge_dict[self.span_col].append(doc2_ents.loc[index2,'Span Text'])
            merge_dict["Match"].append("FP")
            merge_dict[self.span_start_col].append(int(doc2_ents.loc[index2,'start loc']))
            merge_dict[self.span_end_col].append(int(doc2_ents.loc[index2,'end loc']))
            for attr in self.attr:
                merge_dict[attr].append(doc2_ents.loc[index2,attr])
            if label == 1:
                merge_dict[self.label_col].append(doc2_ents.loc[index2,self.label_col])
            else:
                merge_dict[self.label_col].append("")
            
    return pd.DataFrame.from_dict(merge_dict).sort_values(by=['start loc']).reset_index(drop=True)
    """

##To-do:
##Add additional checks for everything. Eg. check that attributes/relationships are in all dataframes
##Put attr_list in for loop for import. Currently the ehost class only permits accessing to one config file