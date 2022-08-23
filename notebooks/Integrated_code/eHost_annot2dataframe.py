# -*- coding: utf-8 -*-
from spacy.lang.en import English
from medspacy_io.reader import EhostDocReader
from medspacy_io.reader import EhostDirReader

from spacy.lang.en import English
from spacy.tokens import Doc
import pandas as pd
from typing import List


def eHost_dir_reader(proj_schema:str,corpus_dir:str) -> List[Doc]: #non-overlapped entities
    dir_reader = EhostDirReader(nlp=English(), recursive=True, schema_file=proj_schema)
    docs = dir_reader.read(txt_dir=corpus_dir)
    return docs

def eHost_dir_reader_overlap(proj_schema:str,corpus_dir:str) -> List[Doc]: #overlapped entities #only for new version of medspacy-io
    dir_reader = EhostDirReader(nlp=English(), support_overlap=True,recursive=True, schema_file=proj_schema)
    docs = dir_reader.read(txt_dir=corpus_dir)
    return docs

def df_builder_overlapSpan(docs:List[Doc])->pd.DataFrame: #modified for the latest medspacy-io
    data = []
    for doc in docs:
        doc_concepts_list = doc._.concepts #now it is a list to used as key to extract SpanGroup
        for concept in doc_concepts_list:
            for sp in doc.spans[concept]: # iterate spans in certain spanGroup
                dicElem = {}
                dicElem['Span Text'] = sp.text
                dicElem['Concept Label'] = sp.label_
                dicElem['start loc'] = sp.start_char
                dicElem['end loc'] = sp.end_char
                dicElem['doc name'] = doc._.doc_name
                data.append(dicElem)
    df = pd.DataFrame(data)
    return df
    
 