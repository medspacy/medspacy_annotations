# -*- coding: utf-8 -*-
from spacy.lang.en import English
from medspacy_io.reader import EhostDocReader
from medspacy_io.reader import EhostDirReader

from spacy.lang.en import English
from spacy.tokens import Doc
import pandas as pd

def eHost_dir_reader(proj_schema:str,corpus_dir:str) -> list[Doc]: #non-overlapped entities
    dir_reader = EhostDirReader(nlp=English(), recursive=True, schema_file=proj_schema)
    docs = dir_reader.read(txt_dir=corpus_dir)
    return docs

def eHost_dir_reader_overlap(proj_schema:str,corpus_dir:str) -> list[Doc]: #overlapped entities #only for new version of medspacy-io
    dir_reader = EhostDirReader(nlp=English(), support_overlap=True,recursive=True, schema_file=proj_schema)
    docs = dir_reader.read(txt_dir=corpus_dir)
    return docs

def df_builder_overlapSpan(docs:list[Doc],files:list[str])->pd.DataFrame:
    data = []
    for i in range(len(docs)):
        doc_concepts_dic = docs[i]._.concepts
        for concept, listSpan in doc_concepts_dic.items():
            for sp in listSpan:
                dicElem = {}
                dicElem['Span Text'] = sp.text
                dicElem['Concept Label'] = sp.label_
                dicElem['start loc'] = sp.start_char
                dicElem['end loc'] = sp.end_char
                dicElem['doc name'] = files[i]
                data.append(dicElem)
    df = pd.DataFrame(data)
    return df
    