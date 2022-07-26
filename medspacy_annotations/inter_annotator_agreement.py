def overlaps(ent1, ent2):
    """Calculates whether two spacy entities overlap

    returns bool
    """
    raise NotImplementedError("Not finished")


def exact_match(ent1, ent2):
    """Calculates whether two spacy entities have the same span

    returns bool
    """
    raise NotImplementedError("Not finished")


def document_span_confusion_matrix(doc1, doc2, fuzzy=False):
    """ Calculates a confusion matrix based on spans between two documents and whether to do fuzzy matching

    returns confusion matrix in sklearn format:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    [[true_positive,false_negative],[false_positive,true_negative]]
    """
    raise NotImplementedError("Not finished")


def corpus_span_confusion_matrix(doc_list_1, doc_list_2, fuzzy=False):
    """ Calculates a confusion matrix based on spans between two corpora and whether to do fuzzy matching.
    Adds document confusion matrices together.

    returns confusion matrix in sklearn format:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    [[true_positive,false_negative],[false_positive,true_negative]]
    """
    raise NotImplementedError("Not finished")


def calculate_f1(confusion_matrix):
    """ Calculates f1 based on a confusion matrix.
    
    returns float
    """
