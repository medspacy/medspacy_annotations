# Pairwise Performance and Annotation import code

Python-based code for comparison and import of doubly annotated text.

## Usage

'AnnotationAggregator' contains the python-based source code for all functionality. See the tutorials and below description for further details on functionality.

Under 'notebooks' are the tutorials for importing and evaluating annotations.

# Description

## Streamlining Inter-Annotator evaluation with pandas, medspaCy, and python
John Stanley, MS1,2, Mengke Hu, PhD1,2, Patrick R. Alba, MS1,2,
Hannah Eyre, MS1,2, Annie Bowles, MS1,2, Qiwei Gan, PhD1,2,
Elizabeth Hanchrow, RN, MSN1, Scott L. DuVall, PhD1,2, Jianlin Shi, MD, PhD1,2
1 VA Salt Lake City Health Care System, UT; 2 University of Utah, Salt Lake City, UT, USA

#### Introduction
In clinical Natural Language Processing (NLP), comparing annotation agreement between human annotators, refer-
ence standards, and NLP output is essential for system development and validation1–3. Many annotation tools exist
to facilitate the human and machine annotation process 1. This contributes to many different annotation schemas,
file types, and internal processes for comparing annotations. Instead of solely relying on these different workflow
approaches, we propose to use spaCy and pandas dataframes as intermediate annotation holders, with standardized
functions for handling these data formats.

The specific contributions of this project are a python-based, open-source package† that extends medspaCy 4 (our
community-utilized medical extension of spaCy) functionality by standardizing import/export for select annotation
types and versatile comparison of annotated corpora, independent of the annotation software or methods employed.

#### Project Description
Figure 1: Overview of component design.
Annotated schemas can be complicated, potentially containing multiple labels, label-sublabel hierarchies, unlabeled
or labeled relationships with or without direction, allowing 2+ overlapping annotations, etc. We designed our im-
port/export, formal metrics, and comparison tables to handle the different annotation structures. Import functionality
currently supports eHost, spaCy, and Brat annotations into pandas dataframe objects. Multiple dataframe objects can
then be compared. The central output consists of a comparison table for pairwise agreement, containing side by side
matched/unmatched spans and all related span information, textual context, and boolean matching values for various
span features. Critically, this gives users the flexibility to independently explore and analyze matching data. The
package also includes functionality to parse the resultant dataframe and output an HTML file of formal performance
and token-based agreement metrics for span, label (and sublabel), and relationship matching.

Future iterations may include additional visualizations, support for more annotation types, and other improvements.

#### Acknowledgements
This work was supported using resources and facilities of the Department of Veterans Affairs (VA) Informatics and
Computing Infrastructure (VINCI), VA ORD 22-D4V.
References
[1] Neves M, ˇSeva J. An extensive review of tools for manual annotation of documents. Briefings in Bioinformatics.
2021 1;22:146-63.
[2] Pustejovsky J, Stubbs A. Natural Language Annotation for Machine Learning; 2012.
[3] Artstein R. In: Inter-annotator Agreement. Springer Netherlands; 2017. p. 297-313.
[4] Eyre H, Chapman AB, Peterson KS, Shi J, Alba PR, Jones MM, et al. Launching into clinical space with
medspaCy: a new clinical text processing toolkit in Python. AMIA Annual Symposium proceedings AMIA
Symposium. 2021;2021:438-47.

†https://github.com/medspacy/medspacy_annotations