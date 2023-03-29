# IAA and Annotation import code

Python-based code for comparison, analysis, and import of doubly annotated text.

## Usage

'IAA' contains the python-based source code for all functionality. See the tutorials and below description for further details on functionality.

Under 'notebooks' are tutorials for importing and evaluating annotations.

# Description

Evaluating Inter-Annotator Agreement using pandas and spaCy

John Stanley, MS1,2, Mengke Hu, PhD1,2, Patrick R. Alba, MS1,2,
Hannah Eyre, MS1,2, Annie Bowles, MS1,2, Qiwei Gan, PhD1,2,
Elizabeth Hanchrow, RN, MSN1, Scott L. DuVall, PhD1,2, Jianlin Shi, MD, PhD1,2
1 VA Salt Lake City Health Care System, UT; 2 University of Utah, Salt Lake City, UT, USA

## Introduction
In clinical Natural Language Processing (NLP), a high-quality, manually annotated corpus is essential for model
development and evaluation of existing methods1–3. Since manual annotation is an expensive, multi-step process, it
is important to use software tools to ensure correct annotations are being produced3,4. One method to approximate
correctness is to measure inter-annotator agreement (IAA) between multiple annotators and perform adjudication.
Adjudicators compare annotations to resolve disagreements, refine annotation guidelines, and determine reliability of
the output set. Many web-based and stand-alone annotation tools exist to facilitate the annotation process2. However,
only some of these tools assist adjudication and most lack clean integration with Python. To assist with evaluation and
adjudication of annotated clinical text, we developed an open-source tool in Python1 that allows flexible, user-friendly,
and robust comparison of annotated corpora.

## Tool Description
Our tool is designed to integrate into natural workflows, be user-friendly, and encourage thorough data comparison. To
accomplish this, we focused input and outputs on pandas dataframes and spaCy objects, which are heavily used within
the data science and NLP communities. Additionally, there is support for importing BRAT and eHOST annotations.
Given a dataframe or spaCy spangroups, entities, or documents, the tool has three basic outputs, including: 1) general
statistics, including true positive, false positive, false negative, precision, recall, and F-micro, 2) a dataframe containing
matched and unmatched spans and other valuable information, and 3) a dataset containing all distinct annotations.
By supporting different span matching criteria and outputing valuable span information, this tool allows for more
robust comparisons and analytics. Span matching criteria includes requiring entity labels and/or other attributes to
match. The output dataframes have boolean values indicating duplicate matches, label matches, and exact matches,
and also indicates true positive, false positive, false negatives, attributes and labels. Using this information, data can
be grouped, filtered, and otherwise manipulated to provide interesting views and statistics. This can be compounded
by including custom attributes as filters and matching criteria, such as sentence length, number of tokens, etc.
Future iterations may include integration with medspaCy5, handling 3+ annotators, and other interface improvements.

## Acknowledgements
This work was supported using resources and facilities of the Department of Veterans Affairs (VA) Informatics and
Computing Infrastructure (VINCI), funded under the research priority to Put VA Data to Work for Veterans (VA ORD
22-D4V). The views expressed are those of the authors and do not necessarily represent the views or policy of the
Department of Veterans Affairs or the United States Government

## References
[1] Chapman WW, Nadkarni PM, Hirschman L, D’Avolio LW, Savova GK, Uzuner O. Overcoming barriers to NLP
for clinical text: the role of shared tasks and the need for additional creative solutions. Journal of the American
Medical Informatics Association. 2011 9;18:540-3.
[2] Neves M, ˇSeva J. An extensive review of tools for manual annotation of documents. Briefings in Bioinformatics.
2021 1;22:146-63.
[3] Pustejovsky J, Stubbs A. Natural Language Annotation for Machine Learning; 2012.
[4] Artstein R. In: Inter-annotator Agreement. Springer Netherlands; 2017. p. 297-313.
[5] Eyre H, Chapman AB, Peterson KS, Shi J, Alba PR, Jones MM, et al. Launching into clinical space with
medspaCy: a new clinical text processing toolkit in Python. AMIA Annual Symposium proceedings AMIA
Symposium. 2021;2021:438-47.
