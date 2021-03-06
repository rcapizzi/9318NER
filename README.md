# 9318NER
NER project.

This project was for COMP9318 Data Warehousing & Data Mining, using scikit-learn and python 3.
Goal of the task was as follows:

Named Entity Recognition. The goal of Named Entity Recognition (NER) is to
locate segments of text from input document and classify them in one of the pre-defined
categories (e.g., PERSON, LOCATION, and ORGNIZATION). 
In this project, you only need to  perform NER for a single category of TITLE. We define a TITLE as an appellation associated
with a person by virtue of occupation, office, birth, or as an honorific. For example, in the
following sentence, both Prime Minister and MP are TITLEs.

The project works by examining the grammatical context of a word to determine its likelihood of being a TITLE. For example, given the training dataset, "Adjective"-"Noun"-"X" usually results in X being a TITLE. These rules were fed into a logistic regression classifier to perform the NER task.

