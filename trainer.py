import pickle
from numpy import array
import sys
from sklearn.linear_model import LogisticRegression
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()

classifier = LogisticRegression()
results = []

titleList = ['adjunct', 'admiral', 'advocate', 'agent', 'ambassador', 'ambassador', 'analyst', 'archbishop', 'archdeacon', 
             'architect', 'assistant', 'atty.', 'baron', 'bishop', 'brother', 'buddha', 'burgess', 'captain', 'cardinal', 'ceo', 
             'chairman', 'chairwoman', 'chancellor', 'chief', 'cleric', 'columnist', 'commander', 'commissioner', 'comrade', 'congressman', 
             'constable', 'consul', 'corporal', 'councillor', 'count', 'count', 'countess', 'crown', 'dalai', 'dame', 'deacon', 'dean', 
             'delegate', 'department', 'deputy', 'director', 'doctor', 'dr.', 'duchess', 'duke', 'elder', 'emperor', 'empress', 
             'excellency', 'executive', 'foreign', 'foreman', 'fr.', 'friar', 'general', 'governor', 'grace', 'grand', 'grandmaster', 
             'head', 'her', 'highness', 'his', 'honorary', 'journalist', 'judge', 'justice', 'king', 'lawyer', 'lieutenant', 'lord', 
             'madam', 'magistrate', 'majesty', 'major', 'manager', 'marquess', 'marquis', 'master', 'mayor', 'minister', 'miss', 
             'monsignor', 'mp', 'mr.', 'mrs.', 'mufti', 'officer', 'owner', 'partner', 'pastor', 'patriarch', 'police', 'pope', 
             'prefect', 'prefect', 'premier', 'president', 'priest', 'prince', 'princess', 'principal', 'professor', 'project', 
             'queen', 'regional', 'representative', 'reverend', 'saint', 'secretary', 'senator', 'senior', 'shadow', 'sir', 
             'sister', 'speaker', 'special', 'sultan', 'superintendent', 'team', 'treasurer', 'vice', 'viscount', 'prime', 'first', 'lady'
             'madame', 'attorney', 'law', 'author', 'chair', 'pm', 'historian', 'leader', 'high', 'gen.', 'gen', 'mr', 'mrs', 'sergeant',
             'mayoralty', 'interior', 'elect', 'acting', 'home', 'prosecutor']


with open(sys.argv[1], 'rb') as t:
    training_data = pickle.load(t)

def ner(sentence):
    length = len(sentence)-1
    for i, chunks in enumerate(sentence):
        nerList = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        sourcetext = chunks[0]
        tag = chunks[1]
        sourcePrevText = 'None'
        prevTag = 'None'
        sourceNextText = 'None'
        nextTag = 'None'
        sourcePrevPrevText = 'None'
        prevPrevTag = 'None'
        sourceNextNextText = 'None'
        nextNextTag = 'None'
        if i != 0:
            prevText = sentence[i-1][0]
            prevTag = sentence[i-1][1]
        if i > 1:
            prevPrevText = sentence[i-2][0]
            prevPrevTag = sentence[i-2][1]
        if i < (length - 1):
            nextNextText = sentence[i+2][0]
            nextNextTag = sentence[i+2][1]
        if i != length:
            nextText = sentence[i+1][0]
            nextTag = sentence[i+1][1]
        prevText = lem.lemmatize(sourcePrevText)
        nextText = lem.lemmatize(sourceNextText)
        prevPrevText = lem.lemmatize(sourcePrevPrevText)
        nextNextText = lem.lemmatize(sourceNextNextText)
        text = lem.lemmatize(sourcetext)
        if chunks[2] == "TITLE":
            nerList[-1] = 1

        if text.lower() in titleList:
            nerList[0] = 1
        if prevPrevTag == 'JJ' and prevTag == 'NN' and (tag == 'NN' or tag == 'NNP'):
            nerList[1] = 1
        if nextText == 'of' and (tag == 'NN' or tag == 'NNP'):
            nerList[2] = 1
        if prevTag == 'PRP' and (tag == 'NN' or tag == 'NNP'):
            nerList[3] = 1
        if prevTag == 'NNP' and (tag == 'NN' or tag == 'NNP'):
            nerList[4] = 1
        if prevTag == 'DT' and (tag == 'NN' or tag == 'NNP'):
            nerList[5] = 1
        if prevTag == 'NN' and (tag == 'NN' or tag == 'NNP'):
            nerList[6] = 1
        if prevPrevTag == 'PRP$' and (tag == 'NN' or tag == 'NNP'):
            nerList[7] = 1
        if prevPrevTag == 'DT' and prevTag == 'NN':
            nerList[8] = 1
        if prevTag == 'CC' and (tag == 'NN' or tag == 'NNP'):
            nerList[9] = 1
        if prevTag == 'POS' and (tag == 'NN' or tag == 'NNP'):
            nerList[10] = 1
        if prevPrevTag == 'NNP' and prevTag == 'POS':
            nerList[11] = 1
        if prevPrevTag == 'NN' and prevTag == 'IN':
            nerList[12] = 1
        if prevTag == 'IN' and (tag == 'NN' or tag == 'NNP'):
            nerList[13] = 1
        if nextNextTag == 'NNP' and nextTag == 'NNP' and (tag == 'NN' or tag == 'NNP'):
            nerList[14] = 1
        if nextTag == 'NNP' and prevTag == 'NNP' and (tag == 'NN' or tag == 'NNP'):
            nerList[15] = 1
        if prevPrevTag == 'NNP' and prevTag == 'NNP' and (tag == 'NN' or tag == 'NNP'):
            nerList[16] = 1
        if text == 'CEO':
            nerList[17] = 1
        if prevTag == 'NNP' and nextTag == 'POS' and (tag == 'NN' or tag == 'NNP'):
            nerList[18] = 1
        if prevTag == 'NNP' and (tag == 'NN' or tag == 'NNP'):
            nerList[19] = 1
        if prevTag == 'NN' and (tag == 'NN' or tag == 'NNP'):
            nerList[20] = 1
        if text.lower() == 'vice':
            nerList[21] = 1
        if text.lower() == 'vice' and nextText == '-':
            nerList[22] = 1
        if prevTag == 'VBD' and (tag == 'NN' or tag == 'NNP'):
            nerList[23] = 1
        if tag == 'NN' or tag == 'NNP':
            nerList[24] = 1
        if sourcetext[0].isupper():
            nerList[25] = 1
        if prevText == '-' and prevPrevText.lower() == 'vice':
            nerList[26] = 1
        if (prevText.lower() in titleList or prevPrevText.lower() in titleList) and text.lower() in titleList:
            nerList[27] = 1
        if text.lower() in titleList and (nextText.lower() in titleList or nextText.lower() in titleList):
            nerList[28] = 1
        results.append(nerList)
        
if __name__ == "__main__":
    for i in training_data:
            ner(i)
    data = array(results)
    features = data[:,:29]
    target = data[:,29]
    classifier.fit(features, target)
    
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(classifier, f)
        
        
        
        
#==============================================================================
#     rez = []
#     rez2 = []
#     targz = []
#     rez.append(classifier.predict(features))
#     for i in rez[0]:
#         rez2.append(i)
#     for i in target:
#         targz.append(i)
#     print(f1_score(targz, rez2))
#==============================================================================

    
