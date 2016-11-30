import pickle
from numpy import array
from spacy.en import English
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from nltk.stem.wordnet import WordNetLemmatizer
import sys

lem = WordNetLemmatizer()
parser = English()
gyep = LogisticRegression()

with open(sys.argv[1], 'rb') as t:
    test_data = pickle.load(t)

with open(sys.argv[2], 'rb') as c:
    classifier = pickle.load(c)

output_path = sys.argv[3]

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
             
results = []

def ner(sentence):
    resultList = []
    length = len(sentence)-1
    for i, chunks in enumerate(sentence):
        nerList = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
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

        targetNum = (classifier.predict([nerList])[0])
        outcome = ""
        if targetNum == 0:
            outcome = "O"
        if targetNum == 1:
            outcome = "TITLE"
        result = (text, outcome)
        resultList.append(result)
    results.append(resultList)

if __name__ == "__main__":
    for items in test_data:
        ner(items)
    with open(output_path, 'wb') as r:
        pickle.dump(results, r)
        
        
    
#==============================================================================
#     dat = array(results)
#     print(dat.shape)
#     X = dat[:,:27]
#     Y = dat[:,27]
#     gyep.fit(X, Y)
#     rez = []
#     rez2 = []
#     targz = []
#     rez.append(gyep.predict(X))
#     for i in rez[0]:
#         rez2.append(i)
#     for i in Y:
#         targz.append(i)
#     print(f1_score(targz, rez2))
#==============================================================================
        
    
    
    
    
    

                 
                 
                 
                 
                 
                 
                 
                 
        
         
#==============================================================================
#          for i in range(0, 1000):
#              for x in training_set[i]:
#                  if "analyst" in x:
#                      ner(training_set[i])
#==============================================================================
                 








#==============================================================================
# if __name__ == "__main__":
#     with open('training_data.dat', 'rb') as f:
#         training_set = pickle.load(f)
#         a = set()
#         for i in training_set:
#             list = []
#             for j in i:
#                 if "TITLE" in j:
#                     list.append(j[0])
#             if list:
#                 print((list))
#==============================================================================
#==============================================================================
# a = np.array([[1,2,3,4], [1,0,0,4], [0,2,3,3]])
# x = a[:, 0:3]
# y = a[:, 3]
# 
# gyep.fit(x, y)
# 
# 
# with open('classifier.dat', 'wb') as f:
#     pickle.dump(gyep, f)
# 
# with open('classifier.dat', 'rb') as g:
#     gyep2 = pickle.load(g)
# print(gyep2.predict([[1,3,3]]))
#==============================================================================

        

