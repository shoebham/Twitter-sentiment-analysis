import random
import nltk
import pprint
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


# ###############################################
# # running this file takes a lot of time 		#
# # and it depends on your processor 			#
# # It took around 90 minutes on my i7-7th gen 	#
# ###############################################

# vote classifier class
class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers
 	
	def classify(self,features):
 		votes = []
 		for c in self._classifiers:
 			v= c.classify(features)
 			votes.append(v)
 		return mode(votes)

	def confidence(self,features):
 		votes = []
 		for c in self._classifiers:
 			v= c.classify(features)
 			votes.append(v)
 		
 		choice_votes = votes.count(mode(votes));

 		conf=choice_votes/len(votes)
 		return conf


short_pos = open("short_reviews/positive.txt").read()
short_neg = open("short_reviews/negative.txt").read()


documents =  []	
all_words =[]
# j- adjective, v-verb, r-adverb
allowed_word_types = ["J"]

# splits the document in positive or negative and makes fills all_words with allowed words
# ------------------------------------------------
for r in short_pos.split('\n'):
	documents.append((r,"pos"))
	words= word_tokenize(r)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())


for r in short_neg.split('\n'):
	documents.append((r,"neg"))
	words= word_tokenize(r)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

# ------------------------------------------------

# print(documents)

save_documents = open("picked_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

# tokenizes by words
short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

# most frequently appearing words in the document
all_words = nltk.FreqDist(all_words)

# takes the first 3000 words and stores it  
word_features = list(all_words.keys())[:5000]

save_word_feature = open("picked_algos/word_feature5k.pickle","wb")
pickle.dump(word_features, save_word_feature)
save_word_feature.close()

# finds the common words between movie review and the desired document
def find_features(document):
	words = word_tokenize(document)
	features={}
	for w in word_features:
		features[w]=(w in words)
	return features



# find common words in pos and neg movie review
feature_set = [(find_features(rev),category) for rev,category in documents]
print(feature_set)

random.shuffle(feature_set)

# positive
# training set of first N words
training_set = feature_set[:10000]
# testing set of rest of the words
testing_set = feature_set[10000:]



# posterior/Bayes algo = (occurences(p(b/a)) * likelihood(p(a)) )/evidence (p(b))

# trains the data by naive bayes algo^
classifier = nltk.NaiveBayesClassifier.train(training_set);


#prints the accuracy of the training data against testing data 
print("original NB accuracy",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)


save_classifier = open("picked_algos/originalNB5k.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()

# multinomial naive bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("multinomial naive bayes accuracy",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)


save_classifier = open("picked_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier,save_classifier)
save_classifier.close()


# Gaussian naive bayes
# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_set)
# print("Gaussian naive bayes accuracy",(nltk.classify.accuracy(GNB_classifier,testing_set))*100)

# Bernoulli naive bayes
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("Bernoulli naive bayes accuracy",(nltk.classify.accuracy(BNB_classifier,testing_set))*100)


save_classifier = open("picked_algos/BNB_classifier5k.pickle","wb")
pickle.dump(BNB_classifier,save_classifier)
save_classifier.close()



# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC
LogisticRegression_classifier = SklearnClassifier(LogisticRegression(max_iter=500))
LogisticRegression_classifier.train(training_set)
print("LogisticRegression naive bayes accuracy",(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

save_classifier = open("picked_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier,save_classifier)
save_classifier.close()




SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier naive bayes accuracy",(nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)


save_classifier = open("picked_algos/SGDClassifier_classifier5k.pickle","wb")
pickle.dump(SGDClassifier_classifier,save_classifier)
save_classifier.close()



# # takes a lot of time, not optimised 
# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC naive bayes accuracy",(nltk.classify.accuracy(SVC_classifier,testing_set))*100)


# save_classifier = open("picked_algos/SVC_classifier5k.pickle","wb")
# pickle.dump(SVC_classifier,save_classifier)
# save_classifier.close()



LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC naive bayes accuracy",(nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)


save_classifier = open("picked_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier,save_classifier)
save_classifier.close()



# # takes a lot of time, not optimised 
# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print("NuSVC naive bayes accuracy",(nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

# save_classifier = open("picked_algos/NuSVC_classifier5k.pickle","wb")
# pickle.dump(NuSVC_classifier,save_classifier)
# save_classifier.close()



voted_classifier = VoteClassifier(classifier,
	MNB_classifier,
	BNB_classifier,
	LogisticRegression_classifier,
	SGDClassifier_classifier,
	NuSVC_classifier,
	LinearSVC_classifier)
print("voted classifier accuracy",(nltk.classify.accuracy(voted_classifier,testing_set))*100)

# print("Classification:",voted_classifier.classify(testing_set[0][0]),"confidence:",voted_classifier.confidence(testing_set[0][0])*100)
# print("Classification:",voted_classifier.classify(testing_set[1][0]),"confidence:",voted_classifier.confidence(testing_set[1][0])*100)
# print("Classification:",voted_classifier.classify(testing_set[2][0]),"confidence:",voted_classifier.confidence(testing_set[2][0])*100)
# print("Classification:",voted_classifier.classify(testing_set[3][0]),"confidence:",voted_classifier.confidence(testing_set[3][0])*100)
# print("Classification:",voted_classifier.classify(testing_set[4][0]),"confidence:",voted_classifier.confidence(testing_set[4][0])*100)
# print("Classification:",voted_classifier.classify(testing_set[6][0]),"confidence:",voted_classifier.confidence(testing_set[6][0])*100)
# print("Classification:",voted_classifier.classify(testing_set[5][0]),"confidence:",voted_classifier.confidence(testing_set[5][0])*100)


def sentiment(text):
	feats = find_features(text)
	return voted_classifier.classify(feats)