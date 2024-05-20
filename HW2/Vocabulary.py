from collections import Counter 
from re import sub, compile
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
import string

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):

		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]

	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	###########################
	## TASK 1.1           	 ##
	###########################
	def tokenize(self, text):
		"""
	    
	    tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

	    :params: 
	    - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

	    :returns:
	    - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenizationhttps://www.w3schools.com/python/matplotlib_getting_started.asp
	    
	    """

		# Download packages
		nltk.download('wordnet')
		nltk.download('punkt')
  
		lemmatizer = WordNetLemmatizer()
		
		# Create tokenized list
		tokenized_text = []
		text_str = ""

		for row in text:
			text_str += row 

		text_str = text_str.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).lower()
		tokenized_text = [w for w in text_str.split() if (len(w)>1) and (not w.isnumeric())]

		#Lemmatize
		tokenized_text = [lemmatizer.lemmatize(token) for token in tokenized_text]

		return tokenized_text


	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self,corpus):
		"""
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over

	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """

		#Turn tokenized_corpus into a dictionary
		freq = {}

		for token in corpus:
			
			# Create word: frequency mapping
			if token in freq:
				freq[token] = freq[token] + 1
			else:
				freq[token] = 1
		
		#Sort tokens by frequency
		freq_sorted = sorted(freq.items(), key=lambda x:x[1], reverse=True)
		
		idx = 0
		idx2word = {}
		word2idx = {}
		freq = {}

		freq_cutoff = 1

		for token_freq in freq_sorted:

			token = token_freq[0]
			frequency = token_freq[1]

			# If token occurs more than freq_cutoff times, add to word2idx
   
			if frequency >= freq_cutoff:
				freq[token] = frequency
				word2idx[token] = idx
				idx2word[idx] = token
				idx += 1
		
		#Add UNK token at the end of the mappings
		word2idx['UNK'] = idx
		idx2word[idx] = 'UNK'
		freq['UNK'] = 1

		return word2idx, idx2word, freq


	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		"""
	    
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details

	    
	    """ 

		frequency_vals = list(self.freq.values())
		frequency_cutoff = 2

		#Token Frequency Distribution
		ids = list(self.idx2word.keys())

		fig = plt.figure(1)
		ax = fig.add_subplot(1, 1, 1)
		ax.set_title("Token Frequency Distribution")
		ax.set_xlabel("Token ID (sorted by frequency)")
		ax.set_ylabel("Frequency")
		ax.set_yscale('log')
		ax.plot(ids, frequency_vals)
		ax.axhline(y = frequency_cutoff, color = 'r', linestyle = '-') 
		ax.text(len(frequency_vals)/2, frequency_cutoff + max(frequency_vals) * 0.01, f'freq={frequency_cutoff}', color="red")


		#Cumulative Fraction Covered
		cutoff_freq_vals = [x for x in frequency_vals if x>=frequency_cutoff]
		frequency_vals_sum = sum(frequency_vals)
		cutoff_freq_vals_sum = sum(cutoff_freq_vals)
		cumulative_fraction_freq = []

		for i, freq in enumerate(frequency_vals):
			if i == 0:
				cumulative_fraction_freq.append(freq/frequency_vals_sum)
			else:
				cumulative_fraction_freq.append(cumulative_fraction_freq[i-1] + freq/frequency_vals_sum)

		#Find X% id
		percentage = round(cutoff_freq_vals_sum/frequency_vals_sum, 2)
		percentage_id = 0
		for i, c in enumerate(cumulative_fraction_freq):
			if c >= percentage:
				percentage_id = i
				break

		fig = plt.figure(2)
		ax2 = fig.add_subplot(1, 1, 1)
		ax2.set_title("Cumulative Fraction Covered")
		ax2.set_xlabel("Token ID (sorted by frequency)")
		ax2.set_ylabel("Fraction of Token Ocurrences Covered")
		ax2.plot(ids, cumulative_fraction_freq)
		ax2.axvline(x = ids[percentage_id], color = 'r', linestyle = '-')
		ax2.text(ids[percentage_id], 0.5, f'{percentage}', color="red")