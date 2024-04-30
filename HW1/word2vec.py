import gensim.downloader

def analogy(a , b , c):
    print(a + " : " + b + " :: " + c + " : ?")
    # print([(w,round(c,3)) for w,c in w2v.most_similar(positive=[c,b], negative=[a])])

    print(w2v.most_similar(positive=[c,b], negative=[a], restrict_vocab=10))

if __name__ == "__main__":
    print('Loading dataset...')
    w2v = gensim.downloader.load('word2vec-google-news-300')
    analogy('man', 'king', 'woman')
