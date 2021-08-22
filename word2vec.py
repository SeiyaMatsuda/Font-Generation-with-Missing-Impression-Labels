from gensim.models import KeyedVectors
import os
def word2vec():
    path = os.path.join(os.path.dirname(__file__), '../', 'word2vec/GoogleNews-vectors-negative300.bin')
    Embedding_model =KeyedVectors.load_word2vec_format(path, binary=True)
    print("Embedding　OK")
    return Embedding_model
if __name__ == '__main__':
    print(word2vec())