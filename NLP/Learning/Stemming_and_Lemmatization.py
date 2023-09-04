from nltk.stem import PorterStemmer, WordNetLemmatizer 
from nltk.corpus import wordnet

def Stemmer() -> None:
    porter = PorterStemmer()

    print(porter.stem("walking"))
    print(porter.stem("ran"))
    print(porter.stem("running"))
    print(porter.stem("replacement"))

    sentence = "Lemmatization is the process of converting a word to its base form".split( )
    for token in sentence:
        print(porter.stem(token), end="  ")

def Lemmat() -> None:
    lemmatizer = WordNetLemmatizer()

    print(lemmatizer.lemmatize("walking"))
    print(lemmatizer.lemmatize(("walking"), pos=wordnet.VERB))
    print(lemmatizer.lemmatize("going"))
    print(lemmatizer.lemmatize(("going"), pos=wordnet.VERB))
    print(lemmatizer.lemmatize(("ran"), pos=wordnet.VERB))

def main() -> None:
    # Stemmer()
    Lemmat()

if __name__ == '__main__':
    # On First run please run the following commands below
    # import nltk
    # nltk.download('wordnet')

    main()