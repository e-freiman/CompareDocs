from sklearn.feature_extraction.text import TfidfVectorizer
import textract
import os
from sklearn.metrics.pairwise import linear_kernel

print('\n\n### PROCESSING ###')

filter = []

with open('filter.txt','r') as file:
    for line in file:  
        for word in line.split():
            filter.append(word)

traget_doc = ''
for filename in os.listdir('target_docs'):
    try:
        traget_doc += textract.process('target_docs/' + filename).decode('utf-8') + ' '
        print("Processed target document: " + filename)
    except:
        print("Traget document: " + filename + ' skipped, couldn\'t process')

docs = [traget_doc]
docs_names = ['TARGET']

for filename in os.listdir('unknown_docs'):
    try:
        text = textract.process('unknown_docs/' + filename).decode('utf-8')
        words = text.split()
        skip = False
        for keyword in filter:
            found = False
            for word in words:
                if keyword == word:
                    found = True
                    break
            if not found:                
                print(filename + ' doesn\'t have "' + word + '", skipped')
                skip = True

        if not skip:
            docs.append(text)
            docs_names.append(filename)
            print("Processed " + filename)
    except:
        print(filename + ' skipped, couldn\'t process')

vectorizer = TfidfVectorizer(use_idf = False)
vectors = vectorizer.fit_transform(docs)

cosine_similarities = linear_kernel(vectors[0:1], vectors).flatten()
related_docs_indices = cosine_similarities.argsort()

print('\n\n### SIMILARITY ###')

for idx in reversed(related_docs_indices):
    print(docs_names[idx] + '\t' + str(cosine_similarities[idx]))

print('\n\n### FEATURES ###')

target_vector_data_indices = vectors[0:1].data.argsort()
features = vectorizer.get_feature_names()
for idx in reversed(target_vector_data_indices):
    print(features[vectors[0:1].indices[idx]] + '\t' + str(vectors[0:1].data[idx]))