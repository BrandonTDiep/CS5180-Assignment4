#-------------------------------------------------------------------------
# AUTHOR: Brandon Diep
# FILENAME: assignment4.py
# SPECIFICATION: This program will
# FOR: CS 5180- Assignment #4
# TIME SPENT: 1 day
#-----------------------------------------------------------*/


from collections import defaultdict
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def connectDataBase():
    # Creating a database connection object using psycopg2
    DB_NAME = "search"
    DB_HOST = "localhost"
    DB_PORT = 27017
    try:
        client = MongoClient(host=DB_HOST, port=DB_PORT)
        db = client[DB_NAME]
        return db
    except:
        print("Database not connected successfully")



def search_engine():
    # Connecting to the database
    db = connectDataBase()
    inverted_index_collection = db["inverted_index"]
    docs_collection = db["docs"]

    documents = [
        "After the medication, headache and nausea were reported by the patient.",
        "The patient reported nausea and dizziness caused by the medication.",
        "Headache and dizziness are common effects of this medication.",
        "The medication caused a headache and nausea, but no dizziness was reported."
    ]

    queries = [
        "nausea and dizziness",
        "effects",
        "nausea was reported",
        "dizziness",
        "the medication"
    ]

    # stores the documents ids in order in which they are inserted
    doc_ids = []

    docs_collection.delete_many({})  

    # store the doc in documents collection
    for doc in documents:
        docs_collection.insert_one({
                "content": doc
        })

    # store the doc ids in docs_ids
    for doc in docs_collection.find({}, {"_id": 1, "content": 1}):
        doc_ids.append((doc['_id'])) 
    
        
    # tfidfvecotrizer is a mutable object like list 
    tfidfvectorizer = TfidfVectorizer(analyzer= 'word', stop_words='english', ngram_range=(1, 3))

    # sparse matrix representation Ex. (0,31) 0.338 | (0,20) 0.338, so to get tf-idf do tfidf_matrix[#,#]
    tfidf_matrix = tfidfvectorizer.fit_transform(documents)

    # gets the terms sorted in order
    terms = tfidfvectorizer.get_feature_names_out()
    # print(tfidfvectorizer.vocabulary_)

    # clear existing data
    inverted_index_collection.delete_many({})  
    
    # get the position and term 
    for term_idx, term in enumerate(terms):
        docs_list = []

        # tfidf_matrix.shape[0]:  document size. 
        for doc_idx in range(tfidf_matrix.shape[0]):
            score = tfidf_matrix[doc_idx, term_idx]
            if score > 0:  
                docs_list.append({"doc": doc_ids[doc_idx],"tfidf_score": score})
        
        inverted_index_collection.insert_one({
                "term": term, 
                "position": term_idx,
                "documents": docs_list
        })


    # convert TF-IDF matrix to document vectors and store them
    # an array of tf-idf vectors corresponding to each document
    document_vectors = tfidf_matrix.toarray()

    # Map/Combine document IDs to document vectors Key(ObjectId('674c1d96accaf10c5385e683)) : array([tf-idf])
    doc_id_to_vector = {doc_id: vector for doc_id, vector in zip(doc_ids, document_vectors)}
    # tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    # print(pd.DataFrame(data = tfidf_matrix.toarray(),index = ['Doc1','Doc2','Doc3','Doc4'],columns = tfidf_tokens))

    # loop thru each query in queries
    for query in queries:
        query = query.strip()
        # creates a tf-idf vector for query for example everything is 0, except at positon 7,26 27 for 'dizziness' 'nausea' 'nausea dizziness'
        query_vector = tfidfvectorizer.transform([query]).toarray()
        # [0] gets the array and inverse transform gets the non zero indicies on query vector
        # Ex. ['dizziness' 'nausea' 'nausea dizziness'] from nausea and dizziness
        query_terms = tfidfvectorizer.inverse_transform(query_vector)[0]  

        # find documents relevant to the query
        relevant_docs_ids = set()
        for term in query_terms:
            term_entry = inverted_index_collection.find_one({"term": term})
            if term_entry:
                for doc in term_entry["documents"]:
                    relevant_docs_ids.add(doc["doc"])

        results = []
        for doc_id in relevant_docs_ids:
            # identify which document vector to target based on doc_id
            doc_vector = doc_id_to_vector[doc_id]
            similarity = cosine_similarity(query_vector, doc_vector.reshape(1, -1))[0][0]
            # retrieve document content
            doc_entry = docs_collection.find_one({"_id": doc_id})
            if doc_entry:
                doc_content = doc_entry["content"]
                results.append({"content": doc_content, "similarity": similarity})

        # sort results by similarity
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        print(f"Query: {query}")
        for result in results:
            print(f"'{result["content"]}', {result["similarity"]}")
        print()





if __name__ == '__main__':
    search_engine()



 