from helpers.pinecone.get_vector_store import get_vector_store
from helpers.pinecone.lost_in_the_middle import lost_in_the_middle


def get_relevant_documents(query: str):
    vector_store = get_vector_store()
    relevant_documents = vector_store.similarity_search(query=query, k=8)

    return lost_in_the_middle(documents=relevant_documents)
