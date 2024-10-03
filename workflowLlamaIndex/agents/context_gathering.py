# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
# from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain.storage import LocalFileStore
# from langchain.embeddings import CacheBackedEmbeddings
# from langchain_qdrant import QdrantVectorStore
# from langchain.docstore.document import Document
# from langchain.retrievers import VectorStoreRetriever
# from typing import List
# import uuid


# def context_gathering(
#         k: int = 5,
#         use_qdrant: bool = False,
#         qdrant_client: QdrantClient = None,
#         collection_name: str = None,
#         docs: List[Document]) -> :

#      # Create a Qdrant vector store with cache backed embeddings
#     collection_name = f"pdf_to_parse_{uuid.uuid4()}"
#     client = QdrantClient(":memory:")
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
#     )
#     core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#     store = LocalFileStore("./cache/")
#     # Caching: Using CacheBackedEmbeddings improves performance by storing and reusing
#     # previously computed embeddings, reducing API calls and processing time.
#     cached_embedder = CacheBackedEmbeddings.from_bytes_store(
#         core_embeddings, store, namespace=core_embeddings.model
#     )
#     vectorstore = QdrantVectorStore(
#         client=client, 
#         collection_name=collection_name,
#         embedding=cached_embedder)
#     vectorstore.add_documents(docs)
#     retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

#     return retriever
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.docstore.document import Document
# from langchain.retrievers import VectorStoreRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from typing import List, Union
import uuid
from uuid import uuid4
import numpy as np
import faiss

def context_gathering(
        docs: List[Document],
        k: int = 5,
        use_qdrant: bool = True,
        qdrant_client: QdrantClient = None,
        collection_name: str = None) -> Union[QdrantVectorStore, FAISS]:
    

    if use_qdrant:
        collection_name = collection_name or f"pdf_to_parse_{uuid.uuid4()}"
        print(collection_name)
        client = qdrant_client or QdrantClient(":memory:")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        store = LocalFileStore("./cache/")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            core_embeddings, store, namespace=core_embeddings.model
        )
        vectorstore = QdrantVectorStore(
            client=client, 
            collection_name=collection_name,
            embedding=cached_embedder)
        vectorstore.add_documents(docs)
        # return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})

    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        document_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]
        
        index = faiss.IndexFlatL2(len(document_embeddings[0]))

        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vectorstore.add_documents(documents=docs, ids=uuids)

    return  vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})

    
        

        