from datasets import load_dataset
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

dataset  = load_dataset("bilgeyucel/seven-wonders", split="train")

docs  = [Document(content= doc["content"], meta = doc["meta"]) for doc in dataset]

document_store = InMemoryDocumentStore()
document_store.write_documents(docs)