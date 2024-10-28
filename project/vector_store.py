import chromadb
from chromadb.config import Settings
from typing import List, Dict
import json

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection("osm_features")
        
    def add_features(self, features: List[Dict]):
        """Add OSM features to vector store"""
        documents = [json.dumps(feature) for feature in features]
        ids = [str(i) for i in range(len(documents))]
        metadatas = [
            {
                'type': feature['type'],
                'tags': json.dumps(feature['tags'])
            } 
            for feature in features
        ]
        
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
    
    def query(self, query_text: str, n_results: int = 5):
        """Query the vector store"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return [json.loads(doc) for doc in results['documents'][0]]
