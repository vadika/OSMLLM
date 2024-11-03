import chromadb
from chromadb.config import Settings
from typing import List, Dict
import json
import logging
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        # Initialize embedding model with GPU support
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection("osm_features")
        
    def add_features(self, features: List[Dict], batch_size: int = 1000):
        """Add OSM features to vector store in batches"""
        logger.info(f"Preparing to add features to vector store")
        
        total = 0
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            
            documents = []
            ids = []
            metadatas = []
            
            for feature in batch:
                documents.append(json.dumps(feature))
                # Create unique ID combining feature type and ID
                unique_id = f"{feature['type']}_{feature['id']}"
                ids.append(unique_id)
                metadatas.append({
                    'type': feature['type'],
                    'tags': json.dumps(feature['tags'])
                })
            
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            
            total += len(batch)
            logger.info(f"Added batch of {len(batch)} features. Total: {total}")
            
            # Clear lists to free memory
            documents.clear()
            ids.clear()
            metadatas.clear()
            
        logger.info(f"Successfully added {total} features to vector store")
    
    def query(self, query_text: str, n_results: int = 5):
        """Query the vector store"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return [json.loads(doc) for doc in results['documents'][0]]
