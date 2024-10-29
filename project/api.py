from fastapi import FastAPI, HTTPException, Depends
import logging
from pydantic import BaseModel
from typing import List, Optional
from project.osm_parser import parse_osm_file
from project.vector_store import VectorStore
from project.llm_interface import OSMQueryInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

class Query(BaseModel):
    text: str
    n_results: Optional[int] = 5

# Initialize components once at module level
vector_store = VectorStore()
llm_interface = OSMQueryInterface()

def get_vector_store():
    return vector_store

def get_llm_interface():
    return llm_interface

class OSMLoadRequest(BaseModel):
    file_path: str

@app.post("/load_osm")
async def load_osm_data(request: OSMLoadRequest, vector_store: VectorStore = Depends(get_vector_store)):
    try:
        logger.info(f"Received load_osm request: {request}")
        features = parse_osm_file(request.file_path)
        vector_store.add_features(features)
        return {"message": f"Loaded {len(features)} features"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_osm(
    query: Query,
    vector_store: VectorStore = Depends(get_vector_store),
    llm_interface: OSMQueryInterface = Depends(get_llm_interface)
):
    try:
        # First get relevant features from vector store
        context_features = vector_store.query(query.text, query.n_results)
        # Then process with LLM
        response = llm_interface.process_query(query.text, context_features)
        return {
            "response": response,
            "context_features": context_features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
