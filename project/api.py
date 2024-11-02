import os
from fastapi import FastAPI, HTTPException, Depends
import logging
from pydantic import BaseModel
from typing import List, Optional, Dict
from project.osm_parser import parse_osm_file
from project.vector_store import VectorStore
from project.llm_interface import OSMQueryInterface

# Initialize FastAPI app
app = FastAPI(title="OSM LLM API")

# Configure logging
logger = logging.getLogger(__name__)

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
        
        # Validate file exists and is accessible
        if not os.path.exists(request.file_path):
            logger.error(f"File not found: {request.file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        if not os.access(request.file_path, os.R_OK):
            logger.error(f"File not readable: {request.file_path}")
            raise HTTPException(status_code=403, detail=f"File not readable: {request.file_path}")
        
        def process_batch(batch: List[Dict]):
            try:
                vector_store.add_features(batch)
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                raise
        
        logger.info(f"Starting to parse OSM file: {request.file_path}")
        total_features = parse_osm_file(request.file_path, batch_callback=process_batch)
        logger.info(f"Successfully loaded {total_features} features")
        
        return {"message": f"Loaded {total_features} features"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error loading OSM data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading OSM data: {str(e)}"
        )

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
