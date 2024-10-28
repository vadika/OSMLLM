from fastapi import FastAPI, HTTPException
import uvicorn
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

# Initialize components
logger.info("Initializing VectorStore...")
vector_store = VectorStore()
logger.info("Initializing LLM Interface...")
llm_interface = OSMQueryInterface()
logger.info("Application components initialized successfully")

@app.post("/load_osm")
async def load_osm_data(file_path: str):
    try:
        features = parse_osm_file(file_path)
        vector_store.add_features(features)
        return {"message": f"Loaded {len(features)} features"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_osm(query: Query):
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

if __name__ == "__main__":
    logger.info("Starting FastAPI application...")
    uvicorn.run(
        "project.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["**/pytorch/**", "**/ittapi/**"]
    )
