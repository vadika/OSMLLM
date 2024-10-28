from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from project.osm_parser import parse_osm_file
from project.vector_store import VectorStore
from project.llm_interface import OSMQueryInterface

app = FastAPI()
vector_store = VectorStore()
llm_interface = OSMQueryInterface()

class Query(BaseModel):
    text: str
    n_results: Optional[int] = 5

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
