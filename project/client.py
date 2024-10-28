import requests
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def load_osm_data(self, file_path: str) -> Dict:
        """Load OSM data from a file into the vector store"""
        url = f"{self.base_url}/load_osm"
        try:
            response = requests.post(url, json=file_path)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error loading OSM data: {e}")
            raise
            
    def query(self, query_text: str, n_results: Optional[int] = 5) -> Dict:
        """Query the system with natural language"""
        url = f"{self.base_url}/query"
        payload = {
            "text": query_text,
            "n_results": n_results
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying system: {e}")
            raise

def main():
    client = OSMClient()
    
    # Example usage
    try:
        # Load some OSM data
        result = client.load_osm_data("path/to/your/osm/file.pbf")
        logger.info(f"Load result: {result}")
        
        # Make a query
        query = "Find all restaurants near the main train station"
        result = client.query(query)
        logger.info(f"Query response: {result['response']}")
        logger.info(f"Context features: {result['context_features']}")
        
    except Exception as e:
        logger.error(f"Error in client operations: {e}")

if __name__ == "__main__":
    main()
