import requests
import logging
import json
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
            response = requests.post(url, json={"file_path": file_path})
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

def display_menu():
    print("\n=== OSM LLM Client Menu ===")
    print("1. Load OSM Data")
    print("2. Make a Query")
    print("3. Exit")
    return input("Choose an option (1-3): ")

def main():
    client = OSMClient()
    
    while True:
        choice = display_menu()
        
        try:
            if choice == "1":
                file_path = input("Enter OSM file path (e.g., ./data/map.osm.pbf): ")
                result = client.load_osm_data(file_path)
                logger.info(f"Load result: {result}")
                
            elif choice == "2":
                query = input("Enter your query (e.g., 'Find all restaurants near the main station'): ")
                n_results = int(input("Enter number of results (default 5): ") or "5")
                result = client.query(query, n_results)
                print("\nResponse:", result['response'])
                print("\nContext Features:")
                for i, feature in enumerate(result['context_features'], 1):
                    print(f"\n{i}. {json.dumps(feature, indent=2)}")
                
            elif choice == "3":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
        except Exception as e:
            logger.error(f"Error in operation: {e}")
            print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
