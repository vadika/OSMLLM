import osmium
import json
import logging
import multiprocessing as mp
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    return Path(file_path).stat().st_size

def process_chunk(chunk_info: tuple) -> List[Dict]:
    """Process a chunk of the OSM file"""
    try:
        file_path, start_pos, size = chunk_info
        handler = OSMHandler()
        handler.apply_file(file_path, locations=True)  # Remove start_pos and size parameters
        return handler.features
    except Exception as e:
        logger.error(f"Error processing chunk at position {start_pos}: {str(e)}")
        return []

class OSMHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.features = []
        
    def node(self, n):
        if len(n.tags) > 0:
            feature = {
                'type': 'node',
                'id': n.id,
                'location': [n.location.lat, n.location.lon],
                'tags': dict(n.tags)
            }
            self.features.append(feature)
    
    def way(self, w):
        if len(w.tags) > 0:
            feature = {
                'type': 'way',
                'id': w.id,
                'nodes': [n.ref for n in w.nodes],
                'tags': dict(w.tags)
            }
            self.features.append(feature)

def parse_osm_file(file_path: str) -> List[Dict]:
    """Parse OSM protobuf file and return list of features using all available CPUs"""
    logger.info(f"Starting to parse OSM file: {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"OSM file not found: {file_path}")
        
    # Process in parallel using process pool
    try:
        cpu_count = mp.cpu_count()
        chunks = [(file_path,) for _ in range(cpu_count)]  # Each process will process the full file
    except Exception as e:
        raise RuntimeError(f"Failed to setup parallel processing: {str(e)}")
        
    logger.info(f"Processing file in {cpu_count} parallel chunks")
    
    # Process chunks in parallel with progress bar
    try:
        with mp.Pool(processes=cpu_count) as pool:
            chunk_results = list(tqdm(
                pool.imap(process_chunk, chunks),
                total=cpu_count,
                desc="Parsing OSM file",
                unit="chunk"
            ))
    except Exception as e:
        raise RuntimeError(f"Failed during parallel processing: {str(e)}")
    
    # Combine results
    features = []
    for chunk in chunk_results:
        features.extend(chunk)
    
    # Remove potential duplicates based on type and id
    unique_features = {(f['type'], f['id']): f for f in features}.values()
    features = list(unique_features)
    
    logger.info(f"Finished parsing OSM file. Found {len(features)} unique features")
    return features
