import osmium
import json
import logging
import time
import multiprocessing as mp
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    return Path(file_path).stat().st_size

def process_chunk(chunk_info: tuple) -> List[List[Dict]]:
    """Process a chunk of the OSM file"""
    try:
        file_path, chunk_id = chunk_info
        process_name = mp.current_process().name
        logger.info(f"Process {process_name} starting chunk {chunk_id}")
        
        handler = OSMHandler()
        start_time = time.time()
        # Use default index type which is always available
        handler.apply_file(file_path, locations=True)
            
        processing_time = time.time() - start_time
        
        feature_count = len(handler.features)
        logger.info(f"Process {process_name} completed chunk {chunk_id}: "
                   f"Found {feature_count} features in {processing_time:.2f} seconds")
        
        # Return features in smaller batches
        batch_size = 10000
        batches = []
        for i in range(0, len(handler.features), batch_size):
            batches.append(handler.features[i:i + batch_size])
        
        # Clear handler features to free memory
        handler.features = []
        return batches
    except Exception as e:
        logger.error(f"Process {mp.current_process().name} failed on chunk {chunk_id}: {str(e)}")
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

def parse_osm_file(file_path: str, batch_callback=None) -> List[Dict]:
    """Parse OSM protobuf file and process features in batches"""
    logger.info(f"Starting to parse OSM file: {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"OSM file not found: {file_path}")
    
    # Setup parallel processing
    try:
        # Use 75% of available CPUs to avoid overwhelming the system
        cpu_count = max(1, int(mp.cpu_count() * 0.75))
        file_size = get_file_size(file_path)
        logger.info(f"Initializing parallel processing with {cpu_count} processes")
        logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
        # Create chunks based on optimized process count
        chunks = [(file_path, i) for i in range(cpu_count)]
        
        start_time = time.time()
    except Exception as e:
        raise RuntimeError(f"Failed to setup parallel processing: {str(e)}")
    
    logger.info(f"Starting parallel processing with {cpu_count} chunks")
    
    # Process chunks in parallel with progress bar
    try:
        seen_features = set()
        total_features = 0
        
        with mp.Pool(processes=cpu_count) as pool:
            for batches in pool.imap_unordered(process_chunk, chunks):
                for batch in batches:
                    # Deduplicate features in each batch
                    unique_batch = []
                    for feature in batch:
                        feature_key = (feature['type'], feature['id'])
                        if feature_key not in seen_features:
                            seen_features.add(feature_key)
                            unique_batch.append(feature)
                    
                    if unique_batch:
                        total_features += len(unique_batch)
                        if batch_callback:
                            batch_callback(unique_batch)
                        
                    # Clear batch to free memory
                    batch.clear()
                    
    except Exception as e:
        raise RuntimeError(f"Failed during parallel processing: {str(e)}")
    
    total_time = time.time() - start_time
    logger.info(f"Finished parsing OSM file in {total_time:.2f} seconds")
    logger.info(f"Found {total_features} unique features")
    logger.info(f"Processing rate: {total_features/total_time:.2f} features/second")
    
    return total_features
