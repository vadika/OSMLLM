import osmium
import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

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
    """Parse OSM protobuf file and return list of features"""
    logger.info(f"Starting to parse OSM file: {file_path}")
    handler = OSMHandler()
    handler.apply_file(file_path)
    logger.info(f"Finished parsing OSM file. Found {len(handler.features)} features")
    return handler.features
