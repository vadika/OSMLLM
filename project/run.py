import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting FastAPI application...")
    uvicorn.run(
        "project.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["**/pytorch/**", "**/ittapi/**"],
        log_level="info"
    )
