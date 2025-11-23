"""
Production Deployment for Unified Morphology Model
===================================================

FastAPI server for serving the unified Qwen model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import logging
from inference import UnifiedMorphologyModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None

# Create FastAPI app
app = FastAPI(
    title="Kazakh Unified Morphology API",
    description="REST API for complete Kazakh morphological analysis using unified Qwen model",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    """Request model for prediction"""
    word: str
    pos_tag: str


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    result: Dict
    word: str
    pos_tag: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    message: str


def init_model(model_path: str):
    """Initialize the model"""
    global model

    logger.info(f"Initializing model from: {model_path}")

    try:
        model = UnifiedMorphologyModel(model_path)
        model.load_model()
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""

    model_loaded = model is not None

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        message="Model is ready" if model_loaded else "Model not loaded"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict complete morphology for a word.

    Args:
        request: PredictionRequest with word and POS tag

    Returns:
        PredictionResponse with complete morphological analysis
    """

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = model.predict(
            word=request.word,
            pos_tag=request.pos_tag
        )

        return PredictionResponse(
            result=result,
            word=request.word,
            pos_tag=request.pos_tag
        )

    except Exception as e:
        logger.error(f"Prediction error for {request.word}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Kazakh Unified Morphology API",
        "version": "1.0.0",
        "model": "Qwen2.5-3B (Unified)",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Morphology prediction",
            "/docs": "API documentation"
        }
    }


def run_server(model_path: str, host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""

    # Initialize model
    init_model(model_path)

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    run_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port
    )
