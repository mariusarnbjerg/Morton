"""
FastAPI server for Morton application.
Main entry point for the API.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from app.api.routes import router
from app.api.dependencies import get_orchestrator

# Load environment variables
load_dotenv()


# ============================================================================
# Lifespan: Startup and shutdown events
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on app startup and shutdown.

    Startup: Initialize orchestrator, connect to databases, etc.
    Shutdown: Cleanup, close connections, etc.
    """
    # Startup
    print("\n" + "="*60)
    print("🚀 Starting API...")
    print("="*60)

    # Initialize the orchestrator (loads models, etc.)
    app.state.orchestrator = get_orchestrator()

    print("✅ API is ready!")
    print("📖 Docs available at: http://localhost:8000/docs")
    print("="*60 + "\n")

    yield  # API runs here

    # Shutdown
    print("\n" + "="*60)
    print("👋 Shutting down API...")
    print("="*60 + "\n")


# ============================================================================
# Create FastAPI app
# ============================================================================

app = FastAPI(
    title="Project API",
    description="AI-assisted pre-anesthesia questionnaire system",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# CORS: Allow frontend to access API
# ============================================================================

# Get allowed origins from .env
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
allowed_origins = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Which frontend URLs can access this API
    allow_credentials=True,          # Allow cookies
    allow_methods=["*"],             # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],             # Allow all headers
)


# ============================================================================
# Include routes
# ============================================================================

# All routes from routes.py will be prefixed with /api/v1
app.include_router(router, prefix="/api/v1")


# ============================================================================
# Root endpoints
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - shows basic API info.

    Visit: http://localhost:8000/
    """
    return {
        "message": "Project API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint - verify API is running.

    Visit: http://localhost:8000/health
    """
    return {
        "status": "healthy",
        "service": "project-api"
    }


# ============================================================================
# Run the server (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get config from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"

    print(f"Starting server at {host}:{port}")

    uvicorn.run(
        "app.api.server:app",  # Path to the app
        host=host,
        port=port,
        reload=reload,         # Auto-reload on code changes
        log_level="info"
    )