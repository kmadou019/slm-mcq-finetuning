#!/usr/bin/env python3

"""
FastAPI Main Application - MCQ Evaluation Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import auth, mcq, validations, admin
from database import init_db

# Create FastAPI app
app = FastAPI(
    title="MCQ Evaluation API",
    description="Backend API for MCQ evaluation application",
    version="1.0.0"
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables"""
    init_db()
    print("🚀 Application started - Database initialized")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",  # Angular dev server
        "http://localhost:3000",  # Alternative frontend port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api")
app.include_router(mcq.router, prefix="/api")
app.include_router(validations.router, prefix="/api")
app.include_router(admin.router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint - Health check
    """
    return {
        "message": "MCQ Evaluation API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
