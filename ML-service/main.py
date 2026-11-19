from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.core.config import settings
from app.core.logging import get_logger
from app.services.inference_service import inference_service
from app.api.v1.router import api_router

logger = get_logger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models
    logger.info("Initializing ML models for Pneumonia Detection Engine...")
    inference_service.load_models()
    logger.info("ML Engine successfully booted and models loaded.")
    yield
    # Shutdown
    logger.info("Shutting down ML Engine...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Microservice providing VGG16, ResNet50, and Custom PneumoniaNet inference with Grad-CAM visualization for chest X-ray scans.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API V1 routes
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=False)
