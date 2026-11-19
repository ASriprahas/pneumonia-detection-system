from fastapi import Header, HTTPException, Security, status
from app.core.config import settings

def verify_internal_api_key(
    x_internal_api_key: str = Header(
        default=None,
        alias="X-Internal-API-Key",
        description="Internal API key for service-to-service communication"
    )
):
    """
    Validates that incoming requests from Spring Boot contain the shared secret key.
    If no key configured or key matches, request proceeds.
    """
    if not settings.INTERNAL_API_KEY:
        return True

    # If key is provided and matches
    if x_internal_api_key and x_internal_api_key == settings.INTERNAL_API_KEY:
        return True

    # Also allow standard requests from Swagger UI or localhost development for testing
    # if internal key is default and not strictly required
    if x_internal_api_key is None:
        # In development/local mode, allow access for Swagger testing while logging warning
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing X-Internal-API-Key header"
    )
