
"""NUMPA v2 — Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from app.api.v1.router import api_router


# Initialize the NUMPA backend.
app = FastAPI(
    title="NUMPA API",
    description="Data Cleaning and ML Readiness Platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Allow requests from the local Next.js frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# Explicitly register the existing API v1 routes.
#
# The current environment creates an _IncludedRouter object
# instead of expanding app.include_router() into endpoints.
# Registering actual APIRoute objects avoids that problem.
for route in api_router.routes:
    if not isinstance(route, APIRoute):
        continue

    app.add_api_route(
        path=f"/api/v1{route.path}",
        endpoint=route.endpoint,
        methods=route.methods,
        response_model=route.response_model,
        status_code=route.status_code,
        tags=route.tags,
        dependencies=route.dependencies,
        summary=route.summary,
        description=route.description,
        response_description=route.response_description,
        responses=route.responses,
        deprecated=route.deprecated,
        operation_id=route.operation_id,
        name=route.name,
        include_in_schema=route.include_in_schema,
        response_class=route.response_class,
    )


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    """Return basic information about NUMPA."""

    return {
        "application": "NUMPA API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/api/v1/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Check whether the backend is running."""

    return {
        "status": "healthy",
        "service": "numpa-api",
    }
