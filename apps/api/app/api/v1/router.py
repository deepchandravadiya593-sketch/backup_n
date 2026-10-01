
"""NUMPA v2 - Central API v1 router.

Registers dataset, cleaning and ML readiness endpoints.

Routes are copied explicitly to preserve compatibility with
the manual route-registration approach used by app/main.py.
"""

from fastapi import APIRouter
from fastapi.routing import APIRoute

from app.api.v1.endpoints import (
    cleaning,
    datasets,
    readiness,
)


# app/main.py mounts this router under /api/v1.
api_router = APIRouter()


def register_routes(
    source_router: APIRouter,
) -> None:
    """Copy API routes from a feature router.

    Preserve endpoint metadata, request validation,
    response schemas and dependency declarations.
    """

    for route in source_router.routes:
        # Only register HTTP API routes. This avoids
        # accidentally treating mounted apps or other
        # route types as ordinary API endpoints.
        if not isinstance(route, APIRoute):
            continue

        api_router.add_api_route(
            path=route.path,
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


# Keep the existing feature routes and add ML readiness.
register_routes(datasets.router)
register_routes(cleaning.router)
register_routes(readiness.router)
