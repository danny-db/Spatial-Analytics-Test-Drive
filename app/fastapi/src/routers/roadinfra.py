import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_async_db
from ..models.roadinfra import (
    CursorPaginationInfo,
    RoadInfra,
    RoadInfraCount,
    RoadInfraListCursorResponse,
    RoadInfraListResponse,
    RoadInfraRead,
    RoadInfraSample,
    RoadInfraStatusUpdate,
    RoadInfraStatusUpdateResponse,
    PaginationInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/roadinfra", tags=["roadinfra"])


@router.get("/count", response_model=RoadInfraCount, summary="Get total road infra count")
async def get_road_infra_count(db: AsyncSession = Depends(get_async_db)):
    """Get the total number of road infrastructure in the database."""
    try:
        stmt = select(func.count(RoadInfra.UFI))
        result = await db.execute(stmt)
        count = result.scalar()
        return RoadInfraCount(total_roadinfra=count)
    except Exception as e:
        logger.error(f"Error getting road infra count: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve road infra count")


@router.get("/sample", response_model=RoadInfraSample, summary="Get 5 random RoadInfra keys")
async def get_sample_roadinfra(db: AsyncSession = Depends(get_async_db)):
    """Get 5 random RoadInfra keys for testing."""
    try:
        stmt = select(UFI).limit(5)
        result = await db.execute(stmt)
        roadinfra_keys = result.scalars().all()
        return RoadInfraSample(sample_roadinfra_keys=roadinfra_keys)
    except Exception as e:
        logger.error(f"Error getting sample roadinfra: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sample roadinfra")


@router.get(
    "/pages",
    response_model=RoadInfraListResponse,
    summary="Get RoadInfra with page-based pagination",
)
async def get_roadinfra_by_page(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(
        100, ge=1, le=1000, description="Number of records per page (max 1000)"
    ),
    include_count: bool = Query(
        True, description="Include total count for pagination info"
    ),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get RoadInfra using traditional page-based pagination.

    **Best for:**
    - Small to medium datasets
    - Building traditional pagination UI with page numbers
    - When users need to jump to specific pages

    **Usage:**
    - `/roadinfra/pages?page=1&page_size=100`
    - `/roadinfra/pages?page=5&page_size=50&include_count=false`
    """
    try:
        if include_count:
            count_stmt = select(func.count(UFI))
            count_result = await db.execute(count_stmt)
            total_count = count_result.scalar()
            total_pages = (total_count + page_size - 1) // page_size
        else:
            total_count = -1
            total_pages = -1

        offset = (page - 1) * page_size
        stmt = (
            select(
                RoadInfra.UFI,
                RoadInfra.PFI,
                RoadInfra.FTYPE_CODE,
                RoadInfra.geom_4326,
                RoadInfra.h3,
            )
            .order_by(RoadInfra.UFI)
            .offset(offset)
            .limit(page_size + 1)  # Get one extra to check has_next
        )

        result = await db.execute(stmt)
        all_roadinfra = result.all()

        has_next = len(all_roadinfra) > page_size
        roadinfras_data = all_roadinfra[:page_size]
        has_previous = page > 1

        roadinfras = [
            RoadInfraRead(
                UFI=row[0],
                PFI=row[1],
                FTYPE_CODE=row[2],
                geom_4326=row[3],
                h3=row[4],
            )
            for row in roadinfras_data
        ]

        next_cursor = roadinfras[-1].UFI if roadinfras and has_next else None
        previous_cursor = (
            roadinfras[0].UFI - page_size if roadinfras and has_previous else None
        )

        pagination_info = PaginationInfo(
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            total_count=total_count,
            has_next=has_next,
            has_previous=has_previous,
            next_cursor=next_cursor,
            previous_cursor=max(0, previous_cursor) if previous_cursor else None,
        )

        return RoadInfraListResponse(roadinfras=roadinfras, pagination=pagination_info)

    except Exception as e:
        logger.error(f"Error getting page-based road infra: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve road infra")


@router.get(
    "/stream",
    response_model=RoadInfraListCursorResponse,
    summary="Get road infras with cursor-based pagination",
)
async def get_roadinfra_by_cursor(
    cursor: int = Query(
        0, ge=0, description="Start after this road infra key (0 for beginning)"
    ),
    page_size: int = Query(
        100, ge=1, le=1000, description="Number of records to fetch (max 1000)"
    ),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get road infra using efficient cursor-based pagination.

    **Best for:**
    - Large datasets (millions of records)
    - High-performance applications
    - Infinite scroll UIs
    - Real-time data feeds

    **Usage:**
    - First page: `/roadinfra/stream?cursor=0&page_size=100`
    - Next page: `/roadinfra/stream?cursor=100&page_size=100`
    - Jump to key: `/roadinfra/stream?cursor=12345&page_size=100` (shows records after key 12345)
    """
    try:
        stmt = (
            select(
                RoadInfra.UFI,
                RoadInfra.PFI,
                RoadInfra.FTYPE_CODE,
                RoadInfra.geom_4326,
                RoadInfra.h3,
            )
            .where(RoadInfra.UFI > cursor)
            .order_by(RoadInfra.UFI)
            .limit(page_size + 1)  # Get one extra to check has_next
        )

        result = await db.execute(stmt)
        all_infras = result.all()

        has_next = len(all_infras) > page_size
        infras_data = all_infras[:page_size]
        has_previous = cursor > 0

        infras = [
            RoadInfraRead(
                UFI=row[0],
                PFI=row[1],
                FTYPE_CODE=row[2],
                geom_4326=row[3],
                h3=row[4],
            )
            for row in roadinfras_data
        ]

        next_cursor = roadinfras[-1].UFI if roadinfras and has_next else None
        previous_cursor = max(0, cursor - page_size) if has_previous else None

        pagination_info = CursorPaginationInfo(
            page_size=page_size,
            has_next=has_next,
            has_previous=has_previous,
            next_cursor=next_cursor,
            previous_cursor=previous_cursor,
        )

        return RoadInfraListCursorResponse(roadinfras=roadinfras, pagination=pagination_info)

    except Exception as e:
        logger.error(f"Error getting cursor-based roadinfras: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve roadinfras")


@router.get("/{roadinfra_key}", response_model=RoadInfraRead, summary="Get an road infra by its key")
async def read_roadinfra(
    roadinfra_key: int, response: Response, db: AsyncSession = Depends(get_async_db)
):
    """
    Fetch a single road infra by its key, returning all road infra fields.
    """
    try:
        if roadinfra_key <= 0:
            raise HTTPException(status_code=400, detail="Invalid road infra key provided")

        stmt = select(RoadInfra).where(RoadInfra.UFI == roadinfra_key)
        result = await db.execute(stmt)
        roadinfra = result.scalars().first()

        if not roadinfra:
            logger.info(f"Road Infra not found: {roadinfra_key}")
            raise HTTPException(
                status_code=404, detail=f"Road Infra with key '{roadinfra_key}' not found"
            )

        return roadinfra

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching road infra {roadinfra_key}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")


@router.post(
    "/{roadinfra_key}/status",
    response_model=RoadInfraStatusUpdateResponse,
    summary="Update road infra status",
)
async def update_roadinfra_status(
    roadinfra_key: int,
    status_data: RoadInfraStatusUpdate,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Update the status for a specific road infra.

    - **roadinfra_key**: The key of the road infra to update
    - **roadinfra_status**: The new status to assign
    """
    try:
        if roadinfra_key <= 0:
            raise HTTPException(status_code=400, detail="Invalid road infra key provided")

        check_stmt = select(RoadInfra).where(RoadInfra.UFI == roadinfra_key)
        check_result = await db.execute(check_stmt)
        existing_roadinfra = check_result.scalars().first()

        if not existing_roadinfra:
            logger.info(f"Road Infra not found for update: {roadinfra_key}")
            raise HTTPException(
                status_code=404, detail=f"Road Infra with key '{roadinfra_key}' not found"
            )

        existing_roadinfra.roadinfrastatus = status_data.roadinfrastatus
        await db.commit()
        await db.refresh(existing_roadinfra)

        logger.info(
            f"Successfully updated road infra {roadinfra_key} status to {status_data.roadinfra_status}"
        )

        return RoadInfraStatusUpdateResponse(
            UFI=roadinfra_key,
            roadinfrastatus=status_data.roadinfrastatus,
            message="Road Infra status updated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating status for road infra {roadinfra_key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update road infra status")
