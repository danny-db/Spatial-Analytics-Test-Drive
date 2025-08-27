import os
from datetime import date
from decimal import Decimal
from typing import Optional

from sqlmodel import Field, SQLModel


class RoadInfraBase(SQLModel):
    UFI: int
    PFI: int
    FTYPE_CODE: str
    geom_4326: str
    h3: int


class RoadInfra(RoadInfraBase, table=True):
    __tablename__ = os.getenv("DEFAULT_POSTGRES_TABLE", "silver_road_infra_sync")
    __table_args__ = {"schema": os.getenv("DEFAULT_POSTGRES_SCHEMA", "vicmap_schema")}
    UFI: Optional[int] = Field(default=None, primary_key=True)


class RoadInfraRead(RoadInfraBase):
    UFI: int


class RoadInfraCount(SQLModel):
    total_roadinfra: int


class RoadInfraSample(SQLModel):
    sample_roadinfra_keys: list[int]


class RoadInfraStatusUpdate(SQLModel):
    roadinfrastatus: str


class RoadInfraStatusUpdateResponse(SQLModel):
    UFI: int
    roadinfrastatus: str
    message: str


class RoadInfraListResponse(SQLModel):
    roadinfra: list[RoadInfraRead]
    pagination: "PaginationInfo"


class PaginationInfo(SQLModel):
    page: int
    page_size: int
    total_pages: int
    total_count: int
    has_next: bool
    has_previous: bool
    next_cursor: int | None = None
    previous_cursor: int | None = None


class CursorPaginationInfo(SQLModel):
    page_size: int
    has_next: bool
    has_previous: bool
    next_cursor: int | None = None
    previous_cursor: int | None = None


class RoadInfraListCursorResponse(SQLModel):
    roadinfra: list[RoadInfraRead]
    pagination: CursorPaginationInfo
