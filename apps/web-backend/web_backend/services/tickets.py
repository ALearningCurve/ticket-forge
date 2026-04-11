"""Ticket business logic.

Handles ticket creation, updates, sizing, moves (drag-and-drop),
and deletion. Routes call these — no direct DB queries in routes.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from web_backend.models.project import Project, ProjectBoardColumn, ProjectMember
from web_backend.models.ticket import ProjectTicket, ProjectTicketCounter
from web_backend.models.user import AuthUser
from web_backend.schemas.inference import TicketSizePredictionRequest
from web_backend.schemas.tickets import (
    TicketCreateRequest,
    TicketMoveRequest,
    TicketUpdateRequest,
)
from web_backend.services.recommendations import (
    apply_ticket_completion_profile_update,
    is_completion_column_name,
    sync_project_ticket_to_ml_tables,
)
from web_backend.services.inference import predict_ticket_size
from shared.logging import get_logger

logger = get_logger(__name__)

MANUAL_SIZE_SOURCE = "manual"
PREDICTED_SIZE_SOURCE = "predicted"
SIZE_RECOMPUTE_FIELDS = {"title", "description", "labels", "type"}


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


async def _verify_membership(
    db: AsyncSession,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
) -> ProjectMember:
    """Verify user is a member of the project. Raises ValueError if not."""
    result = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user_id,
        )
    )
    membership = result.scalar_one_or_none()
    if membership is None:
        msg = "You are not a member of this project"
        raise ValueError(msg)
    return membership


async def _get_project_by_slug(db: AsyncSession, slug: str) -> Project:
    """Fetch project by slug. Raises ValueError if not found."""
    result = await db.execute(select(Project).where(Project.slug == slug))
    project = result.scalar_one_or_none()
    if project is None:
        msg = "Project not found"
        raise ValueError(msg)
    return project


async def _generate_ticket_key(
    db: AsyncSession,
    project_id: uuid.UUID,
    project_slug: str,
) -> str:
    """Generate the next ticket key like TF-1, TF-2, etc."""
    # Get or create counter
    result = await db.execute(
        select(ProjectTicketCounter).where(
            ProjectTicketCounter.project_id == project_id
        )
    )
    counter_row = result.scalar_one_or_none()

    if counter_row is None:
        counter_row = ProjectTicketCounter(project_id=project_id, counter=0)
        db.add(counter_row)
        await db.flush()

    counter_row.counter += 1
    next_num = counter_row.counter

    # Build prefix from slug (e.g. "my-project" → "MY" or "testing" → "TEST")
    parts = project_slug.upper().split("-")
    prefix = parts[0][:5] if parts else "TF"

    return f"{prefix}-{next_num}"


async def _get_next_position(
    db: AsyncSession,
    column_id: uuid.UUID,
) -> int:
    """Get the next position value for a column."""
    result = await db.execute(
        select(func.coalesce(func.max(ProjectTicket.position), -1)).where(
            ProjectTicket.column_id == column_id
        )
    )
    max_pos = result.scalar_one()
    return max_pos + 1


def _assign_manual_size(ticket: ProjectTicket, size_bucket: str | None) -> None:
    """Persist a user-provided size override on a ticket.

    Args:
        ticket: Ticket being modified.
        size_bucket: Explicit size bucket. ``None`` clears any stored size so the
            backend can fall back to prediction.
    """
    if size_bucket is None:
        ticket.size_bucket = None
        ticket.size_source = None
        ticket.size_confidence = None
        ticket.size_updated_at = None
        return

    ticket.size_bucket = size_bucket
    ticket.size_source = MANUAL_SIZE_SOURCE
    ticket.size_confidence = None
    ticket.size_updated_at = datetime.now(UTC)


def _build_ticket_prediction_request(
    project: Project,
    ticket: ProjectTicket,
    *,
    rail: str,
) -> TicketSizePredictionRequest:
    """Convert a project ticket into the serving payload shape.

    Args:
        project: Project that owns the ticket.
        ticket: Ticket to classify.
        rail: Logical serving rail label for inference telemetry.

    Returns:
        TicketSizePredictionRequest for the production classifier.
    """
    assigned_at = ticket.updated_at if ticket.assignee_id is not None else None
    return TicketSizePredictionRequest(
        title=ticket.title,
        body=ticket.description or "",
        repo=project.slug,
        issue_type=ticket.type,
        labels=[str(label) for label in (ticket.labels or [])],
        comments_count=0,
        historical_avg_completion_hours=0.0,
        created_at=ticket.created_at,
        assigned_at=assigned_at,
        rail=rail,
    )


async def _predict_and_persist_ticket_size(
    db: AsyncSession,
    *,
    project: Project,
    ticket: ProjectTicket,
    rail: str,
) -> bool:
    """Infer and store a ticket size without breaking the main request path.

    Args:
        db: Active async database session.
        project: Project that owns the ticket.
        ticket: Ticket to classify.
        rail: Serving telemetry rail label.

    Returns:
        ``True`` when a prediction was stored, otherwise ``False``.
    """
    project_slug = project.slug
    ticket_key = ticket.ticket_key

    try:
        prediction = await predict_ticket_size(
            db,
            _build_ticket_prediction_request(project, ticket, rail=rail),
        )
    except Exception:
        await db.rollback()
        logger.exception(
            "Failed to predict ticket size for project=%s ticket=%s",
            project_slug,
            ticket_key,
        )
        return False

    ticket.size_bucket = prediction.predicted_bucket
    ticket.size_source = PREDICTED_SIZE_SOURCE
    ticket.size_confidence = prediction.confidence
    ticket.size_updated_at = datetime.now(UTC)

    try:
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception(
            "Failed to persist predicted ticket size for project=%s ticket=%s",
            project_slug,
            ticket_key,
        )
        return False

    return True


# ------------------------------------------------------------------ #
#  Create ticket
# ------------------------------------------------------------------ #


async def create_ticket(
    db: AsyncSession,
    slug: str,
    data: TicketCreateRequest,
    user: AuthUser,
) -> ProjectTicket:
    """Create a ticket in a project.

    Raises:
        ValueError: If project not found, not a member, or invalid column/assignee.
    """
    project = await _get_project_by_slug(db, slug)
    await _verify_membership(db, project.id, user.id)

    # Verify column belongs to this project
    col_result = await db.execute(
        select(ProjectBoardColumn).where(
            ProjectBoardColumn.id == data.column_id,
            ProjectBoardColumn.project_id == project.id,
        )
    )
    destination_column = col_result.scalar_one_or_none()
    if destination_column is None:
        msg = "Invalid column for this project"
        raise ValueError(msg)

    # Verify assignee is a project member (if provided)
    if data.assignee_id is not None:
        assignee_result = await db.execute(
            select(ProjectMember).where(
                ProjectMember.project_id == project.id,
                ProjectMember.user_id == data.assignee_id,
            )
        )
        if assignee_result.scalar_one_or_none() is None:
            msg = "Assignee is not a member of this project"
            raise ValueError(msg)

    ticket_key = await _generate_ticket_key(db, project.id, project.slug)
    position = await _get_next_position(db, data.column_id)

    requested_manual_size = (
        data.size_bucket if data.size_bucket is not None else data.size
    )

    ticket = ProjectTicket(
        project_id=project.id,
        column_id=data.column_id,
        assignee_id=data.assignee_id,
        created_by=user.id,
        ticket_key=ticket_key,
        title=data.title,
        description=data.description,
        priority=data.priority,
        type=data.type,
        labels=data.labels,
        size_bucket=requested_manual_size,
        size_source=MANUAL_SIZE_SOURCE if requested_manual_size is not None else None,
        size_updated_at=datetime.now(UTC)
        if requested_manual_size is not None
        else None,
        due_date=data.due_date,
        position=position,
    )
    db.add(ticket)
    await db.commit()
    await db.refresh(ticket)
    await sync_project_ticket_to_ml_tables(db, project, ticket, destination_column)
    await db.commit()

    ticket = await _get_ticket_loaded(db, ticket.id)
    ticket_id = ticket.id
    if ticket.size_bucket is None:
        await _predict_and_persist_ticket_size(
            db,
            project=project,
            ticket=ticket,
            rail="board_ticket_create",
        )

    return await _get_ticket_loaded(db, ticket_id)


# ------------------------------------------------------------------ #
#  Get tickets for board
# ------------------------------------------------------------------ #


async def get_board_tickets(
    db: AsyncSession,
    slug: str,
    user_id: uuid.UUID,
) -> list[ProjectTicket]:
    """Get all tickets for a project board.

    Raises:
        ValueError: If project not found or user not a member.
    """
    project = await _get_project_by_slug(db, slug)
    await _verify_membership(db, project.id, user_id)

    result = await db.execute(
        select(ProjectTicket)
        .where(ProjectTicket.project_id == project.id)
        .options(selectinload(ProjectTicket.assignee))
        .order_by(ProjectTicket.column_id, ProjectTicket.position)
    )
    return list(result.scalars().all())


# ------------------------------------------------------------------ #
#  Get single ticket
# ------------------------------------------------------------------ #


async def _get_ticket_loaded(
    db: AsyncSession,
    ticket_id: uuid.UUID,
) -> ProjectTicket:
    """Fetch a ticket with assignee loaded."""
    result = await db.execute(
        select(ProjectTicket)
        .where(ProjectTicket.id == ticket_id)
        .options(selectinload(ProjectTicket.assignee))
    )
    ticket = result.scalar_one_or_none()
    if ticket is None:
        msg = "Ticket not found"
        raise ValueError(msg)
    return ticket


async def get_ticket_by_key(
    db: AsyncSession,
    slug: str,
    ticket_key: str,
    user_id: uuid.UUID,
) -> ProjectTicket:
    """Get a single ticket by its key.

    Raises:
        ValueError: If not found or user not a member.
    """
    project = await _get_project_by_slug(db, slug)
    await _verify_membership(db, project.id, user_id)

    result = await db.execute(
        select(ProjectTicket)
        .where(
            ProjectTicket.project_id == project.id,
            ProjectTicket.ticket_key == ticket_key,
        )
        .options(selectinload(ProjectTicket.assignee))
    )
    ticket = result.scalar_one_or_none()
    if ticket is None:
        msg = "Ticket not found"
        raise ValueError(msg)
    return ticket


# ------------------------------------------------------------------ #
#  Update ticket
# ------------------------------------------------------------------ #


async def update_ticket(
    db: AsyncSession,
    slug: str,
    ticket_key: str,
    data: TicketUpdateRequest,
    user_id: uuid.UUID,
) -> ProjectTicket:
    """Update ticket fields.

    Raises:
        ValueError: If not found, not a member, or invalid assignee.
    """
    project = await _get_project_by_slug(db, slug)
    await _verify_membership(db, project.id, user_id)

    result = await db.execute(
        select(ProjectTicket).where(
            ProjectTicket.project_id == project.id,
            ProjectTicket.ticket_key == ticket_key,
        )
    )
    ticket = result.scalar_one_or_none()
    if ticket is None:
        msg = "Ticket not found"
        raise ValueError(msg)

    # Validate assignee if changing
    if data.assignee_id is not None:
        assignee_check = await db.execute(
            select(ProjectMember).where(
                ProjectMember.project_id == project.id,
                ProjectMember.user_id == data.assignee_id,
            )
        )
        if assignee_check.scalar_one_or_none() is None:
            msg = "Assignee is not a member of this project"
            raise ValueError(msg)

    # Apply updates
    update_fields = data.model_dump(exclude_unset=True)
    legacy_size_provided = "size" in data.model_fields_set
    manual_size_provided = (
        "size_bucket" in data.model_fields_set or legacy_size_provided
    )
    requested_size_bucket = update_fields.pop("size_bucket", None)
    requested_legacy_size = update_fields.pop("size", None)
    requested_size = (
        requested_size_bucket
        if "size_bucket" in data.model_fields_set
        else requested_legacy_size
    )
    recompute_prediction = bool(SIZE_RECOMPUTE_FIELDS.intersection(update_fields))

    for field, value in update_fields.items():
        setattr(ticket, field, value)

    await db.commit()
    await db.refresh(ticket)
    column_result = await db.execute(
        select(ProjectBoardColumn).where(ProjectBoardColumn.id == ticket.column_id)
    )
    current_column = column_result.scalar_one()
    await sync_project_ticket_to_ml_tables(db, project, ticket, current_column)
    await db.commit()

    ticket = await _get_ticket_loaded(db, ticket.id)
    ticket_id = ticket.id

    if manual_size_provided:
        _assign_manual_size(ticket, requested_size)
        await db.commit()
        ticket = await _get_ticket_loaded(db, ticket.id)
        if requested_size is None:
            await _predict_and_persist_ticket_size(
                db,
                project=project,
                ticket=ticket,
                rail="board_ticket_update",
            )
            return await _get_ticket_loaded(db, ticket_id)

    if ticket.size_source != MANUAL_SIZE_SOURCE and (
        ticket.size_bucket is None or recompute_prediction
    ):
        await _predict_and_persist_ticket_size(
            db,
            project=project,
            ticket=ticket,
            rail="board_ticket_update",
        )

    return await _get_ticket_loaded(db, ticket_id)


# ------------------------------------------------------------------ #
#  Batch classify unsized tickets
# ------------------------------------------------------------------ #


async def classify_missing_ticket_sizes(
    db: AsyncSession,
    slug: str,
    user_id: uuid.UUID,
) -> tuple[int, list[ProjectTicket]]:
    """Classify and persist any unsized tickets for a project."""
    project = await _get_project_by_slug(db, slug)
    await _verify_membership(db, project.id, user_id)

    result = await db.execute(
        select(ProjectTicket)
        .where(
            ProjectTicket.project_id == project.id,
            ProjectTicket.size_bucket.is_(None),
        )
        .options(selectinload(ProjectTicket.assignee))
        .order_by(ProjectTicket.position)
    )
    tickets = list(result.scalars().all())

    updated_count = 0
    for ticket in tickets:
        updated_count += int(
            await _predict_and_persist_ticket_size(
                db,
                project=project,
                ticket=ticket,
                rail="board_ticket_batch",
            )
        )

    refreshed_tickets = await get_board_tickets(db, slug, user_id)
    return updated_count, refreshed_tickets


# ------------------------------------------------------------------ #
#  Move ticket (drag-and-drop)
# ------------------------------------------------------------------ #


async def move_ticket(
    db: AsyncSession,
    slug: str,
    ticket_key: str,
    data: TicketMoveRequest,
    user_id: uuid.UUID,
) -> ProjectTicket:
    """Move a ticket to a different column/position.

    Reorders other tickets in the destination column to make room.

    Raises:
        ValueError: If not found, not a member, or invalid column.
    """
    project = await _get_project_by_slug(db, slug)
    await _verify_membership(db, project.id, user_id)

    # Verify destination column
    col_result = await db.execute(
        select(ProjectBoardColumn).where(
            ProjectBoardColumn.id == data.column_id,
            ProjectBoardColumn.project_id == project.id,
        )
    )
    destination_column = col_result.scalar_one_or_none()
    if destination_column is None:
        msg = "Invalid column for this project"
        raise ValueError(msg)

    result = await db.execute(
        select(ProjectTicket).where(
            ProjectTicket.project_id == project.id,
            ProjectTicket.ticket_key == ticket_key,
        )
    )
    ticket = result.scalar_one_or_none()
    if ticket is None:
        msg = "Ticket not found"
        raise ValueError(msg)

    old_column_id = ticket.column_id
    old_position = ticket.position
    old_column_result = await db.execute(
        select(ProjectBoardColumn).where(ProjectBoardColumn.id == old_column_id)
    )
    old_column = old_column_result.scalar_one()
    was_complete = is_completion_column_name(old_column.name)
    is_complete = is_completion_column_name(destination_column.name)

    # If moving within the same column
    if old_column_id == data.column_id:
        if data.position > old_position:
            # Moving down — shift others up
            await db.execute(
                update(ProjectTicket)
                .where(
                    ProjectTicket.column_id == data.column_id,
                    ProjectTicket.position > old_position,
                    ProjectTicket.position <= data.position,
                    ProjectTicket.id != ticket.id,
                )
                .values(position=ProjectTicket.position - 1)
            )
        elif data.position < old_position:
            # Moving up — shift others down
            await db.execute(
                update(ProjectTicket)
                .where(
                    ProjectTicket.column_id == data.column_id,
                    ProjectTicket.position >= data.position,
                    ProjectTicket.position < old_position,
                    ProjectTicket.id != ticket.id,
                )
                .values(position=ProjectTicket.position + 1)
            )
    else:
        # Moving to different column
        # Close gap in source column
        await db.execute(
            update(ProjectTicket)
            .where(
                ProjectTicket.column_id == old_column_id,
                ProjectTicket.position > old_position,
                ProjectTicket.id != ticket.id,
            )
            .values(position=ProjectTicket.position - 1)
        )
        # Make room in destination column
        await db.execute(
            update(ProjectTicket)
            .where(
                ProjectTicket.column_id == data.column_id,
                ProjectTicket.position >= data.position,
                ProjectTicket.id != ticket.id,
            )
            .values(position=ProjectTicket.position + 1)
        )

    ticket.column_id = data.column_id
    ticket.position = data.position

    await db.commit()
    await db.refresh(ticket)
    await sync_project_ticket_to_ml_tables(db, project, ticket, destination_column)
    await db.commit()
    if not was_complete and is_complete:
        await apply_ticket_completion_profile_update(
            db,
            project=project,
            ticket=ticket,
            column=destination_column,
        )
    return await _get_ticket_loaded(db, ticket.id)


# ------------------------------------------------------------------ #
#  Delete ticket
# ------------------------------------------------------------------ #


async def delete_ticket(
    db: AsyncSession,
    slug: str,
    ticket_key: str,
    user_id: uuid.UUID,
) -> None:
    """Delete a ticket. Any project member can delete.

    Raises:
        ValueError: If not found or not a member.
    """
    project = await _get_project_by_slug(db, slug)
    await _verify_membership(db, project.id, user_id)

    result = await db.execute(
        select(ProjectTicket).where(
            ProjectTicket.project_id == project.id,
            ProjectTicket.ticket_key == ticket_key,
        )
    )
    ticket = result.scalar_one_or_none()
    if ticket is None:
        msg = "Ticket not found"
        raise ValueError(msg)

    column_id = ticket.column_id
    position = ticket.position

    await db.delete(ticket)

    # Close gap in column
    await db.execute(
        update(ProjectTicket)
        .where(
            ProjectTicket.column_id == column_id,
            ProjectTicket.position > position,
        )
        .values(position=ProjectTicket.position - 1)
    )

    await db.commit()
