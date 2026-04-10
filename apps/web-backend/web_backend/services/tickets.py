"""Ticket business logic."""

import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from web_backend.models.project import Project, ProjectBoardColumn, ProjectMember
from web_backend.models.ticket import ProjectTicket, ProjectTicketCounter
from web_backend.models.user import AuthUser
from web_backend.schemas.tickets import (
    TicketCreateRequest,
    TicketMoveRequest,
    TicketUpdateRequest,
)


async def _verify_membership(db, project_id, user_id):
    result = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user_id,
        )
    )
    m = result.scalar_one_or_none()
    if m is None:
        raise ValueError("You are not a member of this project")
    return m


async def _get_project_by_slug(db, slug):
    result = await db.execute(select(Project).where(Project.slug == slug))
    p = result.scalar_one_or_none()
    if p is None:
        raise ValueError("Project not found")
    return p


async def _generate_ticket_key(db, project_id, project_slug):
    result = await db.execute(
        select(ProjectTicketCounter).where(ProjectTicketCounter.project_id == project_id)
    )
    counter_row = result.scalar_one_or_none()
    if counter_row is None:
        counter_row = ProjectTicketCounter(project_id=project_id, counter=0)
        db.add(counter_row)
        await db.flush()
    counter_row.counter += 1
    parts = project_slug.upper().split("-")
    prefix = parts[0][:5] if parts else "TF"
    return f"{prefix}-{counter_row.counter}"


async def _get_next_position(db, column_id):
    result = await db.execute(
        select(func.coalesce(func.max(ProjectTicket.position), -1)).where(
            ProjectTicket.column_id == column_id
        )
    )
    return result.scalar_one() + 1


async def _get_ticket_loaded(db, ticket_id):
    result = await db.execute(
        select(ProjectTicket)
        .where(ProjectTicket.id == ticket_id)
        .options(selectinload(ProjectTicket.assignee))
    )
    t = result.scalar_one_or_none()
    if t is None:
        raise ValueError("Ticket not found")
    return t


async def create_ticket(db, slug, data: TicketCreateRequest, user: AuthUser):
    project = await _get_project_by_slug(db, slug)
    await _verify_membership(db, project.id, user.id)
    col_result = await db.execute(
        select(ProjectBoardColumn).where(
            ProjectBoardColumn.id == data.column_id,
            ProjectBoardColumn.project_id == project.id,
        )
    )
    if col_result.scalar_one_or_none() is None:
        raise ValueError("Invalid column for this project")
    if data.assignee_id is not None:
        a = await db.execute(
            select(ProjectMember).where(
                ProjectMember.project_id == project.id,
                ProjectMember.user_id == data.assignee_id,
            )
        )
        if a.scalar_one_or_none() is None:
            raise ValueError("Assignee is not a member of this project")
    ticket_key = await _generate_ticket_key(db, project.id, project.slug)
    position = await _get_next_position(db, data.column_id)
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
        size=data.size,
        labels=data.labels,
        due_date=data.due_date,
        position=position,
    )
    db.add(ticket)
    await db.commit()
    return await _get_ticket_loaded(db, ticket.id)


async def get_board_tickets(db, slug, user_id):
    project = await _get_project_by_slug(db, slug)
    await _verify_membership(db, project.id, user_id)
    result = await db.execute(
        select(ProjectTicket)
        .where(ProjectTicket.project_id == project.id)
        .options(selectinload(ProjectTicket.assignee))
        .order_by(ProjectTicket.column_id, ProjectTicket.position)
    )
    return list(result.scalars().all())


async def get_ticket_by_key(db, slug, ticket_key, user_id):
    project = await _get_project_by_slug(db, slug)
    await _verify_membership(db, project.id, user_id)
    result = await db.execute(
        select(ProjectTicket)
        .where(ProjectTicket.project_id == project.id, ProjectTicket.ticket_key == ticket_key)
        .options(selectinload(ProjectTicket.assignee))
    )
    t = result.scalar_one_or_none()
    if t is None:
        raise ValueError("Ticket not found")
    return t


async def update_ticket(db, slug, ticket_key, data: TicketUpdateRequest, user_id):
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
        raise ValueError("Ticket not found")
    if data.assignee_id is not None:
        a = await db.execute(
            select(ProjectMember).where(
                ProjectMember.project_id == project.id,
                ProjectMember.user_id == data.assignee_id,
            )
        )
        if a.scalar_one_or_none() is None:
            raise ValueError("Assignee is not a member of this project")
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(ticket, field, value)
    await db.commit()
    return await _get_ticket_loaded(db, ticket.id)


async def move_ticket(db, slug, ticket_key, data: TicketMoveRequest, user_id):
    project = await _get_project_by_slug(db, slug)
    await _verify_membership(db, project.id, user_id)
    col_result = await db.execute(
        select(ProjectBoardColumn).where(
            ProjectBoardColumn.id == data.column_id,
            ProjectBoardColumn.project_id == project.id,
        )
    )
    if col_result.scalar_one_or_none() is None:
        raise ValueError("Invalid column for this project")
    result = await db.execute(
        select(ProjectTicket).where(
            ProjectTicket.project_id == project.id,
            ProjectTicket.ticket_key == ticket_key,
        )
    )
    ticket = result.scalar_one_or_none()
    if ticket is None:
        raise ValueError("Ticket not found")
    old_column_id = ticket.column_id
    old_position = ticket.position
    if old_column_id == data.column_id:
        if data.position > old_position:
            await db.execute(
                update(ProjectTicket).where(
                    ProjectTicket.column_id == data.column_id,
                    ProjectTicket.position > old_position,
                    ProjectTicket.position <= data.position,
                    ProjectTicket.id != ticket.id,
                ).values(position=ProjectTicket.position - 1)
            )
        elif data.position < old_position:
            await db.execute(
                update(ProjectTicket).where(
                    ProjectTicket.column_id == data.column_id,
                    ProjectTicket.position >= data.position,
                    ProjectTicket.position < old_position,
                    ProjectTicket.id != ticket.id,
                ).values(position=ProjectTicket.position + 1)
            )
    else:
        await db.execute(
            update(ProjectTicket).where(
                ProjectTicket.column_id == old_column_id,
                ProjectTicket.position > old_position,
                ProjectTicket.id != ticket.id,
            ).values(position=ProjectTicket.position - 1)
        )
        await db.execute(
            update(ProjectTicket).where(
                ProjectTicket.column_id == data.column_id,
                ProjectTicket.position >= data.position,
                ProjectTicket.id != ticket.id,
            ).values(position=ProjectTicket.position + 1)
        )
    ticket.column_id = data.column_id
    ticket.position = data.position
    await db.commit()
    return await _get_ticket_loaded(db, ticket.id)


async def delete_ticket(db, slug, ticket_key, user_id):
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
        raise ValueError("Ticket not found")
    column_id = ticket.column_id
    position = ticket.position
    await db.delete(ticket)
    await db.execute(
        update(ProjectTicket).where(
            ProjectTicket.column_id == column_id,
            ProjectTicket.position > position,
        ).values(position=ProjectTicket.position - 1)
    )
    await db.commit()
