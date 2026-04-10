"""Ticket endpoints.

Thin layer — parse request, call service, return response.
No business logic lives here.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from web_backend.database import get_db
from web_backend.models.user import AuthUser
from web_backend.schemas.tickets import (
    BoardTicketsResponse,
    TicketAssigneeResponse,
    TicketCreateRequest,
    TicketMoveRequest,
    TicketResponse,
    TicketUpdateRequest,
)
from web_backend.security.dependencies import get_current_user
from web_backend.services.tickets import (
    create_ticket,
    delete_ticket,
    get_board_tickets,
    get_ticket_by_key,
    move_ticket,
    update_ticket,
)

router = APIRouter(prefix="/projects/{slug}/tickets", tags=["Tickets"])


def _ticket_to_response(ticket) -> TicketResponse:
    assignee = None
    if ticket.assignee is not None:
        assignee = TicketAssigneeResponse(
            id=ticket.assignee.id,
            username=ticket.assignee.username,
            first_name=ticket.assignee.first_name,
            last_name=ticket.assignee.last_name,
            email=ticket.assignee.email,
        )

    return TicketResponse(
        id=ticket.id,
        project_id=ticket.project_id,
        column_id=ticket.column_id,
        ticket_key=ticket.ticket_key,
        title=ticket.title,
        description=ticket.description,
        priority=ticket.priority,
        type=ticket.type,
        size=ticket.size,
        labels=ticket.labels,
        due_date=ticket.due_date,
        position=ticket.position,
        assignee=assignee,
        created_by=ticket.created_by,
        created_at=ticket.created_at,
        updated_at=ticket.updated_at,
    )


@router.get("", response_model=BoardTicketsResponse)
async def list_board_tickets(
    slug: str,
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BoardTicketsResponse:
    try:
        tickets = await get_board_tickets(db, slug, current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return BoardTicketsResponse(tickets=[_ticket_to_response(t) for t in tickets])


@router.post("", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
async def create_ticket_endpoint(
    slug: str,
    data: TicketCreateRequest,
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TicketResponse:
    try:
        ticket = await create_ticket(db, slug, data, current_user)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _ticket_to_response(ticket)


@router.get("/{ticket_key}", response_model=TicketResponse)
async def get_ticket_endpoint(
    slug: str,
    ticket_key: str,
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TicketResponse:
    try:
        ticket = await get_ticket_by_key(db, slug, ticket_key, current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _ticket_to_response(ticket)


@router.patch("/{ticket_key}", response_model=TicketResponse)
async def update_ticket_endpoint(
    slug: str,
    ticket_key: str,
    data: TicketUpdateRequest,
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TicketResponse:
    try:
        ticket = await update_ticket(db, slug, ticket_key, data, current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _ticket_to_response(ticket)


@router.patch("/{ticket_key}/move", response_model=TicketResponse)
async def move_ticket_endpoint(
    slug: str,
    ticket_key: str,
    data: TicketMoveRequest,
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TicketResponse:
    try:
        ticket = await move_ticket(db, slug, ticket_key, data, current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _ticket_to_response(ticket)


@router.delete("/{ticket_key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ticket_endpoint(
    slug: str,
    ticket_key: str,
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    try:
        await delete_ticket(db, slug, ticket_key, current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc