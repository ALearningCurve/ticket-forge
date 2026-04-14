"""Semantic recommendation and ML-table sync services."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

try:
    from ml_core.embeddings import get_embedding_service
    from ml_core.keywords import get_keyword_extractor
    from ml_core.profiles import ProfileUpdater
    from ml_core.retrieval.hybrid_retrieval import vector_to_pgvector_text

    _HAS_ML = True
except ImportError:
    _HAS_ML = False

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from web_backend.models.project import Project, ProjectBoardColumn, ProjectMember
from web_backend.models.ticket import ProjectTicket
from web_backend.models.user import AuthUser
from web_backend.schemas.recommendations import (
    EngineerTicketRecommendationsResponse,
    RecommendedEngineerResponse,
    RecommendedTicketResponse,
    TicketEngineerRecommendationsResponse,
)

EMBEDDING_DIM = 384
ZERO_VECTOR_TEXT = "[" + ",".join("0.0" for _ in range(EMBEDDING_DIM)) + "]"
_COMPLETION_COLUMN_NAMES = {"done", "complete", "completed", "closed", "resolved"}
_IN_PROGRESS_COLUMN_NAMES = {"in progress", "in-progress", "in review", "review"}
_CAPACITY_THRESHOLD = 3


@dataclass(slots=True)
class _MemberProfile:
    """Project member and linked ML profile identifiers."""

    user_id: uuid.UUID
    member_id: int
    username: str
    first_name: str
    last_name: str
    email: str


def is_completion_column_name(column_name: str) -> bool:
    """Return whether a board column name should be treated as complete."""

    return column_name.strip().lower() in _COMPLETION_COLUMN_NAMES


def _ticket_status_for_column(column_name: str) -> str:
    """Map a board column name to the ML-side ticket status enum."""

    normalized = column_name.strip().lower()
    if normalized in _COMPLETION_COLUMN_NAMES:
        return "closed"
    if normalized in _IN_PROGRESS_COLUMN_NAMES:
        return "in-progress"
    return "open"


async def _get_project_for_member_validation(
    db: AsyncSession,
    slug: str,
    current_user_id: uuid.UUID,
) -> Project:
    """Fetch a project and ensure the current user is a member."""

    result = await db.execute(
        select(Project)
        .where(Project.slug == slug)
        .join(ProjectMember, ProjectMember.project_id == Project.id)
        .where(ProjectMember.user_id == current_user_id)
    )
    project = result.scalar_one_or_none()
    if project is None:
        msg = "Project not found"
        raise ValueError(msg)
    return project


async def _get_target_project_member(
    db: AsyncSession,
    project_id: uuid.UUID,
    user_id: uuid.UUID,
) -> ProjectMember:
    """Fetch a project member with the linked auth user loaded."""

    result = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id, ProjectMember.user_id == user_id
        )
    )
    member = result.scalar_one_or_none()
    if member is None:
        msg = "Engineer is not a member of this project"
        raise ValueError(msg)
    return member


async def _fetch_auth_user(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> AuthUser:
    """Fetch an auth user by primary key."""

    result = await db.execute(select(AuthUser).where(AuthUser.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        msg = "User not found"
        raise ValueError(msg)
    return user


async def _ensure_member_profile(
    db: AsyncSession,
    user: AuthUser,
) -> _MemberProfile:
    """Ensure an auth user has a linked ML profile row."""

    if user.member_id is not None:
        existing = await db.execute(
            text(
                """
                SELECT member_id
                FROM users
                WHERE member_id = :member_id
                """
            ),
            {"member_id": user.member_id},
        )
        if existing.mappings().first() is not None:
            return _MemberProfile(
                user_id=user.id,
                member_id=user.member_id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
                email=user.email,
            )

    by_username = await db.execute(
        text(
            """
            SELECT member_id
            FROM users
            WHERE github_username = :github_username
            """
        ),
        {"github_username": user.username},
    )
    existing_row = by_username.mappings().first()
    if existing_row is not None:
        member_id = int(existing_row["member_id"])
        user.member_id = member_id
        await db.flush()
        return _MemberProfile(
            user_id=user.id,
            member_id=member_id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
        )

    full_name = f"{user.first_name} {user.last_name}".strip()
    inserted = await db.execute(
        text(
            """
            INSERT INTO users (
              github_username,
              full_name,
              resume_base_vector,
              profile_vector,
              skill_keywords
            )
            VALUES (
              :github_username,
              :full_name,
              NULL,
              CAST(:zero_vector AS vector),
              to_tsvector('english', '')
            )
            RETURNING member_id
            """
        ),
        {
            "github_username": user.username,
            "full_name": full_name,
            "zero_vector": ZERO_VECTOR_TEXT,
        },
    )
    row = inserted.mappings().one()
    member_id = int(row["member_id"])
    user.member_id = member_id
    await db.flush()
    return _MemberProfile(
        user_id=user.id,
        member_id=member_id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        email=user.email,
    )


async def ensure_project_member_profiles(
    db: AsyncSession,
    project_id: uuid.UUID,
) -> dict[uuid.UUID, _MemberProfile]:
    """Ensure every project member has a linked ML profile."""

    result = await db.execute(
        select(AuthUser)
        .join(ProjectMember, ProjectMember.user_id == AuthUser.id)
        .where(ProjectMember.project_id == project_id)
    )
    users = list(result.scalars().all())
    profiles: dict[uuid.UUID, _MemberProfile] = {}
    for user in users:
        profile = await _ensure_member_profile(db, user)
        profiles[user.id] = profile
    await db.commit()
    return profiles


async def _fetch_project_ticket_with_column(
    db: AsyncSession,
    project_id: uuid.UUID,
    ticket_key: str,
) -> tuple[ProjectTicket, ProjectBoardColumn]:
    """Fetch a project ticket and its current board column."""

    ticket_result = await db.execute(
        select(ProjectTicket).where(
            ProjectTicket.project_id == project_id,
            ProjectTicket.ticket_key == ticket_key,
        )
    )
    ticket = ticket_result.scalar_one_or_none()
    if ticket is None:
        msg = "Ticket not found"
        raise ValueError(msg)

    column_result = await db.execute(
        select(ProjectBoardColumn).where(ProjectBoardColumn.id == ticket.column_id)
    )
    column = column_result.scalar_one()
    return ticket, column


def _embed_ticket_text(title: str, description: str | None) -> tuple[str, str]:
    """Generate the pgvector text and lexical query text for a ticket."""

    if not _HAS_ML:
        msg = "ml-core is required for embedding tickets"
        raise RuntimeError(msg)

    embedding_service = get_embedding_service()
    keyword_extractor = get_keyword_extractor()
    combined_text = f"{title}\n{description or ''}".strip()
    raw_vector = embedding_service.embed_text(combined_text)
    vector_values = raw_vector.tolist() if hasattr(raw_vector, "tolist") else raw_vector
    vector_text = vector_to_pgvector_text([float(value) for value in vector_values])
    keywords = keyword_extractor.extract(combined_text, top_n=10)
    keyword_text = " ".join(keywords).strip() or title.strip()
    return vector_text, keyword_text


async def sync_project_ticket_to_ml_tables(
    db: AsyncSession,
    project: Project,
    ticket: ProjectTicket,
    column: ProjectBoardColumn,
) -> None:
    """Upsert a project ticket into the ML-side ticket and assignment tables."""

    if not _HAS_ML:
        return

    vector_text, _ = _embed_ticket_text(ticket.title, ticket.description)
    await db.execute(
        text(
            """
            INSERT INTO tickets (
              ticket_id,
              title,
              description,
              ticket_vector,
              labels,
              status,
              project_id,
              created_at,
              updated_at
            )
            VALUES (
              :ticket_id,
              :title,
              :description,
              CAST(:ticket_vector AS vector),
              CAST(:labels AS jsonb),
              CAST(:status AS ticket_status),
              :project_id,
              :created_at,
              :updated_at
            )
            ON CONFLICT (ticket_id)
            DO UPDATE SET
              title = EXCLUDED.title,
              description = EXCLUDED.description,
              ticket_vector = EXCLUDED.ticket_vector,
              labels = EXCLUDED.labels,
              status = EXCLUDED.status,
              project_id = EXCLUDED.project_id,
              updated_at = EXCLUDED.updated_at
            """
        ),
        {
            "ticket_id": ticket.ticket_key,
            "title": ticket.title,
            "description": ticket.description or "",
            "ticket_vector": vector_text,
            "labels": json.dumps(ticket.labels or []),
            "status": _ticket_status_for_column(column.name),
            "project_id": project.id,
            "created_at": ticket.created_at,
            "updated_at": ticket.updated_at,
        },
    )

    if ticket.assignee_id is None:
        await db.execute(
            text(
                """
                DELETE FROM assignments
                WHERE ticket_id = :ticket_id
                """
            ),
            {"ticket_id": ticket.ticket_key},
        )
        return

    assignee = await _fetch_auth_user(db, ticket.assignee_id)
    profile = await _ensure_member_profile(db, assignee)
    await db.execute(
        text(
            """
            DELETE FROM assignments
            WHERE ticket_id = :ticket_id
              AND engineer_id <> :engineer_id
            """
        ),
        {
            "ticket_id": ticket.ticket_key,
            "engineer_id": profile.member_id,
        },
    )
    await db.execute(
        text(
            """
            INSERT INTO assignments (ticket_id, engineer_id, assigned_at)
            VALUES (:ticket_id, :engineer_id, now())
            ON CONFLICT (ticket_id, engineer_id) DO NOTHING
            """
        ),
        {
            "ticket_id": ticket.ticket_key,
            "engineer_id": profile.member_id,
        },
    )


async def sync_project_tickets_for_recommendations(
    db: AsyncSession,
    project: Project,
) -> None:
    """Sync all project tickets into the ML-side tables before recommendation."""

    if not _HAS_ML:
        return

    ticket_rows = await db.execute(
        select(ProjectTicket, ProjectBoardColumn)
        .join(ProjectBoardColumn, ProjectBoardColumn.id == ProjectTicket.column_id)
        .where(ProjectTicket.project_id == project.id)
    )
    for ticket, column in ticket_rows.all():
        await sync_project_ticket_to_ml_tables(db, project, ticket, column)
    await db.commit()


async def recommend_engineers_for_ticket(
    db: AsyncSession,
    slug: str,
    ticket_key: str,
    current_user_id: uuid.UUID,
) -> TicketEngineerRecommendationsResponse:
    """Return recommended engineers for a project ticket."""

    project = await _get_project_for_member_validation(db, slug, current_user_id)
    member_profiles = await ensure_project_member_profiles(db, project.id)
    ticket, column = await _fetch_project_ticket_with_column(db, project.id, ticket_key)

    if _HAS_ML:
        await sync_project_ticket_to_ml_tables(db, project, ticket, column)
        await db.commit()
        vector_text, keyword_text = _embed_ticket_text(ticket.title, ticket.description)
    else:
        vector_text = ZERO_VECTOR_TEXT
        keyword_text = ticket.title

    allowed_member_ids = [profile.member_id for profile in member_profiles.values()]
    query = text(
        """
        WITH allowed_engineers AS (
          SELECT
            au.id AS user_id,
            u.member_id,
            au.username,
            au.first_name,
            au.last_name,
            au.email,
            COUNT(pt.id) FILTER (
              WHERE pt.assignee_id = au.id
                AND lower(pbc.name) NOT IN ('done', 'complete', 'completed', 'closed', 'resolved')
            ) AS active_ticket_count
          FROM auth_users au
          JOIN project_members pm
            ON pm.user_id = au.id
          JOIN users u
            ON u.member_id = au.member_id
          LEFT JOIN project_tickets pt
            ON pt.project_id = pm.project_id
           AND pt.assignee_id = au.id
          LEFT JOIN project_board_columns pbc
            ON pbc.id = pt.column_id
          WHERE pm.project_id = :project_id
            AND u.member_id = ANY(:member_ids)
          GROUP BY au.id, u.member_id, au.username, au.first_name, au.last_name, au.email
        ),
        semantic_results AS (
          SELECT
            ae.*,
            1 - (u.profile_vector <=> CAST(:ticket_vector AS vector)) AS semantic_similarity
          FROM allowed_engineers ae
          JOIN users u ON u.member_id = ae.member_id
        ),
        lexical_results AS (
          SELECT
            ae.member_id,
            ts_rank(u.skill_keywords, plainto_tsquery('english', :keyword_text)) AS lexical_score
          FROM allowed_engineers ae
          JOIN users u ON u.member_id = ae.member_id
        )
        SELECT
          sr.member_id,
          sr.user_id,
          sr.username,
          sr.first_name,
          sr.last_name,
          sr.email,
          sr.active_ticket_count,
          GREATEST(0.0, 1.0 - (sr.active_ticket_count::float / :capacity_threshold)) AS capacity_score,
          GREATEST(0.0, 1.0 - (sr.active_ticket_count::float / :capacity_threshold)) > 0 AS has_capacity,
          sr.semantic_similarity,
          COALESCE(lr.lexical_score, 0.0) AS lexical_score,
          ((sr.semantic_similarity * 0.7)
            + (COALESCE(lr.lexical_score, 0.0) * 0.2)
            + (GREATEST(0.0, 1.0 - (sr.active_ticket_count::float / :capacity_threshold)) * 0.1))
            AS recommendation_score
        FROM semantic_results sr
        LEFT JOIN lexical_results lr
          ON lr.member_id = sr.member_id
        ORDER BY recommendation_score DESC, sr.username ASC
        LIMIT 5
        """
    )
    result = await db.execute(
        query,
        {
            "project_id": project.id,
            "member_ids": allowed_member_ids,
            "ticket_vector": vector_text,
            "keyword_text": keyword_text,
            "capacity_threshold": float(_CAPACITY_THRESHOLD),
        },
    )
    recommendations = [
        RecommendedEngineerResponse(
            member_id=int(row["member_id"]),
            user_id=row["user_id"],
            username=row["username"],
            first_name=row["first_name"],
            last_name=row["last_name"],
            email=row["email"],
            active_ticket_count=int(row["active_ticket_count"]),
            capacity_score=float(row["capacity_score"]),
            has_capacity=bool(row["has_capacity"]),
            semantic_similarity=float(row["semantic_similarity"]),
            lexical_score=float(row["lexical_score"]),
            recommendation_score=float(row["recommendation_score"]),
        )
        for row in result.mappings().all()
    ]
    return TicketEngineerRecommendationsResponse(
        ticket_key=ticket.ticket_key,
        recommendations=recommendations,
    )


async def recommend_tickets_for_engineer(
    db: AsyncSession,
    slug: str,
    engineer_user_id: uuid.UUID,
    current_user_id: uuid.UUID,
) -> EngineerTicketRecommendationsResponse:
    """Return recommended tickets for a project engineer."""

    project = await _get_project_for_member_validation(db, slug, current_user_id)
    await _get_target_project_member(db, project.id, engineer_user_id)
    member_profiles = await ensure_project_member_profiles(db, project.id)

    if _HAS_ML:
        await sync_project_tickets_for_recommendations(db, project)

    target_profile = member_profiles.get(engineer_user_id)
    if target_profile is None:
        msg = "Engineer profile not found"
        raise ValueError(msg)

    active_result = await db.execute(
        select(func.count(ProjectTicket.id))
        .join(ProjectBoardColumn, ProjectBoardColumn.id == ProjectTicket.column_id)
        .where(
            ProjectTicket.project_id == project.id,
            ProjectTicket.assignee_id == engineer_user_id,
            func.lower(ProjectBoardColumn.name).not_in(_COMPLETION_COLUMN_NAMES),
        )
    )
    active_ticket_count = int(active_result.scalar_one() or 0)

    query = text(
        """
        WITH engineer AS (
          SELECT
            au.id AS user_id,
            au.username,
            au.first_name,
            au.last_name,
            u.member_id,
            u.profile_vector,
            array_to_string(tsvector_to_array(u.skill_keywords), ' ') AS keyword_text
          FROM auth_users au
          JOIN users u ON u.member_id = au.member_id
          WHERE au.id = :engineer_user_id
        ),
        ticket_candidates AS (
          SELECT
            pt.ticket_key,
            pt.title,
            pt.description,
            pt.priority::text AS priority,
            pt.type::text AS type,
            pt.labels,
            pt.due_date,
            pt.assignee_id,
            pbc.name AS column_name,
            t.ticket_vector
          FROM project_tickets pt
          JOIN project_board_columns pbc
            ON pbc.id = pt.column_id
          JOIN tickets t
            ON t.ticket_id = pt.ticket_key
          WHERE pt.project_id = :project_id
            AND lower(pbc.name) NOT IN ('done', 'complete', 'completed', 'closed', 'resolved')
            AND (pt.assignee_id IS NULL OR pt.assignee_id = :engineer_user_id)
        )
        SELECT
          tc.ticket_key,
          tc.title,
          tc.description,
          tc.priority,
          tc.type,
          tc.labels,
          tc.due_date,
          tc.assignee_id,
          tc.column_name,
          1 - (e.profile_vector <=> tc.ticket_vector) AS semantic_similarity,
          ts_rank(
            to_tsvector('english', tc.title || ' ' || COALESCE(tc.description, '')),
            plainto_tsquery('english', COALESCE(NULLIF(e.keyword_text, ''), tc.title))
          ) AS lexical_score,
          ((1 - (e.profile_vector <=> tc.ticket_vector)) * 0.75)
            + (
              ts_rank(
                to_tsvector('english', tc.title || ' ' || COALESCE(tc.description, '')),
                plainto_tsquery('english', COALESCE(NULLIF(e.keyword_text, ''), tc.title))
              ) * 0.25
            ) AS recommendation_score
        FROM engineer e
        CROSS JOIN ticket_candidates tc
        ORDER BY recommendation_score DESC, tc.ticket_key ASC
        LIMIT 8
        """
    )
    result = await db.execute(
        query,
        {
            "engineer_user_id": engineer_user_id,
            "project_id": project.id,
        },
    )
    rows = result.mappings().all()

    assignee_names: dict[uuid.UUID, str] = {}
    for row in rows:
        assignee_id = row["assignee_id"]
        if assignee_id is not None and assignee_id not in assignee_names:
            assignee = await _fetch_auth_user(db, assignee_id)
            assignee_names[assignee_id] = f"{assignee.first_name} {assignee.last_name}"
    recommendations = [
        RecommendedTicketResponse(
            ticket_key=row["ticket_key"],
            title=row["title"],
            description=row["description"] or None,
            priority=row["priority"],
            type=row["type"],
            labels=list(row["labels"] or []),
            due_date=row["due_date"],
            semantic_similarity=float(row["semantic_similarity"]),
            lexical_score=float(row["lexical_score"]),
            recommendation_score=float(row["recommendation_score"]),
            assignee_id=row["assignee_id"],
            assignee_name=(
                assignee_names.get(row["assignee_id"])
                if row["assignee_id"] is not None
                else None
            ),
            column_name=row["column_name"],
        )
        for row in rows
    ]
    return EngineerTicketRecommendationsResponse(
        user_id=target_profile.user_id,
        username=target_profile.username,
        first_name=target_profile.first_name,
        last_name=target_profile.last_name,
        active_ticket_count=active_ticket_count,
        has_capacity=active_ticket_count < _CAPACITY_THRESHOLD,
        recommendations=recommendations,
    )


async def apply_ticket_completion_profile_update(
    db: AsyncSession,
    *,
    project: Project,
    ticket: ProjectTicket,
    column: ProjectBoardColumn,
) -> None:
    """Apply one-time ML profile updates when a project ticket is completed."""

    if not _HAS_ML:
        return

    if ticket.assignee_id is None or not is_completion_column_name(column.name):
        return

    assignee = await _fetch_auth_user(db, ticket.assignee_id)
    profile = await _ensure_member_profile(db, assignee)
    await sync_project_ticket_to_ml_tables(db, project, ticket, column)

    keyword_extractor = get_keyword_extractor()
    updater = ProfileUpdater()
    ticket_text = f"{ticket.title} {ticket.description or ''}".strip()
    keywords_text = " ".join(keyword_extractor.extract(ticket_text))

    assignment_result = await db.execute(
        text(
            """
            INSERT INTO assignments (ticket_id, engineer_id, assigned_at)
            VALUES (:ticket_id, :engineer_id, now())
            ON CONFLICT (ticket_id, engineer_id) DO UPDATE
            SET assigned_at = assignments.assigned_at
            RETURNING assignment_id, replayed_at
            """
        ),
        {
            "ticket_id": ticket.ticket_key,
            "engineer_id": profile.member_id,
        },
    )
    assignment = assignment_result.mappings().one()
    if assignment["replayed_at"] is not None:
        await db.commit()
        return

    _, params = updater.build_profile_update_query(
        ticket_id=ticket.ticket_key,
        engineer_id=profile.member_id,
        keywords_text=keywords_text,
    )
    alpha, new_signal, ticket_id, keyword_query_text, engineer_id = params
    await db.execute(
        text(
            """
            UPDATE users
            SET
              profile_vector =
                (array_fill(CAST(:alpha AS real), ARRAY[384])::vector * profile_vector
                 + array_fill(CAST(:new_signal AS real), ARRAY[384])::vector *
                 (SELECT ticket_vector FROM tickets WHERE ticket_id = :ticket_id)),
              skill_keywords =
                skill_keywords || to_tsvector('english', :keyword_query_text),
              tickets_closed_count = tickets_closed_count + 1,
              updated_at = now()
            WHERE member_id = :engineer_id
            """
        ),
        {
            "alpha": alpha,
            "new_signal": new_signal,
            "ticket_id": ticket_id,
            "keyword_query_text": keyword_query_text,
            "engineer_id": engineer_id,
        },
    )
    await db.execute(
        text(
            """
            UPDATE assignments
            SET replayed_at = now()
            WHERE assignment_id = :assignment_id
            """
        ),
        {
            "assignment_id": assignment["assignment_id"],
        },
    )
    await db.commit()
