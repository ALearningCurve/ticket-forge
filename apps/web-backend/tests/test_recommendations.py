"""Tests for semantic recommendation routes and completion sync."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from web_backend.schemas.tickets import TicketMoveRequest, TicketUpdateRequest
from web_backend.models.project import Project, ProjectBoardColumn, ProjectMember
from web_backend.models.ticket import ProjectTicket
from web_backend.services.tickets import move_ticket, update_ticket

pytestmark = pytest.mark.asyncio

VALID_SIGNUP = {
    "username": "johndoe",
    "first_name": "John",
    "last_name": "Doe",
    "email": "john@example.com",
    "password": "SecurePass1",
}

VALID_SIGNUP_2 = {
    "username": "janedoe",
    "first_name": "Jane",
    "last_name": "Doe",
    "email": "jane@example.com",
    "password": "SecurePass1",
}


class TestRecommendationRoutes:
    """Recommendation endpoint coverage."""

    async def test_ticket_recommendations_require_authentication(
        self, client: AsyncClient
    ) -> None:
        """Reject unauthenticated engineer recommendation requests."""
        response = await client.get(
            "/api/v1/projects/demo/tickets/DEMO-1/recommendations/engineers"
        )
        assert response.status_code == 401

    async def test_ticket_recommendations_return_payload(
        self, client: AsyncClient
    ) -> None:
        """Return engineer recommendations for an authenticated member."""
        signup = await client.post("/api/v1/auth/signup", json=VALID_SIGNUP)
        token = signup.json()["access_token"]

        with patch(
            "web_backend.api.v1.recommendations.recommend_engineers_for_ticket",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    ticket_key="DEMO-1",
                    recommendations=[
                        SimpleNamespace(
                            member_id=101,
                            user_id=uuid.uuid4(),
                            username="janedoe",
                            first_name="Jane",
                            last_name="Doe",
                            email="jane@example.com",
                            active_ticket_count=1,
                            capacity_score=0.66,
                            has_capacity=True,
                            semantic_similarity=0.88,
                            lexical_score=0.5,
                            recommendation_score=0.81,
                        )
                    ],
                )
            ),
        ):
            response = await client.get(
                "/api/v1/projects/demo/tickets/DEMO-1/recommendations/engineers",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["ticket_key"] == "DEMO-1"
        assert len(body["recommendations"]) == 1
        assert body["recommendations"][0]["username"] == "janedoe"

    async def test_engineer_recommendations_return_payload(
        self, client: AsyncClient
    ) -> None:
        """Return ticket recommendations for an authenticated member."""
        signup = await client.post("/api/v1/auth/signup", json=VALID_SIGNUP)
        token = signup.json()["access_token"]
        engineer_id = uuid.uuid4()

        with patch(
            "web_backend.api.v1.recommendations.recommend_tickets_for_engineer",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    user_id=engineer_id,
                    username="janedoe",
                    first_name="Jane",
                    last_name="Doe",
                    active_ticket_count=2,
                    has_capacity=True,
                    recommendations=[
                        SimpleNamespace(
                            ticket_key="DEMO-2",
                            title="Fix login bug",
                            description="Repair auth refresh handling",
                            priority="high",
                            type="bug",
                            labels=["backend", "security"],
                            size_bucket="M",
                            due_date=None,
                            semantic_similarity=0.91,
                            lexical_score=0.42,
                            recommendation_score=0.79,
                            assignee_id=None,
                            assignee_name=None,
                            column_name="To Do",
                        )
                    ],
                )
            ),
        ):
            response = await client.get(
                f"/api/v1/projects/demo/members/{engineer_id}/recommendations/tickets",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "janedoe"
        assert len(body["recommendations"]) == 1
        assert body["recommendations"][0]["ticket_key"] == "DEMO-2"


class TestCompletionSync:
    """Completion-triggered profile update coverage."""

    async def test_move_to_done_triggers_profile_update(self, db_session) -> None:
        """Moving an assigned ticket into Done triggers ML profile sync."""
        owner_id = uuid.uuid4()
        assignee_id = uuid.uuid4()
        project_id = uuid.uuid4()
        todo_column_id = uuid.uuid4()
        done_column_id = uuid.uuid4()

        owner = await self._create_user(
            db_session=db_session,
            user_id=owner_id,
            username=VALID_SIGNUP["username"],
            email=VALID_SIGNUP["email"],
            first_name=VALID_SIGNUP["first_name"],
            last_name=VALID_SIGNUP["last_name"],
        )
        await self._create_user(
            db_session=db_session,
            user_id=assignee_id,
            username=VALID_SIGNUP_2["username"],
            email=VALID_SIGNUP_2["email"],
            first_name=VALID_SIGNUP_2["first_name"],
            last_name=VALID_SIGNUP_2["last_name"],
        )

        project = Project(
            id=project_id,
            name="Demo Project",
            slug="demo-project",
            description="Semantic search test",
            created_by=owner.id,
        )
        todo_column = ProjectBoardColumn(
            id=todo_column_id,
            project_id=project.id,
            name="To Do",
            position=0,
        )
        done_column = ProjectBoardColumn(
            id=done_column_id,
            project_id=project.id,
            name="Done",
            position=1,
        )
        db_session.add_all(
            [
                project,
                todo_column,
                done_column,
                ProjectMember(project_id=project.id, user_id=owner.id, role="owner"),
                ProjectMember(
                    project_id=project.id, user_id=assignee_id, role="member"
                ),
                ProjectTicket(
                    project_id=project.id,
                    column_id=todo_column.id,
                    assignee_id=assignee_id,
                    created_by=owner.id,
                    ticket_key="DEMO-1",
                    title="Finish semantic search",
                    description="Hook recommendation APIs to the board",
                    priority="high",
                    type="story",
                    labels=["backend"],
                    position=0,
                ),
            ]
        )
        await db_session.commit()

        with (
            patch(
                "web_backend.services.tickets.sync_project_ticket_to_ml_tables",
                new=AsyncMock(),
            ) as sync_ticket,
            patch(
                "web_backend.services.tickets.apply_ticket_completion_profile_update",
                new=AsyncMock(),
            ) as apply_update,
        ):
            moved_ticket = await move_ticket(
                db_session,
                "demo-project",
                "DEMO-1",
                TicketMoveRequest(column_id=done_column.id, position=0),
                owner.id,
            )

        assert moved_ticket.column_id == done_column.id
        sync_ticket.assert_awaited_once()
        apply_update.assert_awaited_once()

    async def test_assigning_ticket_syncs_ml_assignment_immediately(
        self, db_session
    ) -> None:
        """Saving a new assignee syncs the ML-side assignment state immediately."""
        owner_id = uuid.uuid4()
        assignee_id = uuid.uuid4()
        project_id = uuid.uuid4()
        todo_column_id = uuid.uuid4()

        owner = await self._create_user(
            db_session=db_session,
            user_id=owner_id,
            username=VALID_SIGNUP["username"],
            email=VALID_SIGNUP["email"],
            first_name=VALID_SIGNUP["first_name"],
            last_name=VALID_SIGNUP["last_name"],
        )
        await self._create_user(
            db_session=db_session,
            user_id=assignee_id,
            username=VALID_SIGNUP_2["username"],
            email=VALID_SIGNUP_2["email"],
            first_name=VALID_SIGNUP_2["first_name"],
            last_name=VALID_SIGNUP_2["last_name"],
        )

        project = Project(
            id=project_id,
            name="Demo Project",
            slug="demo-project",
            description="Immediate assignment sync",
            created_by=owner.id,
        )
        todo_column = ProjectBoardColumn(
            id=todo_column_id,
            project_id=project.id,
            name="To Do",
            position=0,
        )
        ticket = ProjectTicket(
            project_id=project.id,
            column_id=todo_column.id,
            assignee_id=None,
            created_by=owner.id,
            ticket_key="DEMO-2",
            title="Assign me once",
            description="This should sync on the first save",
            priority="medium",
            type="task",
            labels=["backend"],
            position=0,
        )
        db_session.add_all(
            [
                project,
                todo_column,
                ProjectMember(project_id=project.id, user_id=owner.id, role="owner"),
                ProjectMember(
                    project_id=project.id, user_id=assignee_id, role="member"
                ),
                ticket,
            ]
        )
        await db_session.commit()

        with patch(
            "web_backend.services.tickets.sync_project_ticket_to_ml_tables",
            new=AsyncMock(),
        ) as sync_ticket:
            updated_ticket = await update_ticket(
                db_session,
                "demo-project",
                "DEMO-2",
                TicketUpdateRequest.model_validate({"assignee_id": assignee_id}),
                owner.id,
            )

        assert updated_ticket.assignee is not None
        assert updated_ticket.assignee.id == assignee_id
        sync_ticket.assert_awaited_once()

    @staticmethod
    async def _create_user(
        *,
        db_session,
        user_id: uuid.UUID,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
    ):
        """Create a raw auth user for service-layer tests."""
        from web_backend.models.user import AuthUser

        user = AuthUser(
            id=user_id,
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            password_hash="hashed-password",
        )
        db_session.add(user)
        await db_session.flush()
        return user


class TestEngineerTicketRecommendations:
    """Engineer ticket recommendation service coverage."""

    async def test_recommendations_do_not_full_sync_project_tickets(
        self, db_session
    ) -> None:
        """Engineer ticket recommendations should not trigger a full project sync."""
        from web_backend.services.recommendations import recommend_tickets_for_engineer

        current_user_id = uuid.uuid4()
        engineer_user_id = uuid.uuid4()
        project_id = uuid.uuid4()
        project = SimpleNamespace(id=project_id)
        engineer_profile = SimpleNamespace(
            user_id=engineer_user_id,
            username="janedoe",
            first_name="Jane",
            last_name="Doe",
            email="jane@example.com",
            member_id=101,
        )
        query_result = SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: []))

        with (
            patch(
                "web_backend.services.recommendations._get_project_for_member_validation",
                new=AsyncMock(return_value=project),
            ),
            patch(
                "web_backend.services.recommendations._get_target_project_member",
                new=AsyncMock(return_value=SimpleNamespace(user_id=engineer_user_id)),
            ),
            patch(
                "web_backend.services.recommendations.ensure_project_member_profiles",
                new=AsyncMock(return_value={engineer_user_id: engineer_profile}),
            ),
            patch(
                "web_backend.services.recommendations.sync_project_tickets_for_recommendations",
                new=AsyncMock(),
            ) as full_sync,
            patch.object(
                db_session,
                "execute",
                new=AsyncMock(
                    side_effect=[SimpleNamespace(scalar_one=lambda: 0), query_result]
                ),
            ),
        ):
            response = await recommend_tickets_for_engineer(
                db_session, "demo-project", engineer_user_id, current_user_id
            )

        full_sync.assert_not_awaited()
        assert response.user_id == engineer_user_id
        assert response.recommendations == []
