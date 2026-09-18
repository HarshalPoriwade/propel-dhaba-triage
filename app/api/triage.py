"""HTTP route handlers for ticket triage."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_triage_service
from app.repositories.triage import CorruptRecordError, RepositoryError
from app.schemas.request import TicketTriageRequest
from app.schemas.response import TicketTriageResponse
from app.services.triage import TicketInProgressError, TriageService

router = APIRouter(tags=["triage"])


@router.post(
    "/triage",
    response_model=TicketTriageResponse,
    status_code=status.HTTP_200_OK,
    summary="Triage customer support ticket",
    description="Accepts customer ticket JSON and returns category, severity, refund policy, and reply draft.",
)
async def triage_ticket_endpoint(
    request: TicketTriageRequest,
    service: TriageService = Depends(get_triage_service),
) -> TicketTriageResponse:
    """Triage incoming customer ticket.

    - Validates request payload against Dhaba schema.
    - Idempotently prevents duplicate LLM calls or duplicate refunds.
    - Deterministically calculates refund decisions independently of the LLM.
    - Returns structured triage response.
    """
    try:
        return await service.triage_ticket(request)
    except TicketInProgressError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(err),
        ) from err
    except CorruptRecordError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored triage record is corrupted and cannot be retrieved.",
        ) from err
    except RepositoryError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A persistence error occurred while saving the triage result.",
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while processing the ticket.",
        ) from err
