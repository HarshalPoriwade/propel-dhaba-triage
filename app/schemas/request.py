"""HTTP request schema contracts for the Dhaba support triage service."""

from datetime import datetime
from typing import List
from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import PurchaseStatus, PurchaseType
from app.domain.models import Purchase, Ticket


class PurchaseInput(BaseModel):
    """Input representation of a billing transaction from the in-product help form."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64, description="Unique transaction ID (e.g. pay_A1)")
    type: PurchaseType = Field(description="Plan type: trial or renewal")
    amount_inr: int = Field(
        strict=True,
        ge=0,
        description="Exact integer transaction amount in INR (floats strictly rejected)",
    )
    status: PurchaseStatus = Field(description="Payment status: initiated, failed, or successful")
    at: datetime = Field(description="ISO8601 timestamp of transaction")

    def to_domain(self) -> Purchase:
        """Convert request model to immutable domain Purchase model."""
        return Purchase(
            id=self.id,
            type=self.type,
            amount_inr=self.amount_inr,
            status=self.status,
            at=self.at,
        )


class TicketTriageRequest(BaseModel):
    """Incoming request payload for POST /triage matching dhaba_tickets.json format."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        max_length=64,
        description="Unique ticket identifier (e.g. T-1001)",
    )
    received_at: datetime = Field(
        description="ISO8601 timestamp when ticket arrived at support",
    )
    subject: str = Field(
        min_length=1,
        max_length=500,
        description="Customer-submitted subject line",
    )
    body: str = Field(
        min_length=1,
        max_length=10000,
        description="Untrusted customer message body",
    )
    purchases: List[PurchaseInput] = Field(
        default_factory=list,
        max_length=100,
        description="List of billing transactions known at ticket arrival time",
    )
    app_opens_since_renewal: int = Field(
        strict=True,
        ge=0,
        description="Integer count of app launches since most recent renewal",
    )

    def to_domain(self) -> Ticket:
        """Convert validated request payload into an immutable domain Ticket entity."""
        return Ticket(
            id=self.id,
            received_at=self.received_at,
            subject=self.subject,
            body=self.body,
            purchases=tuple(p.to_domain() for p in self.purchases),
            app_opens_since_renewal=self.app_opens_since_renewal,
        )
