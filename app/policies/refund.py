"""Deterministic business policy gate governing refund authorization.

CRITICAL FINANCIAL SAFETY BOUNDARY:
This module is the sole decider of financial refund authorization.
The LLM has ZERO financial authority.

Under no circumstances may an LLM output:
- directly authorize a refund
- specify or override a refund amount
- determine settlement or payment status
- execute payments or cancellations

All monetary amounts are strictly derived from trusted billing transaction
records (ticket.purchases) where status == PurchaseStatus.SUCCESSFUL.

NOTE ON DHABA BUSINESS POLICY SPECIFICATIONS:
The Propel take-home assignment does NOT define an authoritative refund policy table.
Where source specifications are undefined, the safe deterministic default is:
- NO automatic financial action
- Escalate to human review (POLICY_UNDEFINED_MANUAL_REVIEW)

Any specific auto-refund entitlement (such as zero-usage renewal within 48h)
is an IMPLEMENTATION ASSUMPTION, disabled by default in RefundPolicyConfig,
and explicitly documented as such.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import PurchaseStatus, PurchaseType, Severity
from app.domain.models import Purchase, RefundDecision, Ticket
from app.schemas.llm import LLMPerceptionOutput


class RefundReasonCode(str, Enum):
    """Machine-readable reason codes for deterministic refund decisions.

    These codes accurately describe factual determinations without implying
    authoritative Dhaba business policies that were not defined in the source.
    """

    NO_REFUND_REQUEST = "NO_REFUND_REQUEST"
    NO_SUCCESSFUL_PURCHASE = "NO_SUCCESSFUL_PURCHASE"
    PAYMENT_NOT_CONFIRMED = "PAYMENT_NOT_CONFIRMED"
    CONFLICTING_PAYMENT_EVIDENCE = "CONFLICTING_PAYMENT_EVIDENCE"
    HIGH_RISK_MANUAL_REVIEW = "HIGH_RISK_MANUAL_REVIEW"
    POLICY_UNDEFINED_MANUAL_REVIEW = "POLICY_UNDEFINED_MANUAL_REVIEW"
    REFUND_AUTHORIZED_BY_POLICY = "REFUND_AUTHORIZED_BY_POLICY"


class RefundPolicyConfig(BaseModel):
    """Configuration for deterministic refund policy rules.

    IMPLEMENTATION ASSUMPTION NOTICE:
    The Dhaba take-home assignment does NOT specify an authoritative refund policy.
    By default, allow_auto_refund_zero_usage_renewal is False, ensuring that no
    unauthorized financial commitments occur without explicit human review.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # IMPLEMENTATION ASSUMPTION: If True, allows auto-refund for renewals with 0 app opens.
    # Disabled by default (False) to strictly adhere to assignment-defined entitlements.
    allow_auto_refund_zero_usage_renewal: bool = False

    # Maximum hours between renewal timestamp and ticket arrival to be eligible for zero-usage refund
    zero_usage_window_hours: int = 48


class RefundPolicyResult(BaseModel):
    """Strongly typed outcome of deterministic refund policy evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    should_refund: bool = Field(
        description="Whether a refund is authorized under deterministic business policy"
    )
    amount_inr: int = Field(
        ge=0,
        description="Authorized integer INR refund amount (0 if not authorized)"
    )
    reason_code: RefundReasonCode = Field(
        description="Machine-readable policy decision code"
    )
    reason: str = Field(
        min_length=1,
        description="Human-readable factual justification for the decision"
    )
    eligible_purchase_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Transaction IDs authorized for refund"
    )
    requires_human_review: bool = Field(
        description="True if human intervention or reconciliation is mandatory"
    )

    @model_validator(mode="after")
    def validate_financial_invariants(self) -> "RefundPolicyResult":
        """Enforce strict mathematical invariants on financial authorizations."""
        if not self.should_refund:
            if self.amount_inr != 0:
                raise ValueError(
                    f"amount_inr must be 0 when should_refund is False, got {self.amount_inr}"
                )
            if len(self.eligible_purchase_ids) > 0:
                raise ValueError(
                    f"eligible_purchase_ids must be empty when should_refund is False, got {self.eligible_purchase_ids}"
                )
        else:
            if self.amount_inr <= 0:
                raise ValueError(
                    f"amount_inr must be > 0 when should_refund is True, got {self.amount_inr}"
                )
            if len(self.eligible_purchase_ids) == 0:
                raise ValueError(
                    "eligible_purchase_ids must not be empty when should_refund is True"
                )
        return self

    def to_domain_decision(self) -> RefundDecision:
        """Convert policy result to core domain RefundDecision."""
        return RefundDecision(
            should_refund=self.should_refund,
            amount_inr=self.amount_inr,
            reason=f"[{self.reason_code.value}] {self.reason}",
            eligible_purchase_ids=self.eligible_purchase_ids,
        )


def _detect_high_risk(ticket: Ticket, perception: Optional[LLMPerceptionOutput]) -> bool:
    """Detect fraud, cyber cell, legal, or severe financial risk signals."""
    if perception is not None and perception.severity == Severity.CRITICAL:
        return True

    text_to_check = f"{ticket.subject} {ticket.body}".lower()
    high_risk_keywords = (
        "fraud",
        "cyber cell",
        "cybercrime",
        "police",
        "lawyer",
        "court",
        "legal notice",
        "unauthorized charge",
        "stolen card",
        "identity theft",
    )
    return any(keyword in text_to_check for keyword in high_risk_keywords)


def _detect_conflicting_evidence(
    ticket: Ticket,
    perception: Optional[LLMPerceptionOutput],
) -> Tuple[bool, str]:
    """Detect discrepancies between customer billing claims and actual ledger records."""
    text_to_check = f"{ticket.subject} {ticket.body}".lower()
    claimed_issue = perception.claimed_issue.lower() if perception else ""
    full_text = f"{text_to_check} {claimed_issue}"

    # Check for duplicate / double charge claims (e.g. T-1009)
    double_charge_markers = (
        "double charge",
        "charged twice",
        "debited twice",
        "deducted twice",
        "twice on the same day",
        "took 249 twice",
        "charged 2 times",
        "paid twice",
    )
    is_claiming_double = any(marker in full_text for marker in double_charge_markers)

    if is_claiming_double:
        successful = [p for p in ticket.purchases if p.status == PurchaseStatus.SUCCESSFUL]
        failed = [p for p in ticket.purchases if p.status == PurchaseStatus.FAILED]

        # Customer claims double charge, but records show 1 successful and >= 1 failed
        if len(successful) == 1 and len(failed) >= 1:
            failed_ids = ", ".join(p.id for p in failed)
            return True, (
                f"Conflicting payment evidence: Customer claims duplicate charge, but billing records "
                f"show only 1 successful capture ({successful[0].id} for INR {successful[0].amount_inr}) "
                f"and failed attempt(s) ({failed_ids}). No duplicate payment was captured. Manual reconciliation required."
            )
        # Customer claims double charge, and multiple successful charges exist
        if len(successful) >= 2:
            return True, (
                f"Conflicting payment evidence: Customer claims duplicate charge and billing records show "
                f"{len(successful)} successful captures. Manual reconciliation required."
            )

    return False, ""


def _hours_between(t1: datetime, t2: datetime) -> float:
    """Safely calculate absolute hours between two datetimes regardless of timezone awareness."""
    if t1.tzinfo is not None and t2.tzinfo is None:
        t2 = t2.replace(tzinfo=t1.tzinfo)
    elif t1.tzinfo is None and t2.tzinfo is not None:
        t1 = t1.replace(tzinfo=t2.tzinfo)
    return (t1 - t2).total_seconds() / 3600.0


def evaluate_refund_policy(
    ticket: Ticket,
    perception: Optional[LLMPerceptionOutput] = None,
    *,
    user_requested_refund: Optional[bool] = None,
    config: Optional[RefundPolicyConfig] = None,
) -> RefundPolicyResult:
    """Deterministically evaluate refund eligibility from trusted billing facts.

    FINANCIAL INVARIANTS:
    1. Returns integer INR amount >= 0.
    2. amount_inr is 0 whenever should_refund is False.
    3. amount_inr is strictly <= the sum of matching successful purchase records.
    4. Model perception outputs are NEVER used as financial authorization or amounts.
    5. Pure function: idempotent and deterministic for the same ticket and config.

    Args:
        ticket: Trusted ticket domain model with authoritative billing history.
        perception: Optional non-authoritative LLM perception signals.
        user_requested_refund: Explicit override flag for customer intent.
        config: Policy configuration rules (defaults to safe baseline).

    Returns:
        RefundPolicyResult with machine-readable reason_code and decision details.
    """
    cfg = config or RefundPolicyConfig()

    # 1. High-risk / fraud / legal check (Safety priority)
    if _detect_high_risk(ticket, perception):
        return RefundPolicyResult(
            should_refund=False,
            amount_inr=0,
            reason_code=RefundReasonCode.HIGH_RISK_MANUAL_REVIEW,
            reason=(
                "High-risk indicator detected (fraud claim, legal threat, or cyber cell report). "
                "Automatic financial actions are strictly prohibited. Escalated to compliance."
            ),
            eligible_purchase_ids=(),
            requires_human_review=True,
        )

    # 2. Conflicting payment evidence check (e.g. duplicate charge claim vs ledger)
    has_conflict, conflict_msg = _detect_conflicting_evidence(ticket, perception)
    if has_conflict:
        return RefundPolicyResult(
            should_refund=False,
            amount_inr=0,
            reason_code=RefundReasonCode.CONFLICTING_PAYMENT_EVIDENCE,
            reason=conflict_msg,
            eligible_purchase_ids=(),
            requires_human_review=True,
        )

    # 3. Determine whether a refund was requested
    # Perception signals provide untrusted intent, NOT authorization.
    if user_requested_refund is not None:
        is_refund_requested = user_requested_refund
    elif perception is not None:
        is_refund_requested = perception.user_requested_refund
    else:
        is_refund_requested = False

    if not is_refund_requested:
        return RefundPolicyResult(
            should_refund=False,
            amount_inr=0,
            reason_code=RefundReasonCode.NO_REFUND_REQUEST,
            reason="No refund requested by customer.",
            eligible_purchase_ids=(),
            requires_human_review=False,
        )

    # 4. If refund was requested, inspect authoritative purchase history
    if not ticket.purchases:
        return RefundPolicyResult(
            should_refund=False,
            amount_inr=0,
            reason_code=RefundReasonCode.NO_SUCCESSFUL_PURCHASE,
            reason="No purchase records exist for this customer account.",
            eligible_purchase_ids=(),
            requires_human_review=True,
        )

    # Check latest purchase status: initiated purchases must NOT be treated as successful charges
    latest_purchase = ticket.purchases[-1]
    if latest_purchase.status == PurchaseStatus.INITIATED:
        return RefundPolicyResult(
            should_refund=False,
            amount_inr=0,
            reason_code=RefundReasonCode.PAYMENT_NOT_CONFIRMED,
            reason=(
                f"Transaction {latest_purchase.id} of INR {latest_purchase.amount_inr} is in 'initiated' status "
                "and has not been confirmed as successful by the payment gateway. Unsettled payments cannot be refunded."
            ),
            eligible_purchase_ids=(),
            requires_human_review=True,
        )

    successful_purchases = [
        p for p in ticket.purchases if p.status == PurchaseStatus.SUCCESSFUL
    ]
    if not successful_purchases:
        failed_purchases = [
            p for p in ticket.purchases if p.status == PurchaseStatus.FAILED
        ]
        if failed_purchases:
            return RefundPolicyResult(
                should_refund=False,
                amount_inr=0,
                reason_code=RefundReasonCode.PAYMENT_NOT_CONFIRMED,
                reason=(
                    f"Transaction {failed_purchases[-1].id} is recorded as failed. "
                    "No funds were captured; cannot issue refund."
                ),
                eligible_purchase_ids=(),
                requires_human_review=True,
            )
        return RefundPolicyResult(
            should_refund=False,
            amount_inr=0,
            reason_code=RefundReasonCode.NO_SUCCESSFUL_PURCHASE,
            reason="No successful charges recorded on this account.",
            eligible_purchase_ids=(),
            requires_human_review=True,
        )

    latest_successful = successful_purchases[-1]

    # 5. Check if authoritative policy allows auto-refund
    # If the policy entitlement is not defined in the source, safe default is manual review.
    if not cfg.allow_auto_refund_zero_usage_renewal:
        return RefundPolicyResult(
            should_refund=False,
            amount_inr=0,
            reason_code=RefundReasonCode.POLICY_UNDEFINED_MANUAL_REVIEW,
            reason=(
                f"Customer requested refund for verified charge {latest_successful.id} (INR {latest_successful.amount_inr}), "
                "but Dhaba refund policy entitlement is not authoritatively defined in system specifications. "
                "Automatic financial action withheld; escalated to human review."
            ),
            eligible_purchase_ids=(),
            requires_human_review=True,
        )

    # 6. IMPLEMENTATION ASSUMPTION:
    # If explicitly enabled via config, zero-usage renewal (<48h, 0 app opens) authorizes a refund.
    if latest_successful.type == PurchaseType.RENEWAL and ticket.app_opens_since_renewal == 0:
        hours_since_renewal = _hours_between(ticket.received_at, latest_successful.at)
        if 0 <= hours_since_renewal <= cfg.zero_usage_window_hours:
            return RefundPolicyResult(
                should_refund=True,
                amount_inr=latest_successful.amount_inr,
                reason_code=RefundReasonCode.REFUND_AUTHORIZED_BY_POLICY,
                reason=(
                    f"Automatic refund of INR {latest_successful.amount_inr} authorized under zero-usage "
                    f"renewal policy for transaction {latest_successful.id} (0 app opens since renewal)."
                ),
                eligible_purchase_ids=(latest_successful.id,),
                requires_human_review=False,
            )

    # Ineligible under enabled policy rule (e.g. usage > 0, trial, or expired window)
    return RefundPolicyResult(
        should_refund=False,
        amount_inr=0,
        reason_code=RefundReasonCode.POLICY_UNDEFINED_MANUAL_REVIEW,
        reason=(
            f"Customer requested refund for transaction {latest_successful.id} (INR {latest_successful.amount_inr}), "
            f"but account does not qualify for automatic zero-usage renewal refund "
            f"(type={latest_successful.type.value}, app_opens={ticket.app_opens_since_renewal}). "
            "Manual review required."
        ),
        eligible_purchase_ids=(),
        requires_human_review=True,
    )
