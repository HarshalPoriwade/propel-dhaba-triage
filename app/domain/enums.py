"""Domain enumerations directly derived from the Dhaba assignment requirements."""

from enum import Enum


class Category(str, Enum):
    """Categorization for incoming customer support tickets.

    Derived from the 12 representative ticket archetypes in dhaba_tickets.json:
    - BILLING: Charges, duplicate charges, payment status, receipts, invoices.
    - CANCELLATION: Trial cancellation, subscription stoppage, autopay inquiries.
    - TECHNICAL: App crashes, freezes, bugs, device/OS specific failures.
    - ACCOUNT: Order history sync, login issues, account data restoration.
    - FEATURE_REQUEST: Language support, functionality suggestions.
    - COMPLAINT: General dissatisfaction, review threats, severe service complaints.
    - GENERAL: Pre-sales questions, informational inquiries.
    """

    BILLING = "billing"
    CANCELLATION = "cancellation"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    COMPLAINT = "complaint"
    GENERAL = "general"


class Severity(str, Enum):
    """Documented severity scale for customer triage prioritization.

    - LOW: General inquiries, feature requests, invoice requests.
    - MEDIUM: Routine cancellations, minor account issues, standard billing queries.
    - HIGH: App crashes for paying users, debited payments without service, churn threats.
    - CRITICAL: Legal threats, regulatory/cyber cell reports, severe fraud allegations.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PurchaseType(str, Enum):
    """Type of transaction recorded in Dhaba billing system."""

    TRIAL = "trial"
    RENEWAL = "renewal"


class PurchaseStatus(str, Enum):
    """Settlement status of a transaction in Dhaba billing system."""

    INITIATED = "initiated"
    FAILED = "failed"
    SUCCESSFUL = "successful"
