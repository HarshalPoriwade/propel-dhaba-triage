# Dhaba Support Ticket Triage Service

A production-minded, resilient microservice designed to triage incoming customer support tickets for the "Dhaba" subscription food app. The service separates untrusted LLM perception from deterministic business rules, enforcing strict schema validation, locked financial refund policies, idempotency, and zero-PII logging while supporting offline fixture-based replay without external API tokens.

## Current Development Status
**Step 1 Complete — Project Scaffolding and Configuration.**  
The core configuration, settings validation, virtual environment dependencies, and empty FastAPI application factory have been established and verified. No domain models, business logic, endpoints, or persistence layers have been initialized yet.

## Python Version Requirement
- **Python 3.10+** (Tested on Python 3.10.11)

## Local Environment Setup

1. **Activate the virtual environment:**
   - Windows PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Windows CMD:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables (Optional):**
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   *Note: By default, `MODEL_MODE=fixture`, so no API key is required to run the service or tests.*

## Running the Application

To start the empty FastAPI application locally:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Running Tests

To run the verification test suite:

```bash
pytest
```
