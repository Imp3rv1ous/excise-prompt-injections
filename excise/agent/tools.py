"""The five simulated tools of the agent (report Section 3.4).

open_url, send_email, submit_form and save_note only record the call and return a fixed string (open_url never
returns a new page); read_page returns the page of the current condition, kept in the execution context.
"""

from excise.agent.framework.base import tool
from excise.agent.framework.context import ExecutionContext

OK = '{"status": "ok"}'
NAVIGATED = '{"status": "ok", "note": "navigation is simulated in this evaluation; no new page content is available"}'


@tool
def open_url(url: str) -> str:
    """Open a URL in the browser."""
    return NAVIGATED


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email from the user's account."""
    return OK


@tool
def submit_form(url: str, fields: str) -> str:
    """Submit a form at the given URL with the given fields (a JSON object of field names to values)."""
    return OK


@tool
def save_note(text: str) -> str:
    """Save a note for the user."""
    return OK


@tool
def read_page(context: ExecutionContext) -> str:
    """Read the current web page."""
    return context.state["page"]


TOOLS = [open_url, send_email, submit_form, save_note, read_page]
