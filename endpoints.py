"""Deployed endpoint configuration for the Ethiopia Tourism platform.

Single source of truth for the live site + API URLs. The application was
developed against ``localhost``; in the deployed environment the host is the
server IP. Only the HOST changes — API stays on port 8080, the site on 3000.

    Site (React frontend) : http://4.240.116.35:3000
    Backend API           : http://4.240.116.35:8080

Override via environment variables if the host/ports change:
    SITE_BASE_URL, API_BASE_URL
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Base URLs (host swapped from localhost -> deployed IP)
# ---------------------------------------------------------------------------

SITE_BASE_URL = os.getenv("SITE_BASE_URL", "http://4.240.116.35:3000").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "http://4.240.116.35:8080").rstrip("/")


def api(path: str) -> str:
    """Build a full backend API URL from a path fragment."""
    return f"{API_BASE_URL}/{path.lstrip('/')}"


def site(path: str = "") -> str:
    """Build a full frontend site URL from a path fragment."""
    return f"{SITE_BASE_URL}/{path.lstrip('/')}"


# ---------------------------------------------------------------------------
# Backend API endpoints (host = deployed IP, port 8080)
# ---------------------------------------------------------------------------

API_ENDPOINTS = {
    # --- Auth ---
    "auth_register": api("/api/auth/register"),
    "auth_login": api("/api/auth/login"),

    # --- Package Request (end user) ---
    "package_requests_mine": api("/api/package-requests/mine"),
    "package_request_detail": api("/api/package-requests/{id}"),
    "package_request_visa": api("/api/package-requests/{id}/visa"),
    "flight_preferences": api("/api/package-requests/{id}/flight/preferences"),
    "flight_pay": api("/api/package-requests/{id}/flight/pay"),
    "flight_reject": api("/api/package-requests/{id}/flight/reject"),
    "flight_book_self": api("/api/package-requests/{id}/flight/book-self"),
    "hotel_reject": api("/api/package-requests/{id}/hotel/reject"),
    "hotel_book_self": api("/api/package-requests/{id}/hotel/book-self"),
    "hotel_preferences": api("/api/package-requests/{id}/hotel/preferences"),
    "hotel_pay": api("/api/package-requests/{id}/hotel/pay"),
    "visa_pay_family": api("/api/package-requests/{id}/visa/pay-family"),
    "transport_pay": api("/api/package-requests/{id}/transport/pay"),
    "transport_reject": api("/api/package-requests/{id}/transport/reject"),
    "attraction_pay": api("/api/package-requests/{id}/attraction/pay"),
    "attraction_reject": api("/api/package-requests/{id}/attraction/reject"),

    # --- Admin ---
    "admin_login": api("/api/admin/login"),
    "admin_package_requests": api("/api/admin/package-requests"),
    "admin_package_request_detail": api("/api/admin/package-requests/{id}"),
    "admin_package_request_costs": api("/api/admin/package-requests/{id}/costs"),
    "admin_cost_delete": api("/api/admin/package-requests/{id}/costs/{costId}"),
    "admin_visas_process": api("/api/admin/package-requests/{id}/visas/process"),
    "admin_visas_reject": api("/api/admin/package-requests/{id}/visas/reject"),
    "admin_flight_book": api("/api/admin/package-requests/{id}/flight/book"),
    "admin_hotel_book": api("/api/admin/package-requests/{id}/hotel/book"),
    "admin_transport_book": api("/api/admin/package-requests/{id}/transport/book"),
    "admin_attraction_book": api("/api/admin/package-requests/{id}/attraction/book"),
    "admin_organizer": api("/api/admin/package-requests/{id}/organizer"),
    "admin_organizers": api("/api/admin/organizers"),
    "admin_organizer_detail": api("/api/admin/organizers/{id}"),
    "admin_organizer_requests": api("/api/admin/organizers/{id}/package-requests"),
    "admin_organizer_approve": api("/api/admin/organizers/{id}/approve"),
    "admin_notifications": api("/api/admin/notifications"),
    "admin_notification_read": api("/api/admin/notifications/{id}/read"),

    # --- Organizer ---
    "organizer_register": api("/api/organizer/register"),
    "organizer_login": api("/api/organizer/login"),
    "organizer_package_requests": api("/api/organizer/package-requests"),
    "organizer_package_request_detail": api("/api/organizer/package-requests/{id}"),
    "organizer_arrival_status": api(
        "/api/organizer/package-requests/{id}/arrival-status"
    ),
}


# ---------------------------------------------------------------------------
# Frontend page routes (host = deployed IP, port 3000)
#
# NOTE: These are the pages the bot can navigate a user to. They are best-guess
# placeholders derived from the API structure; correct them to match the real
# React routes on the site. This map is the single point of change.
# ---------------------------------------------------------------------------

SITE_PAGES = {
    "home": site("/"),
    "login": site("/login"),
    "register": site("/register"),
    "my_packages": site("/package-requests"),
    "visa": site("/visa"),
    "packages": site("/packages"),
}
