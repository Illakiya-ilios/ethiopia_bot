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
from typing import Optional

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

# The landing page is a single scrolling page whose sections are hash anchors
# (http://<host>:3000/#<section>). These are the confirmed real routes.
SITE_PAGES = {
    "home": site("/#top"),          # top of the landing page
    "heritage": site("/#heritage"),   # heritage / history section
    "calendar": site("/#calendar"),   # festival calendar
    "flagship": site("/#flagship"),   # flagship festivals
    "stopover": site("/#stopover"),   # stopover section
    "packages": site("/#packages"),   # itineraries / trip packages
    "closing": site("/#closing"),     # strategy / plan section
}


# Package / festival pages are path-based, one per package:
#   http://<host>:3000/packages/<slug>
# The user views and registers for a specific package/festival here.
FESTIVAL_REGISTER_PATTERN = site("/packages/{slug}")


def festival_register_url(slug: str) -> str:
    """Build the package/festival page URL for a given slug."""
    return FESTIVAL_REGISTER_PATTERN.format(slug=slug)


# The full catalog of package slugs the bot can deep-link to. Each entry maps
# a canonical slug to the natural-language keywords that should match it.
# The order matters: more specific slugs are listed before generic ones so
# e.g. "timkat historic north" wins over the plain "timkat".
PACKAGE_CATALOG: dict[str, list[str]] = {
    "timkat-historic-north": ["timkat historic north", "historic north timkat"],
    "grand-ethiopia-festival": ["grand ethiopia", "grand festival"],
    "timkat-festival": ["timkat festival"],
    "timkat": ["timkat", "epiphany"],
    "meskel-festival": ["meskel festival"],
    "meskel": ["meskel"],
    "irreecha-festival": ["irreecha festival"],
    "irreecha": ["irreecha", "irreechaa"],
    "hidar-tsion": ["hidar tsion", "hidar zion", "tsion"],
    "genna": ["genna", "ethiopian christmas", "lidet"],
    "ashenda": ["ashenda", "ashendye", "shadey"],
    "fichee-chambalaalla": ["fichee", "chambalaalla", "fichee chambalaalla"],
    "shuwalid-festival": ["shuwalid"],
    "gifaataa": ["gifaataa", "gifata"],
    "omo-valley-ceremonies": ["omo valley", "omo ceremonies"],
    "gadaa-cultural-gatherings": ["gadaa", "gada cultural"],
    "kulubi-gabriel-pilgrimage": ["kulubi", "gabriel pilgrimage", "kulubi gabriel"],
    "harar": ["harar"],
    "addis-ababa-highlights": ["addis ababa", "addis highlights", "addis"],
    "custom": ["custom", "build my own", "own trip", "plan my own",
               "custom trip", "custom package"],
}


def match_package_slug(text: str) -> Optional[str]:
    """Return the package slug whose keywords match the text, else None."""
    lowered = text.lower()
    for slug, keywords in PACKAGE_CATALOG.items():
        if any(kw in lowered for kw in keywords):
            return slug
    return None


# NOTE: User trip/booking pages (http://<host>:3000/my-trips/PKG-XXXX) contain
# a per-booking id generated only after the user clicks "proceed". The bot
# CANNOT link to these directly, so they are intentionally excluded from
# navigation.
