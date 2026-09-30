"""SQLAlchemy data models + SQLite bootstrap for the Ethiopia tourism platform.

This module defines the relational schema (mirroring the production MySQL
tables), an engine/session factory backed by SQLite, and a synthetic-data
seeder useful for local development and testing.

Type mapping notes (MySQL -> portable SQLAlchemy):
    - datetime(6)   -> DateTime
    - bit(1)        -> Boolean
    - enum(...)     -> SQLEnum(PyEnum)
    - longtext      -> Text
    - decimal(38,2) -> Numeric(38, 2)
    - varchar(n)    -> String(n)

Run directly to (re)create the DB and seed it:

    python model.py            # create schema + seed synthetic data
    python model.py --reset    # drop everything first, then recreate + seed
"""

from __future__ import annotations

import argparse
import enum
import os
import random
import uuid
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import List

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    Time,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)


# ============================================================
# 1. ENGINE / SESSION
# ============================================================

# Load .env BEFORE reading DATABASE_URL. This module is imported (transitively)
# at server startup before any other load_dotenv() runs, so without this the
# env var would be unset and we'd silently fall back to SQLite.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///tourism.db")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    # Required for SQLite when used across threads (e.g. web servers).
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


# ============================================================
# 2. BASE
# ============================================================


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# 3. ENUMS
# ============================================================


class UserRole(enum.Enum):
    ADMIN = "ADMIN"
    USER = "USER"


# ============================================================
# 4. MODELS
# ============================================================


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    full_name: Mapped[str | None] = mapped_column(String(255))
    mobile: Mapped[str | None] = mapped_column(String(255))
    password_hash: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(
        SQLEnum(UserRole), default=UserRole.USER
    )

    package_requests: Mapped[List["PackageRequest"]] = relationship(
        back_populates="user"
    )


class PackageRequest(Base):
    __tablename__ = "package_requests"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    adults_female: Mapped[int | None] = mapped_column(Integer)
    adults_male: Mapped[int | None] = mapped_column(Integer)
    airline: Mapped[str | None] = mapped_column(String(255))
    arrival_date: Mapped[date | None] = mapped_column(Date)
    arrival_time: Mapped[time | None] = mapped_column(Time)
    children_ages: Mapped[str | None] = mapped_column(String(500))
    destinations: Mapped[str | None] = mapped_column(String(1000))
    flight_no: Mapped[str | None] = mapped_column(String(255))
    hotel_star_preference: Mapped[str | None] = mapped_column(String(255))
    infants: Mapped[int | None] = mapped_column(Integer)
    inter_transfer_cabs_required: Mapped[str | None] = mapped_column(String(255))
    itinerary_size: Mapped[str | None] = mapped_column(String(255))
    legs: Mapped[int | None] = mapped_column(Integer)
    package_name: Mapped[str | None] = mapped_column(String(255))
    planned_departure_date: Mapped[date | None] = mapped_column(Date)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, default=_utcnow)
    tentative_budget_usd: Mapped[str | None] = mapped_column(String(255))
    travel_days_tentative: Mapped[int | None] = mapped_column(Integer)
    travel_modes: Mapped[str | None] = mapped_column(String(500))
    traveller_email: Mapped[str | None] = mapped_column(String(255))
    traveller_name: Mapped[str | None] = mapped_column(String(255))
    traveller_passport: Mapped[str | None] = mapped_column(String(255))
    traveller_whatsapp: Mapped[str | None] = mapped_column(String(255))
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    services_needed: Mapped[str | None] = mapped_column(String(200))
    children_count: Mapped[int | None] = mapped_column(Integer)

    user: Mapped["User | None"] = relationship(back_populates="package_requests")
    costs: Mapped[List["PackageRequestCost"]] = relationship(
        back_populates="package_request", cascade="all, delete-orphan"
    )
    passengers: Mapped[List["PackageRequestPassenger"]] = relationship(
        back_populates="package_request", cascade="all, delete-orphan"
    )
    notifications: Mapped[List["AdminNotification"]] = relationship(
        back_populates="package_request", cascade="all, delete-orphan"
    )
    visa_applications: Mapped[List["VisaApplication"]] = relationship(
        back_populates="package_request", cascade="all, delete-orphan"
    )


class AdminNotification(Base):
    __tablename__ = "admin_notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    message: Mapped[str | None] = mapped_column(String(500))
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    type: Mapped[str | None] = mapped_column(String(255))
    package_request_id: Mapped[str | None] = mapped_column(
        ForeignKey("package_requests.id")
    )

    package_request: Mapped["PackageRequest | None"] = relationship(
        back_populates="notifications"
    )


class PackageRequestCost(Base):
    __tablename__ = "package_request_costs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    amount_usd: Mapped[Decimal | None] = mapped_column(Numeric(38, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    name: Mapped[str | None] = mapped_column(String(255))
    package_request_id: Mapped[str | None] = mapped_column(
        ForeignKey("package_requests.id")
    )

    package_request: Mapped["PackageRequest | None"] = relationship(
        back_populates="costs"
    )


class PackageRequestPassenger(Base):
    __tablename__ = "package_request_passengers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    age: Mapped[int | None] = mapped_column(Integer)
    category: Mapped[str | None] = mapped_column(String(255))
    name: Mapped[str | None] = mapped_column(String(255))
    position: Mapped[int | None] = mapped_column(Integer)
    package_request_id: Mapped[str | None] = mapped_column(
        ForeignKey("package_requests.id")
    )

    package_request: Mapped["PackageRequest | None"] = relationship(
        back_populates="passengers"
    )


class VisaApplication(Base):
    __tablename__ = "visa_applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    accommodation_address: Mapped[str | None] = mapped_column(String(1000))
    arrival_date: Mapped[date | None] = mapped_column(Date)
    country_of_residence: Mapped[str | None] = mapped_column(String(255))
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    departure_date: Mapped[date | None] = mapped_column(Date)
    email: Mapped[str | None] = mapped_column(String(255))
    emergency_contact_name: Mapped[str | None] = mapped_column(String(255))
    emergency_contact_phone: Mapped[str | None] = mapped_column(String(255))
    emergency_contact_relationship: Mapped[str | None] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(255))
    gender: Mapped[str | None] = mapped_column(String(255))
    home_address: Mapped[str | None] = mapped_column(String(1000))
    nationality: Mapped[str | None] = mapped_column(String(255))
    occupation: Mapped[str | None] = mapped_column(String(255))
    passport_expiry_date: Mapped[date | None] = mapped_column(Date)
    passport_issue_date: Mapped[date | None] = mapped_column(Date)
    passport_issuing_country: Mapped[str | None] = mapped_column(String(255))
    passport_no: Mapped[str | None] = mapped_column(String(255))
    passport_photo_content_type: Mapped[str | None] = mapped_column(String(255))
    passport_photo_data: Mapped[str | None] = mapped_column(Text)
    passport_type: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(255))
    place_of_birth: Mapped[str | None] = mapped_column(String(255))
    port_of_entry: Mapped[str | None] = mapped_column(String(255))
    purpose_of_visit: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str | None] = mapped_column(String(255))
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, default=_utcnow)
    visa_type: Mapped[str | None] = mapped_column(String(255))
    package_request_id: Mapped[str | None] = mapped_column(
        ForeignKey("package_requests.id")
    )
    accommodation_city: Mapped[str | None] = mapped_column(String(255))
    accommodation_name: Mapped[str | None] = mapped_column(String(255))
    accommodation_street_address: Mapped[str | None] = mapped_column(String(255))
    accommodation_telephone: Mapped[str | None] = mapped_column(String(255))
    accommodation_type: Mapped[str | None] = mapped_column(String(255))
    address_city: Mapped[str | None] = mapped_column(String(255))
    address_country: Mapped[str | None] = mapped_column(String(255))
    agree_terms: Mapped[bool | None] = mapped_column(Boolean)
    airline: Mapped[str | None] = mapped_column(String(255))
    applicant_category: Mapped[str | None] = mapped_column(String(20))
    applicant_label: Mapped[str | None] = mapped_column(String(40))
    applicant_position: Mapped[int | None] = mapped_column(Integer)
    citizenship: Mapped[str | None] = mapped_column(String(255))
    country_of_birth: Mapped[str | None] = mapped_column(String(255))
    departure_city: Mapped[str | None] = mapped_column(String(255))
    departure_country: Mapped[str | None] = mapped_column(String(255))
    flight_number: Mapped[str | None] = mapped_column(String(255))
    given_name: Mapped[str | None] = mapped_column(String(255))
    passport_copy_content_type: Mapped[str | None] = mapped_column(String(255))
    passport_copy_data: Mapped[str | None] = mapped_column(Text)
    passport_issuing_authority: Mapped[str | None] = mapped_column(String(255))
    passport_number: Mapped[str | None] = mapped_column(String(255))
    rejection_remark: Mapped[str | None] = mapped_column(String(1000))
    street_address: Mapped[str | None] = mapped_column(String(255))
    surname: Mapped[str | None] = mapped_column(String(255))
    visa_validity: Mapped[str | None] = mapped_column(String(255))

    package_request: Mapped["PackageRequest | None"] = relationship(
        back_populates="visa_applications"
    )


# ============================================================
# 5. SCHEMA HELPERS
# ============================================================


def create_all() -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(engine)


def drop_all() -> None:
    """Drop all tables. Destructive — used only by the --reset flag."""
    Base.metadata.drop_all(engine)


# ============================================================
# 6. SYNTHETIC DATA
# ============================================================

_FIRST_NAMES = [
    "Amanuel", "Sara", "Daniel", "Hanna", "Yonas", "Meron", "Bekele",
    "Liya", "Tewodros", "Selam", "Nardos", "Abel", "Rahel", "Kaleb",
]
_LAST_NAMES = [
    "Tesfaye", "Girma", "Bekele", "Alemu", "Haile", "Assefa", "Mekonnen",
    "Desta", "Kebede", "Wolde",
]
_DESTINATIONS = [
    "Lalibela", "Axum", "Gondar", "Bahir Dar", "Simien Mountains",
    "Danakil Depression", "Omo Valley", "Harar", "Addis Ababa",
]
_AIRLINES = ["Ethiopian Airlines", "Kenya Airways", "Emirates", "Qatar Airways"]
_NATIONALITIES = ["Indian", "American", "German", "French", "Kenyan", "British"]
_VISA_STATUSES = ["PENDING", "APPROVED", "REJECTED", "UNDER_REVIEW"]
_COST_ITEMS = ["Flights", "Hotel", "Guide", "Transport", "Entry Fees", "Meals"]


def _rand_name() -> str:
    return f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"


def _rand_email(name: str) -> str:
    handle = name.lower().replace(" ", ".")
    return f"{handle}{random.randint(1, 999)}@example.com"


def _rand_phone() -> str:
    return f"+2519{random.randint(10_000_000, 99_999_999)}"


def seed(session, num_users: int = 5, requests_per_user: int = 2) -> None:
    """Populate the database with deterministic-ish synthetic data.

    Creates users (one admin), package requests with nested costs/passengers,
    admin notifications, and visa applications. Idempotency is not attempted;
    call with a fresh DB or use --reset.
    """
    random.seed(42)

    # One admin account.
    admin = User(
        email="admin@ethiotours.com",
        full_name="Platform Admin",
        mobile=_rand_phone(),
        password_hash="hashed::admin",
        role=UserRole.ADMIN,
    )
    session.add(admin)

    for _ in range(num_users):
        name = _rand_name()
        user = User(
            email=_rand_email(name),
            full_name=name,
            mobile=_rand_phone(),
            password_hash="hashed::password",
            role=UserRole.USER,
        )
        session.add(user)
        session.flush()  # obtain user.id

        for _ in range(requests_per_user):
            _seed_package_request(session, user)

    session.commit()


def _seed_package_request(session, user: User) -> None:
    req_id = str(uuid.uuid4())
    traveller = _rand_name()
    arrival = date.today() + timedelta(days=random.randint(15, 120))
    travel_days = random.randint(4, 14)
    adults_male = random.randint(1, 2)
    adults_female = random.randint(1, 2)
    children = random.randint(0, 3)

    request = PackageRequest(
        id=req_id,
        adults_male=adults_male,
        adults_female=adults_female,
        airline=random.choice(_AIRLINES),
        arrival_date=arrival,
        arrival_time=time(random.randint(6, 22), random.choice([0, 15, 30, 45])),
        children_ages=",".join(
            str(random.randint(1, 15)) for _ in range(children)
        ) or None,
        destinations=", ".join(
            random.sample(_DESTINATIONS, random.randint(2, 4))
        ),
        flight_no=f"ET{random.randint(100, 999)}",
        hotel_star_preference=f"{random.randint(3, 5)}-star",
        infants=random.randint(0, 1),
        inter_transfer_cabs_required=random.choice(["Yes", "No"]),
        itinerary_size=random.choice(["Small", "Medium", "Large"]),
        legs=random.randint(1, 4),
        package_name=f"{random.choice(_DESTINATIONS)} Explorer",
        planned_departure_date=arrival + timedelta(days=travel_days),
        tentative_budget_usd=str(random.randint(1500, 8000)),
        travel_days_tentative=travel_days,
        travel_modes=random.choice(["Air, Road", "Road", "Air"]),
        traveller_email=_rand_email(traveller),
        traveller_name=traveller,
        traveller_passport=f"EP{random.randint(1_000_000, 9_999_999)}",
        traveller_whatsapp=_rand_phone(),
        user_id=user.id,
        services_needed=random.choice(
            ["Visa, Hotel", "Full Package", "Flights only", "Guide, Transport"]
        ),
        children_count=children,
    )
    session.add(request)

    # Costs
    for item in random.sample(_COST_ITEMS, random.randint(3, len(_COST_ITEMS))):
        session.add(
            PackageRequestCost(
                amount_usd=Decimal(random.randint(100, 3000)),
                name=item,
                package_request=request,
            )
        )

    # Passengers
    total = adults_male + adults_female + children
    for position in range(1, total + 1):
        if position <= adults_male + adults_female:
            category, age = "Adult", random.randint(18, 65)
        else:
            category, age = "Child", random.randint(1, 15)
        session.add(
            PackageRequestPassenger(
                age=age,
                category=category,
                name=_rand_name(),
                position=position,
                package_request=request,
            )
        )

    # Notification
    session.add(
        AdminNotification(
            message=f"New package request from {traveller} for "
            f"{request.destinations}",
            is_read=random.choice([True, False]),
            type="PACKAGE_REQUEST",
            package_request=request,
        )
    )

    # Visa application (roughly half the time)
    if random.random() < 0.6:
        _seed_visa_application(session, request, traveller)


def _seed_visa_application(session, request: PackageRequest, traveller: str) -> None:
    given, surname = (traveller.split() + ["Doe"])[:2]
    dob = date.today() - timedelta(days=random.randint(20 * 365, 60 * 365))
    issue = date.today() - timedelta(days=random.randint(100, 2000))
    status = random.choice(_VISA_STATUSES)

    session.add(
        VisaApplication(
            accommodation_address="Bole Road, Addis Ababa",
            arrival_date=request.arrival_date,
            country_of_residence=random.choice(_NATIONALITIES),
            date_of_birth=dob,
            departure_date=request.planned_departure_date,
            email=request.traveller_email,
            emergency_contact_name=_rand_name(),
            emergency_contact_phone=_rand_phone(),
            emergency_contact_relationship=random.choice(
                ["Spouse", "Parent", "Sibling", "Friend"]
            ),
            full_name=traveller,
            gender=random.choice(["Male", "Female"]),
            home_address="123 Main Street",
            nationality=random.choice(_NATIONALITIES),
            occupation=random.choice(
                ["Engineer", "Teacher", "Doctor", "Student", "Manager"]
            ),
            passport_expiry_date=issue + timedelta(days=3650),
            passport_issue_date=issue,
            passport_issuing_country=random.choice(_NATIONALITIES),
            passport_no=request.traveller_passport,
            passport_type="Ordinary",
            phone=request.traveller_whatsapp,
            place_of_birth=random.choice(_DESTINATIONS),
            port_of_entry="Addis Ababa Bole International Airport",
            purpose_of_visit=random.choice(["Tourism", "Business", "Family Visit"]),
            status=status,
            visa_type=random.choice(["Tourist eVisa", "Business eVisa"]),
            package_request=request,
            accommodation_city="Addis Ababa",
            accommodation_name=f"{random.choice(_DESTINATIONS)} Grand Hotel",
            accommodation_street_address="Bole Road",
            accommodation_telephone=_rand_phone(),
            accommodation_type="Hotel",
            address_city="Addis Ababa",
            address_country="Ethiopia",
            agree_terms=True,
            airline=request.airline,
            applicant_category="PRIMARY",
            applicant_label="Primary Applicant",
            applicant_position=1,
            citizenship=random.choice(_NATIONALITIES),
            country_of_birth=random.choice(_NATIONALITIES),
            departure_city="Mumbai",
            departure_country="India",
            flight_number=request.flight_no,
            given_name=given,
            passport_issuing_authority="Ministry of Foreign Affairs",
            passport_number=request.traveller_passport,
            rejection_remark="Insufficient documents"
            if status == "REJECTED"
            else None,
            street_address="123 Main Street",
            surname=surname,
            visa_validity="90 days",
        )
    )


# ============================================================
# 7. CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the SQLite schema and seed synthetic data."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all tables before recreating and seeding.",
    )
    parser.add_argument(
        "--users", type=int, default=5, help="Number of synthetic users."
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=2,
        help="Package requests per user.",
    )
    args = parser.parse_args()

    if args.reset:
        print("Dropping existing tables...")
        drop_all()

    print("Creating tables...")
    create_all()

    with SessionLocal() as session:
        # Guard against non-idempotent re-seeding: if the DB already has
        # users, skip seeding so a plain `python model.py` won't crash on the
        # unique email constraint. Use --reset to force a clean reseed.
        if session.query(User).count() > 0:
            print(
                "Database already contains data; skipping seed. "
                "Use --reset to wipe and reseed."
            )
        else:
            print("Seeding synthetic data...")
            seed(session, num_users=args.users, requests_per_user=args.requests)

    # Report row counts.
    with SessionLocal() as session:
        for model in (
            User,
            PackageRequest,
            PackageRequestCost,
            PackageRequestPassenger,
            AdminNotification,
            VisaApplication,
        ):
            count = session.query(model).count()
            print(f"  {model.__tablename__:32} {count:>5} rows")

    print(f"\nDone. Database at: {DATABASE_URL}")


if __name__ == "__main__":
    main()
