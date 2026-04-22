import base64
import hashlib
import hmac
import json
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

try:
    from .rag_engine import ask_nlp
except ImportError:
    from rag_engine import ask_nlp


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

HARDCODED_USERS = {
    "admin@msrit.edu": {
        "password": "admin123",
        "name": "MSRIT Admin",
        "role": "admin",
    },
    "student@msrit.edu": {
        "password": "student123",
        "name": "MSRIT Student",
        "role": "student",
    },
}

PAGE_ACCESS = {
    "student": {"student", "admin"},
    "staff": {"staff", "admin"},
}

AUTH_COOKIE_NAME = "msrit_session"
AUTH_SECRET = "msrit-cookie-signing-secret-change-in-production"
SESSION_ID_COOKIE_NAME = "msrit_sid"

CHAT_CACHE_TTL_SECONDS = 600
CHAT_CACHE_MAX_SIZE = 300
CHAT_HISTORY_MAX_ITEMS = 40

CHAT_CACHE: OrderedDict[str, tuple[float, str]] = OrderedDict()
SESSION_CONVERSATIONS: dict[str, list[dict[str, str]]] = {}


class LoginRequest(BaseModel):
    email: str
    password: str


class RagRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    query: str


app = FastAPI(title="Curriculum Gap Detection API")


def _frontend_file(filename: str) -> Path:
    return FRONTEND_DIR / filename


def _current_user(request: Request) -> dict[str, Any] | None:
    token = request.cookies.get(AUTH_COOKIE_NAME)
    if not token:
        return None

    try:
        payload_b64, signature = token.split(".", maxsplit=1)
    except ValueError:
        return None

    expected_signature = hmac.new(
        AUTH_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature, expected_signature):
        return None

    try:
        payload_bytes = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        user = json.loads(payload_bytes.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None

    if isinstance(user, dict):
        return user

    return None


def _issue_auth_cookie(response: Response, user: dict[str, Any]) -> None:
    payload_json = json.dumps(user, separators=(",", ":"), sort_keys=True)
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8")
    signature = hmac.new(
        AUTH_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    token = f"{payload_b64}.{signature}"

    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


def _get_session_id(request: Request) -> str | None:
    sid = request.cookies.get(SESSION_ID_COOKIE_NAME)
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    return None


def _issue_session_id_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_ID_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


def _ensure_session_id(request: Request, response: Response) -> str:
    session_id = _get_session_id(request)
    if session_id:
        if session_id not in SESSION_CONVERSATIONS:
            SESSION_CONVERSATIONS[session_id] = []
        return session_id

    session_id = uuid.uuid4().hex
    SESSION_CONVERSATIONS[session_id] = []
    _issue_session_id_cookie(response, session_id)
    return session_id


def _is_authorized_for_page(user: dict[str, Any], page: str) -> bool:
    role = str(user.get("role", "")).lower()
    allowed_roles = PAGE_ACCESS.get(page, set())
    return role in allowed_roles


def _cache_get(key: str) -> str | None:
    entry = CHAT_CACHE.get(key)
    if not entry:
        return None

    created_at, cached_value = entry
    if time.time() - created_at > CHAT_CACHE_TTL_SECONDS:
        CHAT_CACHE.pop(key, None)
        return None

    CHAT_CACHE.move_to_end(key)
    return cached_value


def _cache_set(key: str, value: str) -> None:
    CHAT_CACHE[key] = (time.time(), value)
    CHAT_CACHE.move_to_end(key)
    while len(CHAT_CACHE) > CHAT_CACHE_MAX_SIZE:
        CHAT_CACHE.popitem(last=False)


def _append_history(session_id: str, sender: str, text: str) -> None:
    history = SESSION_CONVERSATIONS.setdefault(session_id, [])
    history.append({"sender": sender, "text": text})
    if len(history) > CHAT_HISTORY_MAX_ITEMS:
        del history[: len(history) - CHAT_HISTORY_MAX_ITEMS]


def _to_points(raw_text: str, max_points: int = 4) -> list[str]:
    normalized = raw_text.replace("\n", " ").strip()
    if not normalized:
        return ["No relevant data found in the indexed knowledge base."]

    chunks = [part.strip(" -") for part in normalized.split(".") if part.strip()]
    if not chunks:
        return [normalized]

    return chunks[:max_points]

def _is_simple_query(query: str) -> bool:
    q = query.lower().strip()
    return q in ["hi", "hello", "hey"] or len(q.split()) <= 2


def _build_role_prompt(role: str, query: str) -> str:
    cleaned_query = query.strip()
    if role == "staff" or role == "admin":
        return (
            "Staff curriculum analysis request. Focus on curriculum gaps, syllabus updates, "
            "course outcomes, and implementation priorities. Question: "
            f"{cleaned_query}"
        )

    return (
        "Student guidance request. Focus on skill roadmap, placement readiness, prerequisite "
        "topics, and next learning actions. Question: "
        f"{cleaned_query}"
    )


def _format_chat_response(role: str, query: str, rag_answer: str) -> str:
    insight_points = _to_points(rag_answer)
    insight_block = "\n".join(f"- {point}." for point in insight_points)

    if role == "staff" or role == "admin":
        return (
            "Staff-Centric Curriculum Brief\n"
            f"Question: {query}\n\n"
            "1. Gap Summary\n"
            f"{insight_block}\n\n"
            "2. Recommended Curriculum Actions\n"
            "- Add or strengthen modules where gaps are visible in industry relevance.\n"
            "- Rebalance theory/practice with lab or project-based activities.\n"
            "- Update course outcomes and assessment rubrics to track skill attainment.\n\n"
            "3. Department Follow-up\n"
            "- Validate with recent placement trends and recruiter feedback.\n"
            "- Prioritize updates for upcoming semester revision cycles."
        )

    return (
        "Student Learning Guidance\n"
        f"Question: {query}\n\n"
        "1. Personalized Insight\n"
        f"{insight_block}\n\n"
        "2. What To Learn Next\n"
        "- Focus on the skills highlighted above and revise prerequisites first.\n"
        "- Build one mini-project that proves each major skill area.\n"
        "- Practice interview-style questions based on those skills.\n\n"
        "3. Short-Term Plan (2-4 weeks)\n"
        "- Week 1: Foundation revision and notes.\n"
        "- Week 2: Hands-on implementation.\n"
        "- Week 3-4: Project polishing and mock interview practice."
    )


@app.post("/api/auth/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
    email = payload.email.strip().lower()
    password = payload.password

    account = HARDCODED_USERS.get(email)

    if not account or password != account["password"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    user = {
        "email": email,
        "name": account["name"],
        "role": account["role"],
    }
    _issue_auth_cookie(response, user)
    _ensure_session_id(request, response)

    return {"authenticated": True, "user": user}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    session_id = _get_session_id(request)
    if session_id:
        SESSION_CONVERSATIONS.pop(session_id, None)

    response.delete_cookie(key=AUTH_COOKIE_NAME, path="/")
    response.delete_cookie(key=SESSION_ID_COOKIE_NAME, path="/")
    return {"authenticated": False}


@app.get("/api/auth/me")
def auth_me(request: Request) -> dict[str, Any]:
    user = _current_user(request)
    return {"authenticated": bool(user), "user": user}


@app.get("/api/auth/authorize")
def authorize(request: Request, page: str) -> dict[str, Any]:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    page_key = page.strip().lower()
    if page_key not in PAGE_ACCESS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown page")

    if not _is_authorized_for_page(user, page_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    return {"authorized": True, "user": user, "page": page_key}


@app.post("/api/rag/query")
def rag_query(payload: RagRequest, request: Request) -> dict[str, str]:
    user = _current_user(request)
    user_type = str(user.get("role", "student")).lower() if user else "student"
    return {"response": ask_nlp(payload.query, user_type=user_type)}


@app.post("/api/chat/query")
def chat_query(payload: ChatRequest, request: Request, response: Response) -> dict[str, Any]:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty")

    session_id = _ensure_session_id(request, response)
    role = str(user.get("role", "student")).lower()

    cache_key = f"{role}:{query.lower()}"
    response_text = _cache_get(cache_key)
    cache_hit = bool(response_text)

    if not response_text:
        if _is_simple_query(query):
            response_text = "Hello! How can I help you with curriculum insights today?"
        else:
            rag_answer = ask_nlp(query, user_type=role)
            response_text = rag_answer.strip()

        _cache_set(cache_key, response_text)

    _append_history(session_id, "user", query)
    _append_history(session_id, "assistant", response_text)

    return {"response": response_text, "cached": cache_hit}


@app.get("/api/chat/history")
def chat_history(request: Request) -> dict[str, Any]:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    session_id = _get_session_id(request)
    if not session_id:
        return {"messages": []}

    return {"messages": SESSION_CONVERSATIONS.get(session_id, [])}


def _serve_public_page(filename: str) -> FileResponse:
    return FileResponse(_frontend_file(filename))


@app.get("/")
def root() -> FileResponse:
    return _serve_public_page("Home.html")


@app.get("/Home.html")
def home_page() -> FileResponse:
    return _serve_public_page("Home.html")


@app.get("/Login.html")
def student_login_page(request: Request):
    user = _current_user(request)
    if user and _is_authorized_for_page(user, "student"):
        return RedirectResponse(url="/Student.html", status_code=status.HTTP_302_FOUND)
    return _serve_public_page("Login.html")


@app.get("/StaffLogin.html")
def staff_login_page(request: Request):
    user = _current_user(request)
    if user and _is_authorized_for_page(user, "staff"):
        return RedirectResponse(url="/Staff.html", status_code=status.HTTP_302_FOUND)
    return _serve_public_page("StaffLogin.html")


@app.get("/Student.html")
def student_page(request: Request):
    user = _current_user(request)
    if not user:
        return RedirectResponse(url="/Login.html", status_code=status.HTTP_302_FOUND)
    if not _is_authorized_for_page(user, "student"):
        return RedirectResponse(url="/Staff.html", status_code=status.HTTP_302_FOUND)
    return _serve_public_page("Student.html")


@app.get("/Staff.html")
def staff_page(request: Request):
    user = _current_user(request)
    if not user:
        return RedirectResponse(url="/StaffLogin.html", status_code=status.HTTP_302_FOUND)
    if not _is_authorized_for_page(user, "staff"):
        return RedirectResponse(url="/Student.html", status_code=status.HTTP_302_FOUND)
    return _serve_public_page("Staff.html")
