from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import httpx

from .models import Application, Listing, Score

_API = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"

_STATUS_COLORS = {
    "shortlisted": "blue",
    "applied": "yellow",
    "interviewing": "orange",
    "offer": "green",
    "rejected": "red",
    "withdrawn": "gray",
    "not_interested": "default",
}

_DB_PROPERTIES = {
    "Role": {"title": {}},
    "Company": {"rich_text": {}},
    "URL": {"url": {}},
    "Fit": {"number": {"format": "number"}},
    "Location": {"rich_text": {}},
    "Status": {
        "select": {
            "options": [{"name": k, "color": v} for k, v in _STATUS_COLORS.items()]
        }
    },
    "Applied": {"date": {}},
    "Chase": {"date": {}},
    "Source": {"rich_text": {}},
    "Flags": {"rich_text": {}},
    "Notes": {"rich_text": {}},
    "Contacts": {"rich_text": {}},
    "Action": {
        "select": {
            "options": [
                {"name": "Rescore", "color": "blue"},
                {"name": "Prep", "color": "purple"},
                {"name": "Apply", "color": "green"},
                {"name": "Not Interested", "color": "red"},
            ]
        }
    },
    "DB ID": {"number": {"format": "number"}},
}


def _extract_page_id(url_or_id: str) -> str:
    clean = url_or_id.strip().replace("-", "")
    m = re.search(r"([a-f0-9]{32})", clean, re.I)
    if m:
        h = m.group(1).lower()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
    return url_or_id


def _rt(text: str, limit: int = 2000) -> list[dict]:
    return [{"text": {"content": text[:limit]}}]


class NotionSync:
    def __init__(self, token: str, database_id: str = "") -> None:
        self._token = token
        self.database_id = database_id
        self._client = httpx.Client(
            base_url=_API,
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": _NOTION_VERSION,
                "Content-Type": "application/json",
            },
            timeout=30,
        )

    def _post(self, path: str, body: dict) -> dict:
        r = self._client.post(path, json=body)
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, body: dict) -> dict:
        r = self._client.patch(path, json=body)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict:
        r = self._client.get(path)
        r.raise_for_status()
        return r.json()

    # --- database setup ---

    def create_database(self, parent_page_url_or_id: str) -> str:
        parent_id = _extract_page_id(parent_page_url_or_id)
        data = self._post("/databases", {
            "parent": {"type": "page_id", "page_id": parent_id},
            "title": [{"type": "text", "text": {"content": "Job Pipeline"}}],
            "properties": _DB_PROPERTIES,
        })
        self.database_id = data["id"]
        return self.database_id

    def verify_database(self) -> bool:
        try:
            data = self._get(f"/databases/{self.database_id}")
            props = data.get("properties") or {}
            return "Role" in props and "Fit" in props
        except Exception:
            return False

    # --- page operations ---

    def push_listing(self, listing: Listing, score: Score, status: str = "shortlisted") -> str:
        props: dict[str, Any] = {
            "Role": {"title": _rt(listing.title)},
            "Company": {"rich_text": _rt(listing.company)},
            "URL": {"url": listing.url},
            "Fit": {"number": score.fit_score},
            "Status": {"select": {"name": status}},
            "Source": {"rich_text": _rt(listing.source_name)},
            "DB ID": {"number": listing.id},
        }
        if listing.location:
            props["Location"] = {"rich_text": _rt(listing.location)}
        if score.flags:
            props["Flags"] = {"rich_text": _rt(", ".join(score.flags))}

        children = []
        if score.rationale:
            children.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "icon": {"type": "emoji", "emoji": "🤖"},
                    "rich_text": _rt(f"Fit {score.fit_score}/100 — {score.rationale}", 2000),
                },
            })

        page = self._post("/pages", {
            "parent": {"database_id": self.database_id},
            "properties": props,
            "children": children,
        })
        return page["id"]

    def update_status(
        self,
        page_id: str,
        status: str,
        applied_at: datetime | None = None,
        chase_at: datetime | None = None,
        notes: str = "",
        contacts: str = "",
    ) -> None:
        props: dict[str, Any] = {"Status": {"select": {"name": status}}}
        if applied_at:
            props["Applied"] = {"date": {"start": applied_at.date().isoformat()}}
        if chase_at:
            props["Chase"] = {"date": {"start": chase_at.date().isoformat()}}
        if notes:
            props["Notes"] = {"rich_text": _rt(notes)}
        if contacts:
            props["Contacts"] = {"rich_text": _rt(contacts)}
        self._patch(f"/pages/{page_id}", {"properties": props})

    def archive_page(self, page_id: str) -> None:
        self._patch(f"/pages/{page_id}", {"archived": True})

    def update_schema(self) -> None:
        """Patch the existing database to add any missing properties/options."""
        self._patch(f"/databases/{self.database_id}", {"properties": _DB_PROPERTIES})

    def reset_action(self, page_id: str) -> None:
        self._patch(f"/pages/{page_id}", {"properties": {"Action": {"select": None}}})

    def get_pending_actions(self) -> list[dict]:
        data = self._post(f"/databases/{self.database_id}/query", {
            "filter": {"property": "Action", "select": {"is_not_empty": True}},
        })
        pending = []
        for page in data.get("results", []):
            props = page["properties"]
            action_sel = props.get("Action", {}).get("select")
            db_id = props.get("DB ID", {}).get("number")
            if action_sel and db_id is not None:
                pending.append({
                    "page_id": page["id"],
                    "listing_id": int(db_id),
                    "action": action_sel["name"],
                })
        return pending

    def append_prep_content(self, page_id: str, content: str) -> None:
        self._patch(f"/blocks/{page_id}/children", {
            "children": [
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": _rt("Application Prep")},
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": _rt(content[:2000])},
                },
            ]
        })
