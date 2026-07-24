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
    "prepping": "purple",
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
    "Decision": {
        "select": {
            "options": [
                {"name": "Apply", "color": "green"},
                {"name": "No", "color": "red"},
            ]
        }
    },
    "Priority": {"number": {"format": "number"}},
    "Tier": {
        "select": {
            "options": [
                {"name": "T1", "color": "green"},
                {"name": "T2", "color": "blue"},
                {"name": "T3", "color": "yellow"},
                {"name": "T4", "color": "orange"},
                {"name": "none", "color": "gray"},
            ]
        }
    },
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
    # Reserved for operations with no Status equivalent — an LLM call or content
    # generation, not a state transition. Status changes (including applied,
    # rejected, withdrawn, not_interested) are picked up directly by watch()
    # polling the Status field; see NotionSync.list_active_statuses.
    "Action": {
        "select": {
            "options": [
                {"name": "Rescore", "color": "blue"},
                {"name": "Prep", "color": "purple"},
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


def _callout_text(score: Score) -> str:
    head = "APPLY" if score.decision == "apply" else "NO"
    tier = f" {score.tier_label}" if score.tier_label not in ("", "none") else ""
    return f"{head}{tier} — {score.rationale}"


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
            return "Role" in props and "Decision" in props
        except Exception:
            return False

    # --- page operations ---

    @staticmethod
    def score_props(score: Score) -> dict[str, Any]:
        """Score-derived page properties: Decision / Priority / Tier / Flags."""
        props: dict[str, Any] = {}
        if score.decision:
            props["Decision"] = {"select": {"name": "Apply" if score.decision == "apply" else "No"}}
        if score.priority is not None:
            props["Priority"] = {"number": score.priority}
        if score.tier_label:
            props["Tier"] = {"select": {"name": score.tier_label}}
        if score.flags:
            props["Flags"] = {"rich_text": _rt(", ".join(score.flags))}
        return props

    def update_score(self, page_id: str, score: Score) -> None:
        """Refresh one page's score properties and its 🤖 callout."""
        self._patch(f"/pages/{page_id}", {"properties": self.score_props(score)})
        self.update_score_callout(page_id, score)

    def push_listing(self, listing: Listing, score: Score, status: str = "shortlisted") -> str:
        props: dict[str, Any] = {
            "Role": {"title": _rt(listing.title)},
            "Company": {"rich_text": _rt(listing.company)},
            "URL": {"url": listing.url},
            "Status": {"select": {"name": status}},
            "Source": {"rich_text": _rt(listing.source_name)},
            "DB ID": {"number": listing.id},
            **self.score_props(score),
        }
        if listing.location:
            props["Location"] = {"rich_text": _rt(listing.location)}

        children = []
        if score.rationale:
            children.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "icon": {"type": "emoji", "emoji": "🤖"},
                    "rich_text": _rt(_callout_text(score), 2000),
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
                notes_rt = props.get("Notes", {}).get("rich_text", [])
                notes = notes_rt[0]["text"]["content"] if notes_rt else ""
                pending.append({
                    "page_id": page["id"],
                    "listing_id": int(db_id),
                    "action": action_sel["name"],
                    "notes": notes,
                })
        return pending

    def list_active_statuses(self) -> list[dict]:
        """(page_id, listing_id, status, notes) for every non-archived page.

        Archived pages are excluded by the query API automatically, so this
        naturally stops reporting a page once it's been archived (e.g. by the
        sync below). Used to detect a Status edit made by hand on the board —
        Action is reserved for operations that have no status equivalent.
        """
        results = []
        cursor = None
        while True:
            body: dict = {"page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            data = self._post(f"/databases/{self.database_id}/query", body)
            for page in data.get("results", []):
                props = page["properties"]
                db_id = props.get("DB ID", {}).get("number")
                status_sel = props.get("Status", {}).get("select")
                if db_id is None or not status_sel:
                    continue
                notes_rt = props.get("Notes", {}).get("rich_text", [])
                results.append({
                    "page_id": page["id"],
                    "listing_id": int(db_id),
                    "status": status_sel["name"],
                    "notes": notes_rt[0]["text"]["content"] if notes_rt else "",
                })
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
        return results

    def update_score_callout(self, page_id: str, score: "Score") -> None:
        """Update the 🤖 callout block on an existing page with the latest score."""
        children = self._get(f"/blocks/{page_id}/children").get("results", [])
        callout_id = next(
            (b["id"] for b in children if b.get("type") == "callout"),
            None,
        )
        text = _callout_text(score)
        if callout_id:
            self._patch(f"/blocks/{callout_id}", {
                "callout": {
                    "rich_text": _rt(text, 2000),
                    "icon": {"type": "emoji", "emoji": "🤖"},
                },
            })
        else:
            # No callout yet — append one
            self._patch(f"/blocks/{page_id}/children", {
                "children": [{
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "icon": {"type": "emoji", "emoji": "🤖"},
                        "rich_text": _rt(text, 2000),
                    },
                }]
            })

    def append_prep_content(self, page_id: str, content: str) -> None:
        blocks: list[dict] = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": _rt("Application Prep")},
            }
        ]
        for line in content.splitlines():
            if line.startswith("## "):
                blocks.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {"rich_text": _rt(line[3:].strip())},
                })
            elif line.startswith("- ") or line.startswith("* "):
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": _rt(line[2:].strip())},
                })
            elif line.strip():
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": _rt(line.strip())},
                })
        # Notion allows max 100 children per request — batch if needed
        for i in range(0, len(blocks), 100):
            self._patch(f"/blocks/{page_id}/children", {"children": blocks[i:i + 100]})
