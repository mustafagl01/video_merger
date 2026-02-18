import argparse
import json
import os
from typing import Any, Dict, Optional

import requests


def api_request(method: str, base_url: str, token: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = base_url.rstrip("/") + path
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = requests.request(method, url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    if response.text.strip():
        return response.json()
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy app to Dokploy via API")
    parser.add_argument("--api-url", default="http://46.224.148.239:3000/")
    parser.add_argument("--token", default=os.getenv("DOKPLOY_API_TOKEN", ""))
    parser.add_argument("--repo", default="https://github.com/KULLANICI/video-merger-api")
    parser.add_argument("--project-name", default="video-merger-api")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY", ""))
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Missing Dokploy API token. Use --token or DOKPLOY_API_TOKEN.")
    print("Creating Dokploy project...")
    project = api_request(
        "POST",
        args.api_url,
        args.token,
        "/api/projects",
        {"name": args.project_name, "description": "Video merger FastAPI service"},
    )
    project_id = project.get("id") or project.get("projectId")
    if not project_id:
        raise RuntimeError(f"Unable to read project id from response: {project}")

    print("Creating Dokploy application...")
    app_payload = {
        "name": args.project_name,
        "projectId": project_id,
        "provider": "github",
        "repository": args.repo,
        "branch": args.branch,
        "buildType": "dockerfile",
        "dockerfilePath": "Dockerfile",
        "port": 8000,
    }
    application = api_request("POST", args.api_url, args.token, "/api/applications", app_payload)
    app_id = application.get("id") or application.get("applicationId")
    if not app_id:
        raise RuntimeError(f"Unable to read application id from response: {application}")

    if args.gemini_key:
        print("Setting environment variables...")
        api_request(
            "POST",
            args.api_url,
            args.token,
            f"/api/applications/{app_id}/env",
            {"variables": [{"key": "GEMINI_API_KEY", "value": args.gemini_key}]},
        )
    else:
        print("Skipping GEMINI_API_KEY env setup (no value provided).")

    print("Triggering deployment...")
    deploy = api_request("POST", args.api_url, args.token, f"/api/applications/{app_id}/deploy", {})

    print(json.dumps({
        "project_id": project_id,
        "application_id": app_id,
        "deploy": deploy,
    }, indent=2))


if __name__ == "__main__":
    main()
