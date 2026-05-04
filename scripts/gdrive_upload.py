#!/usr/bin/env python3
"""Upload a file to Google Drive.

First-time setup:
  1. Create OAuth client (Desktop app) at https://console.cloud.google.com/
     and download credentials.json
  2. Place it at $XDG_CONFIG_HOME/gdrive_upload/credentials.json
     (or ~/.config/gdrive_upload/credentials.json)
  3. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

Usage:
  python gdrive_upload.py <file> [--folder-id ID] [--name NAME]
"""
import argparse
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CONFIG_DIR = Path.home() / ".config" / "gdrive_upload"
TOKEN_PATH = CONFIG_DIR / "token.json"
CREDENTIALS_PATH = CONFIG_DIR / "credentials.json"


def get_credentials():
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not CREDENTIALS_PATH.exists():
            sys.exit(f"missing OAuth client at {CREDENTIALS_PATH}")
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_PATH), SCOPES
        )
        creds = flow.run_local_server(port=0)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    return creds


def upload(file_path: str, folder_id: str | None, name: str | None):
    creds = get_credentials()
    service = build("drive", "v3", credentials=creds)
    metadata = {"name": name or Path(file_path).name}
    if folder_id:
        metadata["parents"] = [folder_id]
    media = MediaFileUpload(file_path, resumable=True)
    return (
        service.files()
        .create(body=metadata, media_body=media, fields="id, webViewLink")
        .execute()
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("file", help="path to file to upload")
    p.add_argument("--folder-id", help="parent folder ID in Drive")
    p.add_argument("--name", help="override file name on Drive")
    args = p.parse_args()

    if not Path(args.file).is_file():
        sys.exit(f"not a file: {args.file}")

    f = upload(args.file, args.folder_id, args.name)
    print(f"id:   {f['id']}")
    print(f"link: {f['webViewLink']}")


if __name__ == "__main__":
    main()
