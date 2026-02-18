# Video Merger API

FastAPI service to download videos from a public Google Drive folder, analyze and reorder clips with Gemini 2.0 Flash, trim/merge with FFmpeg, optionally add background music, and return a download URL.

## API

### POST /merge
Body:

```json
{
  "drive_link": "https://drive.google.com/drive/folders/...",
  "music_url": "https://example.com/music.mp3"
}
```

Response:

```json
{
  "download_url": "https://your-domain/download/<id>"
}
```

Then download the final mp4 from the `download_url`.

## Local run

1. `python -m venv .venv`
2. `.venv\\Scripts\\activate`
3. `pip install -r requirements.txt`
4. `copy .env.example .env` and set `GEMINI_API_KEY`
5. `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Docker (Dokploy)

Build and run:

1. `docker build -t video-merger-api .`
2. `docker run -p 8000:8000 --env GEMINI_API_KEY=your_key video-merger-api`

## Dokploy Auto Deploy Script

Script: `scripts/deploy_dokploy.py`

Example:

`python scripts/deploy_dokploy.py --token <DOKPLOY_TOKEN> --gemini-key <GEMINI_API_KEY>`

The script attempts to:
1. Create a project
2. Create an app from `https://github.com/KULLANICI/video-merger-api`
3. Set `GEMINI_API_KEY`
4. Trigger deployment

If your Dokploy instance has different API routes, adjust paths in `scripts/deploy_dokploy.py`.
