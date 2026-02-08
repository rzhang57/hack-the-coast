# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ADHD productivity assistant: when a user says they're stuck, the app analyzes their recent screen activity (rolling 2-minute screenshot buffer) via Google Gemini and provides AI-powered suggestions to get unblocked.

Monorepo with two components:
- **`server/`** — Python/Flask backend: rolling screenshot buffer + Gemini AI chat
- **`client/hack-the-coast/`** — React/TypeScript frontend (Vite, React 19)

## Running the Server

```bash
cd server
source venv/bin/activate
flask --app app run          # or: python -m flask --app app run
```

Requires `server/.env` with `CHAT_API_KEY` (Google Gemini API key).

## Running the Client

```bash
cd client/hack-the-coast
npm install
npm run dev      # Vite dev server
npm run build    # tsc + vite build
npm run lint     # ESLint
```

## Architecture

### Server (`server/`)

- **`app.py`** — Flask app entry point. Initializes the Gemini client (`google.genai`), creates routes, wires up services.
- **`services/buffer_service.py`** — `BufferService` class: manages the rolling screenshot buffer. Intended to run a background thread capturing screenshots every 5s using `mss`, storing base64 images + metadata (timestamp, active window) in a circular buffer (24 slots = 2 min). Exposes `flush_buffer()` to drain and return all buffered data.
- **`services/chat_service.py`** — Chat service for Gemini API interaction (stub).

### Key Endpoints

- `GET /buffer/status` — Returns buffer state
- `POST /assist/chat/init/` — Flushes buffer and sends screenshots to Gemini for analysis

### Data Flow

1. `BufferService` continuously captures screenshots in a background thread
2. User clicks "I'm stuck" in the frontend
3. Frontend calls `POST /assist/chat/init/`
4. Server flushes the buffer (retrieves + clears all screenshots)
5. Screenshots sent to Gemini API as multimodal input
6. AI response returned to frontend

## Key Dependencies

- **Server**: Flask, `google-genai`, `python-dotenv`, `mss` (for screenshots)
- **Client**: React 19, Vite 7, TypeScript 5.9

## Important Notes

- The server uses a virtualenv at `server/venv/` — always activate before running
- No `requirements.txt` exists yet; packages are installed directly in the venv
- The codebase is early-stage — `BufferService` and `ChatService` are stubs being built out
- Gemini model target: `gemini-1.5-flash` for multimodal screenshot analysis
- Do not include unnecessary comments
