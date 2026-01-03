# Demo deployment (App Engine)

This repo can be deployed as a static frontend served by the FastAPI backend.

## Build the demo frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE_URL="" NEXT_PUBLIC_DEMO_MODE="true" npm run build
```

The build output lands in `frontend/out/`.

## Deploy to App Engine

```bash
gcloud app deploy app.yaml
```

## Notes

- `app.yaml` enables `DEMO_MODE`, which blocks pre-training, fine-tuning, and inference endpoints.
- The UI disables those controls when hosted on `demo.buildanllm.com` or when `NEXT_PUBLIC_DEMO_MODE=true`.
- If you use a different demo hostname, update `frontend/lib/demo.ts` or set `NEXT_PUBLIC_DEMO_MODE=true` at build time.
