import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.backend import data, pages

DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

api = APIRouter(prefix="/api")


def _json(payload):
    return JSONResponse(jsonable_encoder(payload))


@api.get("/runs")
def runs():
    return _json({"runs": pages.history()})


@api.get("/runs/latest")
def latest_run():
    return _json(pages.resolve_run(None))


@api.get("/overview")
def overview(run_id: str | None = None):
    return _json(pages.overview(run_id))


@api.get("/churn")
def churn(run_id: str | None = None):
    return _json(pages.churn(run_id))


@api.get("/clv")
def clv(run_id: str | None = None):
    return _json(pages.clv(run_id))


@api.get("/demand")
def demand(run_id: str | None = None):
    return _json(pages.demand(run_id))


def _warm_cache():
    for page in (pages.overview, pages.churn, pages.clv, pages.demand):
        try:
            page(None)
        except Exception:  # an empty or unreachable dataset shows up on the requests themselves
            pass


@asynccontextmanager
async def lifespan(_):
    """Fill the cache in the background so the first visitor doesn't wait for BigQuery."""
    if not data.fixture_mode():
        threading.Thread(target=_warm_cache, daemon=True).start()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="vertex-demo dashboard", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.exception_handler(data.DatasetError)
    def dataset_error(_, exc: data.DatasetError):
        return JSONResponse({"error": str(exc)}, status_code=500)

    @app.exception_handler(pages.NoRun)
    def no_run(_, exc: pages.NoRun):
        return JSONResponse(jsonable_encoder(pages.empty()))

    @app.exception_handler(HTTPException)
    def http_error(_, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.get("/healthz")
    @app.get("/health")  # run.app reserves some paths ending in z, so the deployed probe uses /health
    def healthz():
        return {"status": "ok"}

    app.include_router(api)

    @app.api_route("/api/{rest:path}", methods=["GET"], include_in_schema=False)
    def api_not_found(rest: str):
        raise HTTPException(404, "not found")

    if (DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = (DIST / path).resolve()
        if path and DIST in target.parents and target.is_file():
            return FileResponse(target)
        index = DIST / "index.html"
        if not index.is_file():
            raise HTTPException(404, "UI not built")
        return FileResponse(index)

    return app


app = create_app()
