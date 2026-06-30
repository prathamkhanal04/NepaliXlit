from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from time import time
from typing import List
import asyncio
import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from transliteration import XlitEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Tunables -------------------------------------------------------------
# XlitEngine is not guaranteed thread-safe internally for concurrent calls
# on a single instance, so we keep a small pool of engine instances and
# dispatch work across them from a ThreadPoolExecutor.
#
# ENGINE_POOL_SIZE should match physical CPU cores, not request concurrency.
# This is CPU-bound inference work, so on a 2-core box, 2 engine instances is
# the right number -- more threads than cores just adds context-switch
# overhead and doubles memory use (each instance loads its own model
# weights) without doing any more work per second. If you move to a bigger
# box, raise this to match core count.
ENGINE_POOL_SIZE = int(os.environ.get("ENGINE_POOL_SIZE", "2"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "100000"))
BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", "10"))


class EnginePool:
    """A small round-robin pool of XlitEngine instances, each pinned to its
    own worker thread, so concurrent requests don't serialize on a single
    engine object or fight over the GIL more than necessary."""

    def __init__(self, size: int, beam_width: int):
        self.size = size
        self._engines = [XlitEngine("ne", beam_width=beam_width) for _ in range(size)]
        # One worker thread per engine instance avoids two requests ever
        # calling into the same engine object concurrently.
        self._executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix="xlit")
        self._counter = 0
        self._lock = asyncio.Lock()

    async def _next_engine_index(self) -> int:
        async with self._lock:
            idx = self._counter % self.size
            self._counter += 1
            return idx

    def _translit_one(self, engine_idx: int, text: str) -> str:
        return self._engines[engine_idx].translit_sentence(text)

    async def translit(self, text: str) -> str:
        idx = await self._next_engine_index()
        loop = asyncio.get_running_loop()
        fn = partial(self._translit_one, idx, text)
        return await loop.run_in_executor(self._executor, fn)

    async def translit_many(self, texts: List[str]) -> List[str | None]:
        """Fan a batch out across the pool concurrently instead of awaiting
        each item one at a time. Individual failures don't kill the whole
        batch -- a failed item comes back as None and is reported separately,
        which matters once you're at 10k-100k lines and don't want one bad
        line to discard everything else.
        """
        results = await asyncio.gather(
            *(self.translit(t) for t in texts), return_exceptions=True
        )
        return results

    def shutdown(self):
        self._executor.shutdown(wait=True)


engine_pool: EnginePool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_pool
    logger.info("Loading %d XlitEngine instance(s)...", ENGINE_POOL_SIZE)
    engine_pool = EnginePool(ENGINE_POOL_SIZE, BEAM_WIDTH)
    logger.info("Engine pool ready.")
    yield
    logger.info("Shutting down engine pool...")
    engine_pool.shutdown()


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=BASE_DIR)


class Query(BaseModel):
    text: str = Field(..., min_length=1)


class BatchQuery(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/transliterate")
async def transliterate(query: Query):
    try:
        start_time = time()
        out = await engine_pool.translit(query.text)
        end_time = time()
        return {
            "original": query.text,
            "translit": out,
            "time_taken": end_time - start_time,
        }
    except Exception as exp:
        logger.exception("Transliteration failed")
        raise HTTPException(status_code=500, detail=str(exp))


@app.post("/transliterate/batch")
async def transliterate_batch(query: BatchQuery):
    """Batch endpoint for large corpus processing.

    Texts are processed concurrently across the engine pool rather than
    sequentially, while never blocking the event loop (each call runs in a
    worker thread via run_in_executor). With a 2-worker engine pool this
    won't make individual items faster, but it does mean:
      - the server can still accept other requests while a big batch runs
      - both CPU cores are kept busy instead of one
      - a single bad input doesn't abort the rest of a 100k-line job
    """
    try:
        start_time = time()
        raw_results = await engine_pool.translit_many(query.texts)
        end_time = time()

        results = []
        errors = []
        for i, (original, res) in enumerate(zip(query.texts, raw_results)):
            if isinstance(res, Exception):
                errors.append({"index": i, "original": original, "error": str(res)})
                results.append({"original": original, "translit": None})
            else:
                results.append({"original": original, "translit": res})

        return {
            "count": len(query.texts),
            "success_count": len(query.texts) - len(errors),
            "error_count": len(errors),
            "results": results,
            "errors": errors,
            "time_taken": end_time - start_time,
        }
    except Exception as exp:
        logger.exception("Batch transliteration failed")
        raise HTTPException(status_code=500, detail=str(exp))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn app:app --reload
