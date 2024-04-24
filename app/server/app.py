from fastapi import FastAPI
from server.routes.corpus import router as CorpusRouter

app = FastAPI()
app.include_router(CorpusRouter, tags=["Corpus"], prefix="/api")


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "This is an API used to detect profanity in russian texts."}
