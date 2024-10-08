import uvicorn
from config import HOST, PORT
import asyncio

if __name__ == "__main__":
    uvicorn.run("server.app:app", host=HOST, port=PORT, reload=True)
