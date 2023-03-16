import uvicorn
from fastapi import FastAPI
from app.api.api_v1.router import router
from app.core.scheduler import dailyEncode

app = FastAPI()
app.include_router(router)

# Default root path
@app.get('/')
async def root():

    message = {
        'message': 'This is face recognition POLDA API v1.0'
    }

    return message

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3344)