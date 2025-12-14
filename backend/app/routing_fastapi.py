from fastapi import FastAPI
from api_routes import data_routes

app = FastAPI(title="CSV to JSON API")

# Include routes
app.include_router(data_routes.router, prefix="/api", tags=["Data Endpoints"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the CSV to JSON API! Visit /docs for API documentation."}
