from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
def get_status():
    """Returns the API status."""
    return {"status": "ok", "message": "API is running."}
