from fastapi import APIRouter

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.get("/now")
def get_users():
    return {"message": "Users endpoint"}
