from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

app = FastAPI(title="2048-RL API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,             # CORS 문제 해결을 위한 미들웨어 추가
    allow_origins=["*"],        # 어떤 출처에서 오는 요청을 허용할지 ("*"는 모두 허용)
    allow_credentials=True,     # 쿠키나 인증정보(Authorization 헤더) 허용 여부
    allow_methods=["*"],        # 허용할 HTTP 메서드 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],        # 허용할 요청 헤더 (예: Authorization, Content-Type 등)
)

class SessionResp(BaseModel):
    session_id: str
    action: int | None = None

@app.post("/api/session", response_model=SessionResp)
def create_session():
    sid = str(uuid.uuid4())
    # env = Game2048()