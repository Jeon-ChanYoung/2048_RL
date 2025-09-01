from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from game_agent import Game2048Env, RLAgent

app = FastAPI(
    title="2048-RL API", 
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    path="/",
    app=StaticFiles(directory="static", html=True),
    name="static"
)

env = Game2048Env()
agent = RLAgent()

@app.get("/reset")
def reset():
    state = env.reset()
    return {"board" : state.tolist() }