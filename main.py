from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from game_agent import Game2048Env, RLAgent
import uvicorn
import os

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

env = Game2048Env()
agent = RLAgent()

@app.get("/api/reset")
def reset():
    state = env.reset()
    return {"board" : state.tolist() }

@app.get("/api/step")
def step():
    action = agent.act(env)
    if action is None:
        return {
            "board" : env.state.tolist(),
            "reward": 0,
            "done": True,
            "action": None
        }
    
    next_state, reward, done = env.step(action)
    return {
        "board": next_state.tolist(),
        "reward": reward,
        "done": done,
        "action": action
    }

app.mount(
    path="/static",
    app=StaticFiles(directory="static", html=True),
    name="static"
)

@app.get("/")
def root():
    return FileResponse(os.path.join("static", "index.html"))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)