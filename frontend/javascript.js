async function newSession() {
    const r = await fetch(`${API}/api/session`, { method: "POST"})
    const data = await r.json();
    alert(data);
}

const API = "http://localhost:8000"

document.getElementById("new").onclick = newSession;