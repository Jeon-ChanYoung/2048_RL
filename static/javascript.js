async function resetGame() {
    stopAutoPlay();
    const res = await fetch("/api/reset");
    const data = await res.json(); 
    renderBoard(data.board);
}

async function stepGame() {
    stopAutoPlay();
    const res = await fetch("/api/step");
    const data = await res.json();
    renderBoard(data.board);
}

let isAutoPlaying = false;
async function autoPlay() {
    if (isAutoPlaying) return;
    isAutoPlaying = true;

    while (isAutoPlaying) {
        const res = await fetch("/api/step")
        const data = await res.json();
        renderBoard(data.board);

        if (data.done) {
            isAutoPlaying = false;
            break
        }

        await new Promise(r => setTimeout(r, 5));
    }
}

function stopAutoPlay() {
    isAutoPlaying = false;
}

// 처음 한 번만 보드 구조 생성
function initBoard() {
    const boardDiv = document.getElementById("board");
    boardDiv.innerHTML = ""; // 초기화
    
    for(let i=0; i<4; i++) {
        for(let j=0; j<4; j++) {
            const cellDiv = document.createElement("div");
            cellDiv.classList.add("tile", "tile-0"); // 초기값 0
            cellDiv.id = `cell-${i}-${j}`;
            boardDiv.appendChild(cellDiv);
        }
    }
    resetGame();
}

// 보드 숫자 갱신
function renderBoard(board) {
    for(let i=0; i<4; i++) {
        for(let j=0; j<4; j++) {
            const cellDiv = document.getElementById(`cell-${i}-${j}`);
            const value = board[i][j];
            cellDiv.textContent = value === 0 ? "" : value;
            
            // 기존 tile-* 클래스 제거
            cellDiv.className = "tile";
            cellDiv.classList.add(`tile-${value}`);
        }
    }
}

document.getElementById("resetBtn").addEventListener("click", resetGame);
document.getElementById("stepBtn").addEventListener("click", stepGame);
document.getElementById("autoBtn").addEventListener("click", autoPlay);
document.getElementById("stopBtn").addEventListener("click", stopAutoPlay);

window.onload = () => {
    initBoard();
};