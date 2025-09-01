async function resetGame() {
    const res = await fetch("/api/reset");
    const data = await res.json();   // 서버에서 JSON 반환한다고 가정
    renderBoard(data.board);
    stopAutoPlay();
}

async function stepGame() {
    const res = await fetch("/api/step");
    const data = await res.json();
    renderBoard(data.board);
    stopAutoPlay();
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

        await new Promise(r => setTimeout(r, 10));
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

// 페이지 로드 시 초기화
window.onload = () => {
    initBoard();
};