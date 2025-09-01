async function resetGame() {
    const res = await fetch("/reset");
    const data = await res.json();   // 서버에서 JSON 반환한다고 가정
    console.log(data);
}