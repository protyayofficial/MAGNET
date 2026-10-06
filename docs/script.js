const matrix = document.querySelector('.matrix');
if (matrix) {
  for (let row = 0; row < 11; row += 1) {
    for (let col = 0; col < 11; col += 1) {
      const cell = document.createElement('span');
      const distance = Math.abs(row - col);
      const signal = Math.sin((row + 1) * (col + 1) * 1.7) * 0.5 + 0.5;
      cell.style.setProperty('--strength', (distance === 0 ? 1 : Math.max(0.12, 0.78 - distance * 0.065 + signal * 0.16)).toFixed(2));
      cell.style.setProperty('--delay', `${(row + col) * 28}ms`);
      matrix.append(cell);
    }
  }
}
