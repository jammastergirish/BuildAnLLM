"use client";

import { useEffect, useRef } from "react";

export default function Heatmap({
  matrix,
  labels,
}: {
  matrix: number[][];
  labels: string[];
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || matrix.length === 0) {
      return;
    }
    const size = matrix.length;
    const cell = 18;
    canvas.width = size * cell;
    canvas.height = size * cell;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const max = Math.max(...matrix.flat());
    const min = Math.min(...matrix.flat());
    const range = max - min || 1;

    for (let i = 0; i < size; i += 1) {
      for (let j = 0; j < size; j += 1) {
        const value = (matrix[i][j] - min) / range;
        const hue = 18;
        const light = 90 - value * 55;
        ctx.fillStyle = `hsl(${hue}deg 70% ${light}%)`;
        ctx.fillRect(j * cell, i * cell, cell, cell);
      }
    }

    ctx.strokeStyle = "rgba(0,0,0,0.15)";
    for (let i = 0; i <= size; i += 1) {
      ctx.beginPath();
      ctx.moveTo(0, i * cell);
      ctx.lineTo(size * cell, i * cell);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(i * cell, 0);
      ctx.lineTo(i * cell, size * cell);
      ctx.stroke();
    }
  }, [matrix]);

  return (
    <div style={{ overflowX: "auto" }}>
      <canvas ref={canvasRef} />
      <div className="inline-row" style={{ marginTop: 12, fontSize: 12 }}>
        {labels.map((label, idx) => (
          <span key={`${label}-${idx}`} className="badge">
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
