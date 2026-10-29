import { memo, useLayoutEffect, useRef, useState, type PointerEvent as RPE } from "react";

export interface Mark {
  col: number;
  row: number;
}

interface Props {
  cols: number;
  rows: number;
  // One colour index per cell, row-major, into `css`.
  values: ArrayLike<number>;
  css: readonly string[];
  cellH: number;
  gap?: number;
  label: string;
  marks?: readonly Mark[];
  // Empty columns on the left (context shorter than the window) are drawn as faint placeholders.
  firstCol?: number;
  onHover?: (cell: { col: number; row: number; x: number; y: number } | null) => void;
  className?: string;
}

// Cells on a canvas, crisp at any device pixel ratio. Used for strips and small matrices;
// the colour indices are computed by the caller, so a redraw happens only when they change.
export const HeatCanvas = memo(function HeatCanvas({ cols, rows, values, css, cellH, gap = 1, label, marks, firstCol = 0, onHover, className }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);
  const height = rows * cellH + (rows - 1) * gap;

  useLayoutEffect(() => {
    const el = ref.current?.parentElement;
    if (!el) return;
    const read = () => setWidth(Math.floor(el.clientWidth));
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useLayoutEffect(() => {
    const cv = ref.current;
    if (!cv || width <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(width * dpr);
    cv.height = Math.round(height * dpr);
    const g = cv.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, height);
    const cw = (width - (cols - 1) * gap) / cols;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (c < firstCol) {
          g.fillStyle = "rgb(255 255 255 / 0.04)";
        } else {
          g.fillStyle = css[values[r * cols + c]] ?? css[0];
        }
        g.fillRect(c * (cw + gap), r * (cellH + gap), Math.max(cw, 0.5), cellH);
      }
    }
    if (marks?.length) {
      g.strokeStyle = "oklch(0.78 0.14 55)";
      g.lineWidth = 1.5;
      for (const m of marks) g.strokeRect(m.col * (cw + gap) + 0.75, m.row * (cellH + gap) + 0.75, Math.max(cw, 1) - 1.5, cellH - 1.5);
    }
  }, [width, height, cols, rows, values, css, cellH, gap, marks, firstCol]);

  const locate = (e: RPE<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const col = Math.min(cols - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * cols)));
    const row = Math.min(rows - 1, Math.max(0, Math.floor((e.clientY - rect.top) / (cellH + gap))));
    return { col, row, x: e.clientX, y: e.clientY };
  };

  return (
    <canvas
      ref={ref}
      className={className}
      role="img"
      aria-label={label}
      style={{ width: "100%", height, display: "block", touchAction: onHover ? "pan-y" : undefined }}
      onPointerMove={onHover ? (e) => onHover(locate(e)) : undefined}
      onPointerDown={onHover ? (e) => onHover(locate(e)) : undefined}
      onPointerLeave={onHover ? () => onHover(null) : undefined}
    />
  );
});
