import { StrictMode } from "react";
import { renderToString } from "react-dom/server";
import { App } from "./App";

// The page in its first state, as HTML. scripts/prerender.mjs puts it into dist/index.html so the
// text paints before any script has run; the client then renders the same tree over it.
export function renderShell(): string {
  return renderToString(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
