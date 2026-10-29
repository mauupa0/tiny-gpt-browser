import { StrictMode } from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import { App } from "./App";
import "./styles/tokens.css";
import "./styles/base.css";

const root = document.getElementById("root") as HTMLElement;
const tree = (
  <StrictMode>
    <App />
  </StrictMode>
);

// The production build ships the first state as HTML (see scripts/prerender.mjs), so the text is on
// screen before this script runs and React takes over the same nodes instead of replacing them.
if (root.hasChildNodes()) hydrateRoot(root, tree);
else createRoot(root).render(tree);
