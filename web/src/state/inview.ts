import { useEffect, useState, type RefObject } from "react";

// True once the element has entered the upper part of the viewport. Stays true, and works for elements taller than the screen.
export function useSeen(ref: RefObject<Element | null>, bottomMargin = "-30%"): boolean {
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin: `0px 0px ${bottomMargin} 0px` },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [ref, bottomMargin, seen]);
  return seen;
}

// Live visibility, for sections that should switch work on and off.
export function useVisible(ref: RefObject<Element | null>, margin = "0px"): boolean {
  const [vis, setVis] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setVis(e.isIntersecting), { rootMargin: margin });
    io.observe(el);
    return () => io.disconnect();
  }, [ref, margin]);
  return vis;
}
