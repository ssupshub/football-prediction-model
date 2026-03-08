// ─── AnimatedNumber ───────────────────────────────────────────────────────────
// Smoothly counts from 0 to `value` over ~900ms using requestAnimationFrame.

import { useState, useEffect, useRef } from "react";

export default function AnimatedNumber({ value, decimals = 1, suffix = "" }) {
  const [display, setDisplay] = useState(0);
  const rafRef = useRef(null);

  useEffect(() => {
    const end = value, duration = 900, startTime = performance.now();
    const tick = (now) => {
      const elapsed  = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased    = 1 - Math.pow(1 - progress, 3);
      setDisplay(end * eased);
      if (progress < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value]);

  return <>{display.toFixed(decimals)}{suffix}</>;
}
