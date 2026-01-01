"use client";

import { useEffect, useState } from "react";
import { Viz } from "@viz-js/viz";

export default function GraphvizDiagram({ dot }: { dot: string }) {
  const [svg, setSvg] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const render = async () => {
      try {
        const viz = new Viz();
        const output = await viz.renderString(dot);
        if (mounted) {
          setSvg(output);
        }
      } catch (err) {
        if (mounted) {
          setError((err as Error).message);
        }
      }
    };
    render();

    return () => {
      mounted = false;
    };
  }, [dot]);

  if (error) {
    return <div className="code-block">{error}</div>;
  }

  if (!svg) {
    return <div className="card">Rendering diagram...</div>;
  }

  return (
    <div
      className="card"
      style={{ boxShadow: "none", background: "rgba(255,255,255,0.55)" }}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
