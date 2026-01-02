"use client";

import { useEffect, useState } from "react";
import { instance } from "@viz-js/viz";
import type { Viz } from "@viz-js/viz";

let vizInstancePromise: Promise<Viz> | null = null;

const getVizInstance = () => {
  if (!vizInstancePromise) {
    vizInstancePromise = instance();
  }
  return vizInstancePromise;
};

export default function GraphvizDiagram({ dot }: { dot: string }) {
  const [svg, setSvg] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const render = async () => {
      try {
        const viz = await getVizInstance();
        const output = viz.renderString(dot, { format: "svg", engine: "dot" });
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
      className="card graphviz-diagram"
      style={{ boxShadow: "none", background: "var(--card-muted)" }}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
