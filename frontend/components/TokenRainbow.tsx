\"use client\";

import { useMemo } from "react";

function colorForIndex(idx: number) {
  const hue = (idx * 47) % 360;
  return `hsl(${hue}deg 70% 80%)`;
}

export default function TokenRainbow({ tokens }: { tokens: string[] }) {
  const colors = useMemo(() => tokens.map((_, idx) => colorForIndex(idx)), [tokens]);
  return (
    <div style={{ lineHeight: 1.8, fontFamily: "JetBrains Mono, Courier New, monospace" }}>
      {tokens.map((token, idx) => (
        <span
          key={`${token}-${idx}`}
          style={{
            background: colors[idx],
            padding: "2px 6px",
            marginRight: 4,
            borderRadius: 6,
            display: "inline-block",
          }}
        >
          {token || " "}
        </span>
      ))}
    </div>
  );
}
