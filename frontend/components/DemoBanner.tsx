"use client";

import { useDemoMode } from "../lib/demo";

export default function DemoBanner() {
  const isDemo = useDemoMode();

  if (!isDemo) {
    return null;
  }

  return (
    <div className="demo-banner is-demo">
      <div className="demo-banner-card">
        <div>
          <div className="demo-banner-title">Demo mode</div>
          <p className="demo-banner-text">
            <a
              href="https://github.com/jammastergirish/buildanllm"
              target="_blank"
              rel="noopener noreferrer"
            >
              This public demo disables key features.<br/>Clone the repo to access them.
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
