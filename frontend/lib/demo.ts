import { useEffect, useState } from "react";

const DEMO_HOSTS = new Set(["demo.buildanllm.com"]);

export function resolveDemoMode(hostname?: string): boolean {
  if (process.env.NEXT_PUBLIC_DEMO_MODE === "true") {
    return true;
  }
  if (!hostname) {
    return false;
  }
  return DEMO_HOSTS.has(hostname);
}

export function useDemoMode(): boolean {
  const defaultDemo = process.env.NEXT_PUBLIC_DEMO_MODE === "true";
  const [isDemo, setIsDemo] = useState(defaultDemo);

  useEffect(() => {
    if (defaultDemo || typeof window === "undefined") {
      return;
    }
    setIsDemo(resolveDemoMode(window.location.hostname));
  }, [defaultDemo]);

  return isDemo;
}
