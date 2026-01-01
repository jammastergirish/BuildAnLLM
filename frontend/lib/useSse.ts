"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "./env";

export type SseEvent<T = unknown> = {
  type: string;
  payload: T;
};

export function useSse<T = unknown>(path?: string, active = true) {
  const [lastEvent, setLastEvent] = useState<SseEvent<T> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!path || !active) {
      return;
    }

    const source = new EventSource(`${API_BASE_URL}${path}`);
    sourceRef.current = source;

    const handle = (event: MessageEvent, type: string) => {
      try {
        const payload = event.data ? JSON.parse(event.data) : null;
        setLastEvent({ type, payload });
      } catch (err) {
        setError((err as Error).message);
      }
    };

    const eventTypes = ["status", "metrics", "eval", "checkpoint", "done", "error"] as const;
    eventTypes.forEach((type) => {
      source.addEventListener(type, (event) => handle(event as MessageEvent, type));
    });

    source.onerror = () => {
      setError("SSE connection error");
    };

    return () => {
      source.close();
    };
  }, [path, active]);

  return { lastEvent, error };
}
