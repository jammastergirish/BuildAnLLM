"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "./env";

export type SseEvent<T = unknown> = {
  type: string;
  payload: T;
};

export function useSse<T = unknown>(path?: string, active = true) {
  // The server can send two events back to back, such as "paused" followed by
  // the metrics of the step that was already running. React batches those state
  // updates, so holding only the newest event would drop the earlier one. Queue
  // them instead and expose one per render, so effects that depend on lastEvent
  // see every event in order. The newest event stays at the head once it's shown.
  const [queue, setQueue] = useState<SseEvent<T>[]>([]);
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
        setQueue((prev) => [...prev, { type, payload }]);
      } catch (err) {
        setError((err as Error).message);
      }
    };

    const eventTypes = ["status", "metrics", "eval", "checkpoint", "log", "done", "error"] as const;
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

  useEffect(() => {
    if (queue.length > 1) {
      setQueue((prev) => prev.slice(1));
    }
  }, [queue]);

  return { lastEvent: queue[0] ?? null, error };
}
