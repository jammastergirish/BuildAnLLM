import { useEffect, useState } from "react";
import { act, render, screen } from "@testing-library/react";

import { useSse } from "../lib/useSse";

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  url: string;
  listeners = new Map<string, Array<(event: MessageEvent) => void>>();
  onerror: ((event?: Event) => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, callback: (event: MessageEvent) => void) {
    const bucket = this.listeners.get(type) ?? [];
    bucket.push(callback);
    this.listeners.set(type, bucket);
  }

  emit(type: string, payload: unknown) {
    const data = typeof payload === "string" ? payload : JSON.stringify(payload);
    const event = { data } as MessageEvent;
    (this.listeners.get(type) ?? []).forEach((callback) => callback(event));
  }

  close() {
    this.closed = true;
  }
}

function SseTest({ path, active }: { path?: string; active?: boolean }) {
  const { lastEvent, error } = useSse(path, active);
  return (
    <div>
      <div data-testid="type">{lastEvent?.type ?? ""}</div>
      <div data-testid="payload">{lastEvent ? JSON.stringify(lastEvent.payload) : ""}</div>
      <div data-testid="error">{error ?? ""}</div>
    </div>
  );
}

function SseEffectLog({ path }: { path: string }) {
  const { lastEvent } = useSse(path);
  const [seen, setSeen] = useState<string[]>([]);
  useEffect(() => {
    if (lastEvent) {
      setSeen((prev) => [...prev, lastEvent.type]);
    }
  }, [lastEvent]);
  return (
    <div>
      <div data-testid="seen">{seen.join(",")}</div>
      <div data-testid="last">{lastEvent?.type ?? ""}</div>
    </div>
  );
}

describe("useSse", () => {
  beforeEach(() => {
    FakeEventSource.instances = [];
    vi.stubGlobal("EventSource", FakeEventSource);
  });

  it("connects and emits events", () => {
    render(<SseTest path="/events" active />);
    const instance = FakeEventSource.instances[0];

    act(() => {
      instance.emit("status", { ok: true });
    });

    expect(screen.getByTestId("type")).toHaveTextContent("status");
    expect(screen.getByTestId("payload")).toHaveTextContent('{"ok":true}');
  });

  it("shows effects every event when React batches them", () => {
    render(<SseEffectLog path="/events" />);
    const instance = FakeEventSource.instances[0];

    act(() => {
      instance.emit("status", { status: "paused" });
      instance.emit("metrics", { iter: 5 });
      instance.emit("log", { message: "hello" });
    });

    expect(screen.getByTestId("seen")).toHaveTextContent("status,metrics,log");
    expect(screen.getByTestId("last")).toHaveTextContent("log");
  });

  it("shows an effect each event only once across separate arrivals", () => {
    render(<SseEffectLog path="/events" />);
    const instance = FakeEventSource.instances[0];

    act(() => {
      instance.emit("status", { status: "running" });
    });
    act(() => {
      instance.emit("metrics", { iter: 1 });
    });

    expect(screen.getByTestId("seen")).toHaveTextContent(/^status,metrics$/);
  });

  it("reports JSON parse errors", () => {
    render(<SseTest path="/events" active />);
    const instance = FakeEventSource.instances[0];

    act(() => {
      instance.emit("status", "not-json");
    });

    expect(screen.getByTestId("error").textContent).not.toBe("");
  });

  it("skips connection when inactive", () => {
    render(<SseTest path="/events" active={false} />);
    expect(FakeEventSource.instances.length).toBe(0);
  });
});
