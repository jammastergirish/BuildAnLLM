"use client";

import { useEffect, useState } from "react";

import { fetchJson } from "../lib/api";
import StatCard from "./StatCard";

type SystemInfo = {
  cpu: string;
  ram_gb: string;
  gpu: string;
  mps: string;
  os: string;
  python: string;
  torch: string;
};

export default function SystemInfoPanel() {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);

  useEffect(() => {
    let active = true;
    fetchJson<SystemInfo>("/api/system/info")
      .then((data) => {
        if (active) {
          setSystemInfo(data);
        }
      })
      .catch(() => {
        if (active) {
          setSystemInfo(null);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="card">
      {systemInfo ? (
        <div className="grid-3">
          <StatCard label="CPU" value={systemInfo.cpu} />
          <StatCard label="RAM" value={`${systemInfo.ram_gb} GB`} />
          <StatCard label="GPU" value={systemInfo.gpu} />
          <StatCard label="MPS" value={systemInfo.mps} />
          <StatCard label="OS" value={systemInfo.os} />
          <StatCard label="Python" value={systemInfo.python} />
          <StatCard label="PyTorch" value={systemInfo.torch} />
        </div>
      ) : (
        <p>Unable to load system info.</p>
      )}
    </div>
  );
}
