import type { ReactNode } from "react";

export default function StatCard({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="stat">
      <h4>{label}</h4>
      <p>{value}</p>
    </div>
  );
}
