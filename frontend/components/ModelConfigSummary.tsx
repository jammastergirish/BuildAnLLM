import type { ReactNode } from "react";

import StatCard from "./StatCard";

type ConfigValue = string | number | boolean | null | undefined;

type ConfigRecord = Record<string, ConfigValue>;

type SummaryItem = {
  label: string;
  value: ReactNode;
};

type ConfigItem = {
  key: string;
  label: string;
  value: ConfigValue;
};

type ConfigSection = {
  title: string;
  items: ConfigItem[];
};

const LABELS: Record<string, string> = {
  architecture: "Architecture",
  positional_encoding: "Positional Encoding",
  normalization: "Normalization",
  activation: "Activation",
  attention_type: "Attention Type",
  tokenizer_type: "Tokenizer",
  d_model: "d_model",
  n_layers: "n_layers",
  n_heads: "n_heads",
  n_kv_heads: "n_kv_heads",
  d_head: "d_head",
  d_mlp: "d_mlp",
  n_ctx: "n_ctx",
  d_vocab: "d_vocab",
  rope_theta: "RoPE theta",
  use_moe: "Use MoE",
  num_experts: "Experts",
  num_experts_per_tok: "Experts per token",
  use_shared_experts: "Shared experts",
  num_shared_experts: "Shared expert count",
  router_type: "Router type",
  load_balancing_loss_weight: "Load balance weight",
  expert_capacity_factor: "Expert capacity factor",
  init_range: "Init range",
  layer_norm_eps: "Layer norm epsilon",
  debug: "Debug",
};

const SECTION_DEFS: Array<{ title: string; keys: string[] }> = [
  {
    title: "Components",
    keys: ["architecture", "positional_encoding", "normalization", "activation", "attention_type", "tokenizer_type"],
  },
  {
    title: "Dimensions",
    keys: ["d_model", "n_layers", "n_heads", "n_kv_heads", "d_head", "d_mlp", "n_ctx", "d_vocab"],
  },
  {
    title: "Positional Encoding",
    keys: ["rope_theta"],
  },
  {
    title: "Mixture of Experts",
    keys: [
      "use_moe",
      "num_experts",
      "num_experts_per_tok",
      "use_shared_experts",
      "num_shared_experts",
      "router_type",
      "load_balancing_loss_weight",
      "expert_capacity_factor",
    ],
  },
  {
    title: "Initialization",
    keys: ["init_range", "layer_norm_eps", "debug"],
  },
];

const formatValue = (value: ConfigValue) => {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return value.toString();
    }
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
};

const labelForKey = (key: string) => LABELS[key] ?? key;

const inferAttentionType = (config: ConfigRecord) => {
  const nHeads = Number(config.n_heads);
  const nKvHeads = Number(config.n_kv_heads);
  if (!Number.isFinite(nHeads) || !Number.isFinite(nKvHeads) || nHeads <= 0) {
    return null;
  }
  if (nKvHeads === nHeads) {
    return "Multi-Head (MHA)";
  }
  if (nKvHeads === 1) {
    return "Multi-Query (MQA)";
  }
  return "Grouped Query (GQA)";
};

const buildSections = (config: ConfigRecord): ConfigSection[] => {
  const attentionType = inferAttentionType(config);
  const usedKeys = new Set<string>();
  const sections: ConfigSection[] = [];

  SECTION_DEFS.forEach(({ title, keys }) => {
    const items: ConfigItem[] = [];
    keys.forEach((key) => {
      usedKeys.add(key);
      if (key === "attention_type") {
        if (attentionType) {
          items.push({ key, label: labelForKey(key), value: attentionType });
        }
        return;
      }
      const value = config[key];
      if (value === undefined || value === null) {
        return;
      }
      if (key === "rope_theta" && config.positional_encoding !== "rope") {
        return;
      }
      if (title === "Mixture of Experts" && config.use_moe === false && key !== "use_moe") {
        return;
      }
      items.push({ key, label: labelForKey(key), value });
    });
    if (items.length) {
      sections.push({ title, items });
    }
  });

  const remainingEntries = Object.entries(config).filter(([key, value]) => {
    if (usedKeys.has(key)) {
      return false;
    }
    return value !== undefined && value !== null;
  });

  if (remainingEntries.length) {
    const items = remainingEntries
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => ({ key, label: labelForKey(key), value }));
    sections.push({ title: "Other", items });
  }

  return sections;
};

export default function ModelConfigSummary({
  config,
  summaryItems = [],
}: {
  config: ConfigRecord;
  summaryItems?: SummaryItem[];
}) {
  const sections = buildSections(config);
  const hasSummary = summaryItems.length > 0;

  return (
    <div>
      {hasSummary && (
        <div className="grid-3" style={{ marginTop: 16 }}>
          {summaryItems.map((item) => (
            <StatCard key={item.label} label={item.label} value={item.value} />
          ))}
        </div>
      )}
      <div style={{ marginTop: hasSummary ? 16 : 0 }}>
        {sections.map((section) => (
          <details className="expander" key={section.title}>
            <summary>{section.title}</summary>
            <div className="expander-content">
              <table className="table">
                <tbody>
                  {section.items.map((item) => (
                    <tr key={item.key}>
                      <td>{item.label}</td>
                      <td>{formatValue(item.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}
