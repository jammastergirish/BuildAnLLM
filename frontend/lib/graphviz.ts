import type { ModelConfig } from "./modelConfig";

export function generateGraphvizArchitecture(config: ModelConfig) {
  const nHeads = config.n_heads || 4;
  const posEnc = config.positional_encoding || "learned";
  const activation = config.activation || "gelu";

  const colorBoxFill = "#E6DAC3";
  const colorBoxBorder = "#5C5248";
  const colorText = "#4A4A4A";
  const fontMain = "Helvetica";
  const fontMath = "Times-Roman";

  const dot: string[] = [];
  dot.push("digraph TransformerArchitecture {");
  dot.push('    bgcolor="transparent";');
  dot.push("    rankdir=BT;");
  dot.push("    nodesep=0.5;");
  dot.push("    ranksep=0.5;");
  dot.push("    splines=ortho;");
  dot.push(
    `    node [shape=rect, style="filled,rounded", fillcolor="${colorBoxFill}", color="${colorBoxBorder}", penwidth=1.5, fontname="${fontMain}", fontcolor="${colorText}"];`
  );
  dot.push(`    edge [color="${colorBoxBorder}", penwidth=1.2, arrowsize=0.8];`);

  const makeNoteLabel = (text: string, subtext = "") => {
    return `<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" FACE="${fontMain}">${text}</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="14" FACE="${fontMath}"><i>${subtext}</i></FONT></TD></TR>
        </TABLE>
        >`;
  };

  const nKv = config.n_kv_heads || nHeads;
  let headsTxt = nKv === nHeads ? "Attention Heads" : `GQA Heads (${nKv} KV)`;
  if (posEnc === "rope") {
    headsTxt += "\\n(RoPE)";
  } else if (posEnc === "alibi") {
    headsTxt += "\\n(ALiBi)";
  }

  let mlpTxt = "MLP m";
  if (config.use_moe) {
    const nExp = config.num_experts || 8;
    const kExp = config.num_experts_per_tok || 2;
    mlpTxt = `MoE MLP\\n(${nExp} Experts, top-${kExp})`;
  } else if (activation === "swiglu") {
    mlpTxt += "\\n(SwiGLU)";
  }

  dot.push('    tokens [label="tokens", shape=box3d, group=main];');
  dot.push('    embed [label="embed", group=main];');
  dot.push(`    note_embed [shape=plaintext, style=none, label=${makeNoteLabel("Token embedding.", "x&#8320; = W&#7431;t")}];`);

  dot.push('    unembed [label="unembed", group=main];');
  dot.push('    logits [label="logits", group=main];');
  dot.push(`    note_unembed [shape=plaintext, style=none, label=${makeNoteLabel("Unembedding", "T(t) = W&#7516; x&#8331;&#8321;")}];`);

  dot.push("    tokens -> embed [weight=100];");
  dot.push("    unembed -> logits [weight=100];");

  dot.push("    { rank=same; tokens; note_embed; }");
  dot.push("    { rank=same; unembed; note_unembed; }");
  dot.push("    embed -> note_embed [style=invis, minlen=2];");
  dot.push("    unembed -> note_unembed [style=invis, minlen=2];");

  dot.push("    subgraph cluster_residual {");
  dot.push('        label="One residual block (repeated)";');
  dot.push('        fontcolor="#888888";');
  dot.push("        style=dashed;");
  dot.push('        color="#888888";');
  dot.push("        margin=20;");

  dot.push('        x0 [shape=plaintext, style=none, label=<<i>x</i><sub>0</sub>>, group=main];');
  dot.push('        x1 [shape=plaintext, style=none, label=<<i>x</i><sub>i+1</sub>>, group=main];');
  dot.push('        x2 [shape=plaintext, style=none, label=<<i>x</i><sub>i+2</sub>>, group=main];');

  dot.push('        plus1 [shape=circle, label="+", fixedsize=true, width=0.4, group=main];');
  dot.push('        plus2 [shape=circle, label="+", fixedsize=true, width=0.4, group=main];');

  dot.push(`        heads [label="${headsTxt}", width=2.5];`);
  dot.push(`        mlp [label="${mlpTxt}", width=2.5];`);

  dot.push(
    `        note_attn [shape=plaintext, style=none, label=${makeNoteLabel(
      "Attention Layer",
      "x&#7522;&#8330;&#8321; = x&#7522; + &Sigma; h(x&#7522;)"
    )}];`
  );
  dot.push(
    `        note_mlp [shape=plaintext, style=none, label=${makeNoteLabel(
      "MLP Layer",
      "x&#7522;&#8330;&#8322; = x&#7522;&#8330;&#8321; + m(x&#7522;&#8330;&#8321;)"
    )}];`
  );

  dot.push("        { rank=same; x0; heads; note_attn; }");
  dot.push("        { rank=same; x1; mlp; note_mlp; }");

  dot.push("        x0 -> plus1 [weight=100];");
  dot.push("        plus1 -> x1 [weight=100];");
  dot.push("        x1 -> plus2 [weight=100];");
  dot.push("        plus2 -> x2 [weight=100];");

  dot.push("        x0 -> heads [constraint=false];");
  dot.push("        x1 -> mlp [constraint=false];");

  dot.push("        heads -> plus1;");
  dot.push("        mlp -> plus2;");

  dot.push("        heads -> note_attn [style=invis, minlen=1];");
  dot.push("        mlp -> note_mlp [style=invis, minlen=1];");

  dot.push("    }");

  dot.push("    embed -> x0 [weight=100];");
  dot.push('    x_final [shape=plaintext, style=none, label=<<i>x</i><sub>-1</sub>>, group=main];');
  dot.push('    x2 -> x_final [style=dotted, label="repeat...", weight=100];');
  dot.push("    x_final -> unembed [weight=100];");

  dot.push("}");

  return dot.join("\n");
}
