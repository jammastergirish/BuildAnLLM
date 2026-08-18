# CBRN Emergency-Preparedness Fine-Tuning

This example fine-tunes `google/flan-t5-small` on questions about chemical,
biological, radiological, and nuclear emergency preparedness. It uses a small,
curated prompt-response dataset based on public guidance from CDC and FEMA.

The project is deliberately limited to protective public-health information.
It excludes agent creation, weaponization, harmful dispersal, optimization, and
evasion of safety or security controls.

## Run in Google Colab

1. Upload `cbrn_finetuning_colab.ipynb` to
   [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → T4 GPU** when available.
3. Run the cells in order.
4. When prompted, upload `finetune_cbrn.py` and
   `cbrn_preparedness.jsonl` from this folder.
5. Download the trained model and `evaluation.json` report from the final cell.

## Run locally

Install Transformers and start training:

```bash
uv pip install "transformers>=4.40,<6"
python examples/cbrn_finetuning/finetune_cbrn.py
```

Use fewer epochs for a quick smoke test:

```bash
python examples/cbrn_finetuning/finetune_cbrn.py \
  --epochs 2 \
  --batch-size 4 \
  --output checkpoints/cbrn-flan-t5-smoke
```

The output folder contains a standard Hugging Face checkpoint and an
`evaluation.json` file with:

- training and validation losses;
- baseline and fine-tuned answers for held-out questions;
- mean token-overlap F1;
- explicit-refusal and safe-redirection rates on held-out harmful requests;
- unsupported contact-detail rate to expose a common hallucination;
- the source associated with each reference answer.

## Verified run

A local CPU run used 39 training, 6 validation, and 7 test examples. Early
stopping selected epoch 5. Held-out token-overlap F1 increased from `0.116` to
`0.278`, and both harmful test prompts were redirected to safe topics.

The model also invented contact details in 2 of 7 test answers. This result is
reported as a limitation, not hidden as a successful prediction.

## Data sources

- [CDC Public Health Emergency Preparedness](https://www.cdc.gov/readiness/php/phep/index.html)
- [CDC Chemical Emergencies](https://www.cdc.gov/chemical-emergencies/about/index.html)
- [CDC Radiation Emergency Safety](https://www.cdc.gov/radiation-emergencies/safety/index.html)
- [CDC Radiation Emergency FAQ](https://www.cdc.gov/radiation-emergencies/faq/index.html)
- [CDC Viral Hemorrhagic Fever Response Planning](https://www.cdc.gov/viral-hemorrhagic-fevers/php/public-health-strategy/vhf-response-planning.html)
- [FEMA all-hazards preparedness sheets](https://www.ready.gov/sites/default/files/2025-02/fema_full-suite-hazard-info-sheets.pdf)

This educational model is not a substitute for emergency services, medical
advice, incident-specific instructions, or official public-health guidance.
