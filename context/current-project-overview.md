# LLM Distillation 

**Target:** 1B–4B parameter Domain specialist SLM deployable on phone/laptop (int4 quantization)  
**Teachers:** Frontier Models including Claude, OpenAI, Gemini, DeepSeek, Qwen, Nvidia, Kimi etc.   
**Domain:** Domain reasoning, scientific explanation, knowledge retrieval, mechanistic pathway analysis, and research assistance

---

## Core Design Principles

1. Knowledge before reasoning.
2. Retrieval before memorization whenever practical.
3. Distill probabilities before representations.
4. Optimize for deployed scientific competence, not teacher imitation.
5. Separate development, validation, and release evaluation.
6. Prioritize reliability, calibration, and hallucination resistance.
7. Match the adaptation method (full-parameter vs. parameter-efficient) to available compute and the magnitude of domain shift — do not default to the most expensive method when a cheaper one meets the same gate.

---

