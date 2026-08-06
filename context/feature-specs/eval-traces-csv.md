# Feature Spec: Evaluation Traces CSV Logging

## Overview
Replaces JSON-based evaluation sample output logging (`EVAL_SAMPLES_FILE` / `logs/dapt_eval_samples.json`) with CSV-based evaluation trace logging (`EVAL_TRACES_FILE` / `logs/dapt_eval_traces.csv`).

## Key Requirements
1. Rename configuration environment variable from `EVAL_SAMPLES_FILE` to `EVAL_TRACES_FILE`.
2. Rename `LoggingConfig.eval_samples_file` to `LoggingConfig.eval_traces_file` (default `logs/dapt_eval_traces.csv`).
3. Update variable names across evaluation runners, probes, and tests from `eval_samples` to `eval_traces`.
4. Output detailed evaluation pass results to CSV instead of JSON.
5. Standardize CSV schema across all probe types (`QA`, `Cloze`, `Concept`):
   - `Eval #`: Evaluation pass number (e.g. `1`, `2`, `final`)
   - `Eval Category`: Category of evaluation probe (`QA`, `Cloze`, or `Concept`)
   - `Eval Seq #`: 1-based sequence index of the sample in its corresponding input file
   - `Eval`: JSON string dump of the raw input sample entry
   - `Generated Answer by the model`: Model output string (chosen option, generated text, or completions)
   - `Matching Score`: Metric score (`1.0`/`0.0` for exact match probes, float for continuous metrics like BERTScore/F1)
   - `Result`: `Pass` or `Fail` based on threshold/correctness

## Output File Format Example
```csv
Eval #,Eval Category,Eval Seq #,Eval,Generated Answer by the model,Matching Score,Result
1,QA,1,"{""question"": ""What is dopamine?"", ""choices"": [""Neurotransmitter"", ""Hormone""], ""answer_idx"": 0}",Neurotransmitter,1.0,Pass
1,Cloze,1,"{""prompt"": ""___ is a neurotransmitter."", ""target_term"": ""dopamine"", ""category"": ""neuro""}",dopamine,1.0,Pass
1,Concept,1,"{""prompt"": ""Explain synaptic plasticity"", ""reference"": ""Ref text""}",Generated response text,0.8542,Pass
```
