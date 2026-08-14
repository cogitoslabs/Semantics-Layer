# Current Feature

Trace logging functionality: 
1. When ever we run eval, we should log the results (model's output, reference answer, score, etc.) in @logs/traces/[eval_name]/[timestamp].csv
2. We should add a UI in @ui/trace_log.py to view the trace logs
    a. We should be able to chose the eval_name (Cloze, QA, Concept)   
    b. We should be able to select the timestamp (from most recent to oldest) or checkpoint
    c. We shoul have a flag to show only traces that changed from base model output
    d. Should be able to click thru all the traces for the selected eval and timestamp
    e. For each trace, we should show the prompt, model's output, reference answer, score, etc. both for base model and check point model
    


<!-- Goals -->

<!-- Notes -->