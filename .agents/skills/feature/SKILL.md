---
name: feature
description: Manage current feature workflow - start, review, explain or complete
argument-hint: (load|design|build|review|explain|complete) [feature-name]
---

# Feature Workflow

Manages the full lifecycle of a feature from spec to merge.

## Working File

@context/current-feature.md

current-feature.md has the objective and highlevel details of the current feature.  

## Task

Parse the input: `$ARGUMENTS`
- **Action**: The first word of `$ARGUMENTS` (e.g., `load`, `design`, `build`, `review`, `explain`, `complete`).
- **Feature Name**: The second word of `$ARGUMENTS` (optional). If not specified, fall back to deriving the feature name from `context/current-feature.md`.

Execute the requested Action using the parsed Feature Name.

| Action | Description |
|--------|-------------|
| `load` | Load feature from feature-hub into current-feature.md |
| `design` | Write the detailed Design Specifications |
| `build` | Begin implementation, create branch |
| `review` | Check goals met, code quality |
| `explain` | Document what changed and why |
| `complete` | Commit, push, merge, reset |

See [actions/](actions/) for detailed instructions.

If no action provided, explain the available options.