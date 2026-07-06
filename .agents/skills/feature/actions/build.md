# Build Action

1. Read `context/current-feature.md`
2. Read `context/features-specs/[feature-name].md` (using the parsed Feature Name if provided; otherwise, derive it from the active feature in `current-feature.md`).
3. Set Status to "Build in Progress" in `context/change-history.md`
4. Create and checkout the feature branch (derive name from the H1 heading or the feature name)
5. List the goals, then implement them one by one
6. **Test** - write end to end test in @tests/[feature-name].md using pytest framework
7. **Iterate** - Iterate and change things if needed
