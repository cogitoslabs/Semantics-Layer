# Test Action

1. Read `context/current-feature.md` and `context/features-specs/[feature-name].md` to understand the goals
2. Check if tests already exist for these functions
3. For functions without tests that have testable logic, write unit tests:
   - Create unit tests using Vitest
   - Test happy path and error cases
   - Do not write tests just to write them. Use your best judgement
4. Use "uv run pytest" to execute the tests
5. Report test coverage for the new feature code