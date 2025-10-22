# Coding Agent Instructions

## Programming Style

- Inside the code, always fail fast: throw exceptions immediately for unexpected errors instead of hiding them or applying silent workarounds. This makes bugs obvious and easier to debug during early release.
- At system and UI boundaries, gracefully catch errors: log and report them (for crash reporting, telemetry, analytics) while converting them into user-friendly fallback behavior. The goal is to avoid jarring UX while still surfacing problems to developers.
- Do not swallow errors silently. Every caught exception should either be reported or transformed into a clear, safe fallback state for the user.
- Favor clear error boundaries (e.g., global exception handlers, supervisors, or UI error boundaries) that isolate failures without compromising the rest of the app.
