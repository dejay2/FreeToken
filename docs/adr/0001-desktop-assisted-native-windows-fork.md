# ADR 0001: Use the Desktop-provided Windows engine for the public fork

- Status: Accepted
- Date: 2026-08-30
- Decision owner: Jay (`dejay2`)

## Context

FreeToken has no official standalone Windows wheel or command-only installer. The official install guide and release assets are Linux-only, and the request for a Windows wheel remains open as issue #130.

The proven native-Windows setup runs PR #279 source from a separate checkout but reuses the Python environment, compiled Windows DLLs, CUDA sources, and libraries installed by FreeToken Desktop. The Desktop application is not opened or modified while the server runs.

Closed PR #232 previously demonstrated the same broad source-checkout/installed-kernel pattern, loopback TCP worker addresses, a Windows-compatible Uvicorn loop, Windows JIT flags, and a parameter-based PowerShell launcher. It was closed because its model implementation was superseded, not because the Windows ideas were rejected.

## Decision

The first public fork will generalize the setup that was actually proven:

1. Require an existing FreeToken Desktop installation on Windows.
2. Run the fork's Python source independently from the Desktop UI.
3. Discover the Desktop Python and kernel locations from standard installation paths or explicit parameters, never from Jay's personal paths.
4. Keep all compatibility behavior in a clearly labeled Windows shim and launcher rather than claiming a standalone source build.
5. Credit PR #232 where its earlier approach is adapted.
6. Publish the fork and branch, open no upstream pull request, and post only one factual validation comment on PR #279 after the measurements pass review.

## Consequences

- Other Windows Desktop users can reproduce the tested SSD-backed PLE setup without editing their Desktop installation.
- The fork is not a standalone Windows distribution and must say so clearly.
- The launcher must check prerequisites and fail with plain, actionable messages.
- A future standalone Windows package remains separate work.
- Internal planning files, pi settings, and machine-specific paths are not public fork content.
