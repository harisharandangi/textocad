# Security

Please do not open public issues for security problems.
Email security@textocad.com with details and steps to reproduce. You will get a reply within 72 hours.

## Running generated code

TextoCAD runs AI-written CadQuery / OpenSCAD scripts. The built-in checks (import allowlist,
banned names, resource limits) reduce risk but are not a security boundary. If you expose the
HTTP server to other people, run it in a container with no secrets in its environment, no
outbound network, a non-root user and a read-only filesystem apart from the build folder.

## Your API key

A key added in the web page is stored in `~/.textocad/credentials.json` (readable only by your user on macOS and Linux) and is sent only to the provider it belongs to. The server refuses key changes from other websites and from any address other than `127.0.0.1` / `localhost`.
