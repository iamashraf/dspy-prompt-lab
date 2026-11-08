# Security

## Reporting a vulnerability

Please do not open a public issue. Use the repository's private "Report a vulnerability" option under the Security tab. The maintainer replies and keeps you updated on the fix.

## Keys and logs

- API keys are read from the environment or a git-ignored `.env`. The repo never stores them.
- Run logs in `artifacts/logs/` hold prompts and model replies. The trace file keeps an allow-listed set of fields and never the API key, but it can contain your data. Do not share these files without reading them.
- `program.save(path, save_program=True)` writes a pickle. Loading a pickle runs code, so load only files you trust. The examples use the JSON form.
