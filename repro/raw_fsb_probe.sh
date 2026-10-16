#!/bin/bash
# Raw authenticated probe of api.opentela.ai flash-sandbox start_sandbox,
# the same way run_train_qwen3p5_bristen.sh extracts the key.
set -uo pipefail
KEY="$(sed -nE 's/^[[:space:]]*export[[:space:]]+OTELA_API_KEY=([^[:space:]]+).*/\1/p' "${HOME}/.zshrc" | head -1)"
# strip surrounding quotes
KEY="${KEY%\"}"; KEY="${KEY#\"}"
KEY="${KEY%\'}"; KEY="${KEY#\'}"
echo "[raw] key len=${#KEY} starts_sk=${KEY:0:3}"

echo "=== raw authenticated POST start_sandbox (libkrun) — exact status + body ==="
curl -sS -o /tmp/_fsb_body.txt -w "HTTP %{http_code}  time=%{time_total}s  ctype=%{content_type}\n" \
  --max-time 45 \
  -X POST "https://api.opentela.ai/v1/service/flash-sandbox/" \
  -H "Authorization: Bearer ${KEY}" \
  -H "Content-Type: application/json" \
  -d '{"image":"sweb.eval.x86_64.docker","type":"libkrun","command":["tail","-f","/dev/null"],"ttl_seconds":300}'
echo "=== body ==="
cat /tmp/_fsb_body.txt 2>/dev/null; echo
rm -f /tmp/_fsb_body.txt

echo "=== GET /sandboxes (list) — is auth accepted? ==="
curl -sS -o /tmp/_lst.txt -w "GET /sandboxes -> HTTP %{http_code}  ctype=%{content_type}\n" --max-time 20 \
  "https://api.opentela.ai/v1/service/flash-sandbox/sandboxes" \
  -H "Authorization: Bearer ${KEY}"
echo "--- list body (first 200 chars) ---"; head -c 200 /tmp/_lst.txt 2>/dev/null; echo
rm -f /tmp/_lst.txt

echo "=== which backend names does the service advertise? GET / (service root) ==="
curl -sS -o /tmp/_root.txt -w "GET / -> HTTP %{http_code}\n" --max-time 20 \
  "https://api.opentela.ai/v1/service/flash-sandbox/" \
  -H "Authorization: Bearer ${KEY}"
echo "--- root body (first 300 chars) ---"; head -c 300 /tmp/_root.txt 2>/dev/null; echo
rm -f /tmp/_root.txt
