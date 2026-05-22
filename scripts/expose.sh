#!/usr/bin/env bash
# Phase 21: expose the local wrangler dev to the internet via a
# cloudflared quick tunnel. The tunnel URL is printed once it's live;
# share it with friends to add their browser tabs as real peers.
#
# Requires `cloudflared` (https://github.com/cloudflare/cloudflared).
# No Cloudflare auth required for quick tunnels — they spin up a
# random `*.trycloudflare.com` URL.
#
# Caveats:
#  - Quick tunnels are ephemeral; URL changes every run.
#  - WebSocket works through the tunnel (verified for /api/*/ws).
#  - The `wrangler dev` instability is unchanged; this just changes
#    how the same wrangler instance is reachable.

set -euo pipefail

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERR: cloudflared not found. Install with:"
  echo "  brew install cloudflared   # macOS"
  echo "  apt install cloudflared    # Debian/Ubuntu"
  echo "  see https://github.com/cloudflare/cloudflared/releases for binaries"
  exit 2
fi

# Verify wrangler dev is up locally
if ! curl -s -o /dev/null -w '%{http_code}' http://localhost:8787/ | grep -q 200; then
  echo "ERR: nothing on http://localhost:8787 — start \`npx wrangler dev --port 8787\` first"
  exit 2
fi

echo "Starting cloudflared quick tunnel to http://localhost:8787 ..."
echo "(your tunnel URL will print below; share it to add cross-machine peers)"
echo ""
cloudflared tunnel --url http://localhost:8787
