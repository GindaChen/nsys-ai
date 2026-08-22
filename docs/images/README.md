# Documentation screenshots

These PNGs are committed screenshots of the browser surfaces. The tree,
guided-loop, and diff captures use a 1440×1000 viewport; the timeline capture
uses 1440×420 so its controls and GPU lanes fill the frame instead of leaving
most of the image empty. Each file must remain below **300 KiB**.

The images are referenced by the README, [`user/viewers.md`](../user/viewers.md),
and [`guided-loop-setup.md`](../guided-loop-setup.md). Keep the four captures in
one directory so a documentation review can find and refresh them together.

## Regenerate

Install Playwright in a temporary environment if it is not already available:

```bash
mkdir -p /tmp/nsys-ai-docs-capture
cd /tmp/nsys-ai-docs-capture
npm install --no-save playwright
npx playwright install chromium
```

Start each local surface in a separate terminal from the repository root:

```bash
PROFILE=tests/fixtures/h100_2gpu_1s.sqlite
BEFORE=tests/fixtures/mfu_2gpu_before.sqlite
AFTER=tests/fixtures/mfu_2gpu_after.sqlite

nsys-ai web "$PROFILE" --port 18242 --no-browser
nsys-ai timeline-web "$PROFILE" --port 18244 --no-browser

cp "$BEFORE" /tmp/nsys-ai-docs-capture/before.sqlite
cp "$AFTER" /tmp/nsys-ai-docs-capture/after.sqlite
nsys-ai diff-web \
  /tmp/nsys-ai-docs-capture/before.sqlite \
  /tmp/nsys-ai-docs-capture/after.sqlite \
  --port 18245 --no-browser
```

Capture all four pages with the repository helper. It opens the Workflow
inspector for `guided-loop.png` and closes it for `timeline-web.png`, so the
checked-in views can be regenerated without relying on browser state:

```bash
NODE_PATH=/tmp/nsys-ai-docs-capture/node_modules \
  node scripts/capture_docs_screenshots.cjs \
  --web-url=http://127.0.0.1:18242/ \
  --timeline-url=http://127.0.0.1:18244/ \
  --diff-url=http://127.0.0.1:18245/ \
  --output=docs/images
```

Use the printed URL when a requested port is busy. Before committing, run the
documentation contract test; it checks that every referenced capture exists,
is a PNG, and stays within the size budget.
