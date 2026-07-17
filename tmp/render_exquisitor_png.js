const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const dir = "C:/Users/Windows/.codex/visualizations/2026/07/21/019f8283-e4e6-7793-93e8-d6d032dcdc44";
  const fragment = fs.readFileSync(path.join(dir, "exquisitor-temporal-search.html"), "utf8");
  const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    :root {
      --background: #ffffff;
      --foreground: #111827;
      --card: #ffffff;
      --muted: #f3f4f6;
      --muted-foreground: #4b5563;
      --border: #9ca3af;
      --viz-series-1: #3b82f6;
      --viz-series-2: #22c55e;
      --viz-series-3: #eab308;
      --viz-series-4: #ef4444;
      --viz-series-5: #f97316;
    }
    body {
      margin: 0;
      background: var(--background);
      padding: 20px;
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
    }
    .wrap { width: 1240px; }
  </style>
</head>
<body>
  <div class="wrap">${fragment}</div>
</body>
</html>`;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 760 }, deviceScaleFactor: 2 });
  await page.setContent(html, { waitUntil: "load" });
  const element = await page.$("#exquisitor-temporal-search");
  await element.screenshot({ path: path.join(dir, "exquisitor-temporal-search.png") });
  await browser.close();
  console.log(path.join(dir, "exquisitor-temporal-search.png"));
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
