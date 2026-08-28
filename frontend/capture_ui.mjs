import { chromium } from "playwright-core";

const browser = await chromium.launch({
  headless: true,
  executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
await page.goto("http://127.0.0.1:5173", { waitUntil: "networkidle" });

await page.locator('input[placeholder*="Describe"]').fill("black leather shoulder bag for women");
await page.locator('textarea[placeholder*="color"]').fill("color#:#black#;#material#:#leather");
await page.screenshot({ path: "../docs/report/figure/ui-raw-query-form.png", fullPage: true });

await page.locator('input[placeholder*="Example: 624257086318"]').fill("624257086318");
await page.getByRole("button", { name: "Search products" }).click();
await page.locator(".results-panel").waitFor({ timeout: 30000 });
await page.screenshot({ path: "../docs/report/figure/ui-catalog-search.png", fullPage: true });

await page.locator(".result-card").first().click();
await page.locator(".product-detail").waitFor({ timeout: 30000 });
await page.screenshot({ path: "../docs/report/figure/ui-product-detail.png", fullPage: true });

await browser.close();
