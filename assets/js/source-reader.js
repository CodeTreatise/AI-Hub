/**
 * source-reader.js
 * 
 * Shared utilities for rendering captured web content (stored as JSON)
 * into a clean reader page without Tailwind/Alpine framework noise.
 */

export function absolutizeUrl(maybeUrl, baseUrl) {
    if (!maybeUrl) return null;
    try {
        return new URL(maybeUrl, baseUrl).toString();
    } catch {
        return maybeUrl;
    }
}

/**
 * Remove Tailwind/Alpine markup and interactive UI elements that
 * break without their frameworks (buttons, dropdowns, hidden elements, etc.)
 */
export function stripFrameworkNoise(root) {
    // Elements that are usually UI chrome / interactive controls in captured docs
    const removeSelectors = [
        "details",
        "summary",
        "button",
        "template",
        "noscript",
        "svg.hidden",
        "[aria-hidden='true']",
        ".sr-only",
        "[data-pagefind-ignore]"
    ];
    root.querySelectorAll(removeSelectors.join(",")).forEach((el) => el.remove());

    // These typically explode visually without Tailwind (icons + hidden blocks)
    root.querySelectorAll(".hidden, .icon-svg, .md-dropdown, .nx-sr-only").forEach((el) => el.remove());

    // Remove mobile TOC container from docs pages (avoid nuking code blocks)
    root.querySelectorAll("div.block").forEach((el) => {
        const text = (el.textContent || "").toLowerCase();
        if (text.includes("table of contents")) el.remove();
    });

    // Remove share buttons, copy page buttons, and other social chrome
    root.querySelectorAll("[aria-label*='Share'], [aria-label*='Copy']").forEach((el) => el.remove());

    // Remove Alpine.js attributes (x-*, @click, etc.) but keep the content
    root.querySelectorAll("*").forEach((el) => {
        [...el.attributes].forEach((attr) => {
            const name = attr.name;
            if (name.startsWith("x-") || name.startsWith("@") || name.startsWith(":")) {
                el.removeAttribute(name);
            }
            if (
                name === "data-heap-id" ||
                name === "data-pagefind-ignore" ||
                name === "data-pagefind-weight" ||
                name.startsWith("data-component") ||
                name.startsWith("data-testid")
            ) {
                el.removeAttribute(name);
            }
        });
    });
}

export function normalizeLinksAndImages(root, baseUrl) {
    // Force links to open in a new tab + fix relative URLs
    root.querySelectorAll("a").forEach((a) => {
        const href = a.getAttribute("href");
        if (href) a.setAttribute("href", absolutizeUrl(href, baseUrl));
        a.setAttribute("target", "_blank");
        a.setAttribute("rel", "noopener");
    });

    root.querySelectorAll("img").forEach((img) => {
        const src = img.getAttribute("src");
        if (src) img.setAttribute("src", absolutizeUrl(src, baseUrl));
        img.removeAttribute("srcset");
    });
}

export function stripClassesAndStyles(root) {
    // Remove classes to avoid weirdness from missing Tailwind utilities.
    // (We keep ids so heading anchors still work.)
    root.querySelectorAll("*").forEach((el) => {
        el.removeAttribute("class");
        el.removeAttribute("style");
    });
}

/**
 * Build a DocumentFragment from raw HTML content,
 * sanitized for display without Tailwind/Alpine.
 */
export function buildContentFragment(contentHtml, baseUrl) {
    const parsed = new DOMParser().parseFromString(contentHtml || "", "text/html");
    const article = parsed.body.querySelector("article") || parsed.body;

    stripFrameworkNoise(article);
    normalizeLinksAndImages(article, baseUrl);
    stripClassesAndStyles(article);

    const fragment = document.createDocumentFragment();
    [...article.childNodes].forEach((n) => fragment.appendChild(n.cloneNode(true)));
    return fragment;
}

/**
 * Initialize a source reader page.
 * 
 * @param {Object} options
 * @param {string} options.sourceJson - Path to the JSON file with captured content
 * @param {string} options.statusSelector - Selector for status element (default: "#status")
 * @param {string} options.contentSelector - Selector for content element (default: "#content")
 * @param {string} options.titleSelector - Selector for title element (default: "#title")
 * @param {string} options.sourceLinkSelector - Selector for source link (default: "#sourceLink")
 * @param {number} options.pageIndex - Which page from pages array to display (default: 0)
 */
export async function initSourceReader(options) {
    const {
        sourceJson,
        statusSelector = "#status",
        contentSelector = "#content",
        titleSelector = "#title",
        sourceLinkSelector = "#sourceLink",
        pageIndex = 0
    } = options;

    const statusEl = document.querySelector(statusSelector);
    const contentEl = document.querySelector(contentSelector);
    const titleEl = document.querySelector(titleSelector);
    const sourceLinkEl = document.querySelector(sourceLinkSelector);

    try {
        const res = await fetch(sourceJson, { cache: "no-store" });
        if (!res.ok) throw new Error(`Failed to fetch ${sourceJson} (${res.status})`);

        const data = await res.json();
        const page = (data.pages && data.pages[pageIndex]) || {};

        const pageUrl = page.url || data.url || (sourceLinkEl && sourceLinkEl.href) || "";
        const pageTitle = page.title || data.name || (titleEl && titleEl.textContent) || "Source";

        if (titleEl) titleEl.textContent = pageTitle;
        if (sourceLinkEl) {
            sourceLinkEl.href = pageUrl;
            sourceLinkEl.textContent = pageUrl;
        }

        const fragment = buildContentFragment(page.content_html, pageUrl);
        if (contentEl) {
            contentEl.replaceChildren(fragment);
            contentEl.hidden = false;
        }

        if (statusEl) statusEl.remove();
    } catch (err) {
        if (statusEl) {
            statusEl.textContent = `Could not render captured content. ${err && err.message ? err.message : ""}`;
        }
        if (contentEl) contentEl.hidden = true;
    }
}

/**
 * Default CSS for source reader pages.
 */
export const readerStyles = `
    body {
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem;
    }
    .reader-container {
        background: rgba(30, 30, 50, 0.7);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .meta-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.25);
    }
    .meta-header a { color: #58a6ff; }

    /* Readability helpers for captured HTML */
    .content img { max-width: 100%; height: auto; border-radius: 8px; margin: 1rem 0; }
    .content pre { background: rgba(20, 20, 40, 0.7); padding: 1rem; border-radius: 10px; overflow-x: auto; }
    .content code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .content h1, .content h2, .content h3 { margin-top: 1.4rem; margin-bottom: 0.6rem; }
    .content h1 { font-size: 1.8rem; }
    .content h2 { font-size: 1.5rem; }
    .content h3 { font-size: 1.25rem; }
    .content p { line-height: 1.7; margin: 0.75rem 0; color: #c9d1d9; }
    .content ul, .content ol { margin: 0.75rem 0 0.75rem 1.5rem; }
    .content ul { list-style-type: disc; }
    .content ol { list-style-type: decimal; }
    .content li { margin: 0.35rem 0; color: #c9d1d9; }
    .content blockquote { border-left: 4px solid rgba(99, 102, 241, 0.6); padding-left: 1rem; color: #b8c0cc; margin: 1rem 0; }
    .content table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    .content th, .content td { border: 1px solid rgba(99, 102, 241, 0.25); padding: 0.5rem; }
    .content a { color: #58a6ff; text-decoration: none; }
    .content a:hover { text-decoration: underline; }
    .content hr { border: none; border-top: 1px solid rgba(99, 102, 241, 0.25); margin: 1.5rem 0; }

    .status {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        background: rgba(20, 20, 40, 0.55);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #b8c0cc;
    }
`;
