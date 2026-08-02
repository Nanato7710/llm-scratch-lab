(function () {
  "use strict";

  const documents = window.LLM_SCRATCH_DOCS;
  const sourceTargets = {
    "architecture.md": "#architecture",
    "adding-components.md": "#adding-components",
    "experiment-guide.md": "#experiment-guide",
  };
  const usedIds = new Set();

  function escapeHtml(value) {
    return value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function uniqueId(documentId, heading) {
    const normalized = heading
      .replace(/`([^`]+)`/g, "$1")
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .normalize("NFKC")
      .toLowerCase()
      .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
      .replace(/^-|-$/g, "");
    const base = `${documentId}-${normalized || "section"}`;
    let candidate = base;
    let index = 2;
    while (usedIds.has(candidate) || document.getElementById(candidate)) {
      candidate = `${base}-${index}`;
      index += 1;
    }
    usedIds.add(candidate);
    return candidate;
  }

  function normalizeLink(target) {
    const [path, hash] = target.split("#", 2);
    if (sourceTargets[path]) {
      return hash ? `${sourceTargets[path]}-${hash}` : sourceTargets[path];
    }
    return target;
  }

  function renderInline(value) {
    const codeTokens = [];
    let working = value.replace(/`([^`]+)`/g, (_, code) => {
      const token = `DOCSCODETOKEN${codeTokens.length}X`;
      codeTokens.push(code);
      return token;
    });
    working = escapeHtml(working);
    working = working.replace(/\[([^\]]+)]\(([^)]+)\)/g, (_, label, target) => {
      const normalized = normalizeLink(target);
      const external = /^https?:\/\//.test(normalized);
      const attributes = external ? ' target="_blank" rel="noreferrer"' : "";
      return `<a href="${escapeHtml(normalized)}"${attributes}>${label}</a>`;
    });
    working = working.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    codeTokens.forEach((code, index) => {
      working = working.replace(
        `DOCSCODETOKEN${index}X`,
        `<code>${escapeHtml(code)}</code>`,
      );
    });
    return working;
  }

  function isTableDivider(line) {
    return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line);
  }

  function tableCells(line) {
    return line
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());
  }

  function isBlockStart(lines, index) {
    const line = lines[index] || "";
    const next = lines[index + 1] || "";
    return (
      !line.trim() ||
      /^#{1,3}\s+/.test(line) ||
      /^```/.test(line) ||
      /^[-*]\s+/.test(line) ||
      /^\d+\.\s+/.test(line) ||
      (line.includes("|") && isTableDivider(next))
    );
  }

  function renderMarkdown(markdown, documentId) {
    const lines = markdown.replace(/\r\n/g, "\n").split("\n");
    const html = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }

      const fence = line.match(/^```([\w+-]*)\s*$/);
      if (fence) {
        const language = fence[1] || "text";
        const code = [];
        index += 1;
        while (index < lines.length && !/^```\s*$/.test(lines[index])) {
          code.push(lines[index]);
          index += 1;
        }
        index += 1;
        html.push(
          `<div class="code-block"><div class="code-toolbar"><span>${escapeHtml(language.toUpperCase())}</span><button class="copy-button" type="button">コピー</button></div><pre><code class="language-${escapeHtml(language)}">${escapeHtml(code.join("\n"))}</code></pre></div>`,
        );
        continue;
      }

      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        const level = heading[1].length;
        const text = heading[2].trim();
        const id = level === 1 ? `${documentId}-title` : uniqueId(documentId, text);
        html.push(
          `<h${level} id="${id}" data-heading-text="${escapeHtml(text)}"><a class="heading-anchor" href="#${id}" aria-label="${escapeHtml(text)}へのリンク">#</a>${renderInline(text)}</h${level}>`,
        );
        index += 1;
        continue;
      }

      if (line.includes("|") && isTableDivider(lines[index + 1] || "")) {
        const headers = tableCells(line);
        index += 2;
        const rows = [];
        while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
          rows.push(tableCells(lines[index]));
          index += 1;
        }
        html.push(
          `<div class="table-wrap"><table><thead><tr>${headers.map((cell) => `<th>${renderInline(cell)}</th>`).join("")}</tr></thead><tbody>${rows
            .map(
              (row) =>
                `<tr>${row.map((cell) => `<td>${renderInline(cell)}</td>`).join("")}</tr>`,
            )
            .join("")}</tbody></table></div>`,
        );
        continue;
      }

      const unordered = line.match(/^[-*]\s+(.+)$/);
      const ordered = line.match(/^\d+\.\s+(.+)$/);
      if (unordered || ordered) {
        const tag = ordered ? "ol" : "ul";
        const pattern = ordered ? /^\d+\.\s+(.+)$/ : /^[-*]\s+(.+)$/;
        const items = [];
        while (index < lines.length) {
          const item = lines[index].match(pattern);
          if (!item) break;
          items.push(`<li>${renderInline(item[1])}</li>`);
          index += 1;
        }
        html.push(`<${tag}>${items.join("")}</${tag}>`);
        continue;
      }

      const paragraph = [line.trim()];
      index += 1;
      while (index < lines.length && !isBlockStart(lines, index)) {
        paragraph.push(lines[index].trim());
        index += 1;
      }
      html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    }
    return html.join("\n");
  }

  function renderDocuments() {
    if (!Array.isArray(documents)) {
      throw new Error("Web documentation content could not be loaded");
    }
    documents.forEach((documentData) => {
      const target = document.getElementById(`${documentData.id}-content`);
      if (!target) return;
      target.innerHTML = renderMarkdown(documentData.markdown, documentData.id);
    });
  }

  function buildTableOfContents() {
    const container = document.getElementById("generated-toc");
    documents.forEach((documentData, documentIndex) => {
      const group = document.createElement("div");
      group.className = "toc-group";
      const parent = document.createElement("a");
      parent.href = `#${documentData.id}`;
      parent.innerHTML = `<span>0${documentIndex + 1}</span>${escapeHtml(documentData.title)}`;
      group.append(parent);

      const children = document.createElement("div");
      children.className = "toc-children";
      document
        .querySelectorAll(`#${documentData.id}-content h2`)
        .forEach((heading) => {
          const link = document.createElement("a");
          link.href = `#${heading.id}`;
          link.textContent = heading.dataset.headingText;
          children.append(link);
        });
      group.append(children);
      container.append(group);
    });
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    const copied = document.execCommand("copy");
    area.remove();
    return copied ? Promise.resolve() : Promise.reject(new Error("Copy failed"));
  }

  function enableCopyButtons() {
    document.querySelectorAll(".copy-button").forEach((button) => {
      button.addEventListener("click", async () => {
        const code = button.closest(".code-block").querySelector("code").textContent;
        try {
          await copyText(code);
          button.textContent = "コピー済み";
        } catch {
          button.textContent = "選択してコピー";
        }
        window.setTimeout(() => {
          button.textContent = "コピー";
        }, 1600);
      });
    });
  }

  function setupSearch() {
    const input = document.getElementById("doc-search");
    const results = document.getElementById("search-results");
    const entries = [
      ...documents.map((documentData) => ({
        title: documentData.title,
        chapter: "章の先頭",
        href: `#${documentData.id}`,
      })),
      ...Array.from(document.querySelectorAll(".rendered-markdown h2, .rendered-markdown h3")).map(
        (heading) => ({
          title: heading.dataset.headingText,
          chapter: heading.closest(".doc-chapter").querySelector(".chapter-header h2").textContent,
          href: `#${heading.id}`,
        }),
      ),
    ];

    function closeResults() {
      results.hidden = true;
      input.setAttribute("aria-expanded", "false");
    }

    function updateResults() {
      const query = input.value.trim().normalize("NFKC").toLowerCase();
      if (!query) {
        closeResults();
        return;
      }
      const matches = entries
        .filter((entry) => `${entry.title} ${entry.chapter}`.normalize("NFKC").toLowerCase().includes(query))
        .slice(0, 10);
      results.innerHTML = matches.length
        ? matches
            .map(
              (entry) =>
                `<a class="search-result" href="${entry.href}"><strong>${escapeHtml(entry.title)}</strong><small>${escapeHtml(entry.chapter)}</small></a>`,
            )
            .join("")
        : '<div class="search-empty">一致する見出しがありません。</div>';
      results.hidden = false;
      input.setAttribute("aria-expanded", "true");
    }

    input.addEventListener("input", updateResults);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        input.value = "";
        closeResults();
        input.blur();
      }
      if (event.key === "ArrowDown") {
        const first = results.querySelector("a");
        if (first) {
          event.preventDefault();
          first.focus();
        }
      }
    });
    results.addEventListener("click", closeResults);
    document.addEventListener("click", (event) => {
      if (!event.target.closest(".search-wrap")) closeResults();
    });
    document.addEventListener("keydown", (event) => {
      const tag = document.activeElement.tagName;
      if (event.key === "/" && tag !== "INPUT" && tag !== "TEXTAREA") {
        event.preventDefault();
        input.focus();
      }
    });
  }

  function setupTheme() {
    const button = document.getElementById("theme-toggle");
    const stored = (() => {
      try {
        return localStorage.getItem("llm-scratch-docs-theme");
      } catch {
        return null;
      }
    })();
    const initial = stored || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.dataset.theme = initial;

    button.addEventListener("click", () => {
      const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = theme;
      try {
        localStorage.setItem("llm-scratch-docs-theme", theme);
      } catch {
        // The selected theme still applies for the current page.
      }
    });
  }

  function setupMobileNavigation() {
    const button = document.getElementById("menu-toggle");
    const sidebar = document.getElementById("sidebar");
    const backdrop = document.getElementById("sidebar-backdrop");

    function setOpen(open) {
      sidebar.classList.toggle("open", open);
      backdrop.hidden = !open;
      button.setAttribute("aria-expanded", String(open));
      button.setAttribute("aria-label", open ? "目次を閉じる" : "目次を開く");
    }

    button.addEventListener("click", () => setOpen(!sidebar.classList.contains("open")));
    backdrop.addEventListener("click", () => setOpen(false));
    sidebar.addEventListener("click", (event) => {
      if (event.target.closest("a")) setOpen(false);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && sidebar.classList.contains("open")) setOpen(false);
    });
  }

  function setupReadingState() {
    const progress = document.getElementById("reading-progress");
    const links = Array.from(document.querySelectorAll("#table-of-contents a[href^='#']"));
    const targets = links
      .map((link) => document.getElementById(link.getAttribute("href").slice(1)))
      .filter(Boolean);

    function updateProgress() {
      const available = document.documentElement.scrollHeight - window.innerHeight;
      const ratio = available > 0 ? Math.min(window.scrollY / available, 1) : 0;
      progress.style.width = `${ratio * 100}%`;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (!visible) return;
        links.forEach((link) => {
          link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
        });
      },
      { rootMargin: "-18% 0px -70% 0px", threshold: 0 },
    );
    targets.forEach((target) => observer.observe(target));
    window.addEventListener("scroll", updateProgress, { passive: true });
    updateProgress();
  }

  function scrollToInitialHash() {
    if (!window.location.hash) return;
    const target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
    if (target) window.requestAnimationFrame(() => target.scrollIntoView());
  }

  renderDocuments();
  buildTableOfContents();
  enableCopyButtons();
  setupSearch();
  setupTheme();
  setupMobileNavigation();
  setupReadingState();
  scrollToInitialHash();
})();
