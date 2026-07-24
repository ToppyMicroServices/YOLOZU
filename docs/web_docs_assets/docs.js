(function () {
  "use strict";

  const searchInput = document.querySelector("[data-docs-search]");
  const searchResults = document.querySelector("[data-docs-search-results]");
  const localCards = Array.from(document.querySelectorAll("[data-search-card]"));
  let searchIndex = [];

  function normalize(value) {
    return String(value || "").toLocaleLowerCase().replace(/\s+/g, " ").trim();
  }

  function renderGlobalResults(query) {
    if (!searchResults) return;
    searchResults.replaceChildren();
    if (!query || searchIndex.length === 0) return;

    const matches = searchIndex
      .filter((item) => normalize(item.search_text).includes(query))
      .slice(0, 12);
    for (const item of matches) {
      const link = document.createElement("a");
      link.className = "search-result";
      link.href = item.href;
      link.textContent = item.title;
      const detail = document.createElement("small");
      detail.textContent = `${item.kind} · ${item.summary}`;
      link.appendChild(detail);
      searchResults.appendChild(link);
    }
    if (matches.length === 0) {
      const empty = document.createElement("div");
      empty.className = "search-result";
      empty.textContent = "No matching generated entry.";
      searchResults.appendChild(empty);
    }
  }

  function applySearch() {
    if (!searchInput) return;
    const query = normalize(searchInput.value);
    for (const card of localCards) {
      const haystack = normalize(card.getAttribute("data-search-text") || card.textContent);
      card.hidden = Boolean(query) && !haystack.includes(query);
    }
    renderGlobalResults(query);
  }

  if (searchInput) {
    searchInput.addEventListener("input", applySearch);
    fetch("search-index.json", { credentials: "same-origin" })
      .then((response) => {
        if (!response.ok) throw new Error("search index unavailable");
        return response.json();
      })
      .then((payload) => {
        searchIndex = Array.isArray(payload) ? payload : [];
        applySearch();
      })
      .catch(() => {
        searchIndex = [];
      });
  }

  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) return;
    const target = event.target.closest("[data-docs-event]");
    if (!target || typeof window.plausible !== "function") return;
    window.plausible(target.getAttribute("data-docs-event"), {
      props: {
        target: target.getAttribute("data-event-target") || "unknown",
        page: document.body.getAttribute("data-page") || "unknown"
      }
    });
  });
})();
