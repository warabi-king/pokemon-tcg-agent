const cardsTarget = document.getElementById("cards");
const searchInput = document.getElementById("search");
const deckFileInput = document.getElementById("deckFile");
const downloadButton = document.getElementById("download");
const validationResult = document.getElementById("validationResult");
const deck = new Map();

let cards = [];
let cardsById = new Map();

const filterConfig = [
  ["kindFilter", "kind"],
  ["ruleFilter", "rule"],
  ["categoryFilter", "category"],
  ["typeFilter", "type"],
];

const totalCards = () => [...deck.values()].reduce((sum, count) => sum + count, 0);
const cardCount = (id) => deck.get(id) || 0;

function addCard(id) {
  if (totalCards() >= 60) return;
  deck.set(id, cardCount(id) + 1);
  clearValidation();
  renderAll();
}

function removeCard(id) {
  const count = cardCount(id);
  if (count <= 1) deck.delete(id);
  else deck.set(id, count - 1);
  clearValidation();
  renderAll();
}

function setFilterOptions(filters) {
  const sources = {
    kindFilter: filters.kinds,
    ruleFilter: filters.rules,
    categoryFilter: filters.categories,
    typeFilter: filters.types,
  };
  for (const [id, values] of Object.entries(sources)) {
    const select = document.getElementById(id);
    const first = select.firstElementChild;
    select.replaceChildren(first, ...values.map((value) => new Option(value, value)));
  }
}

function currentFilters() {
  return Object.fromEntries(filterConfig.map(([id, key]) => [key, document.getElementById(id).value]));
}

function filteredCards() {
  const query = searchInput.value.trim().toLowerCase();
  const filters = currentFilters();
  const sortMode = document.getElementById("sortMode").value;
  const result = cards.filter((card) => {
    if (query && !card.searchText.includes(query)) return false;
    return Object.entries(filters).every(([key, value]) => !value || card[key] === value);
  });

  result.sort((a, b) => {
    if (sortMode === "name") return a.name.localeCompare(b.name, "ja") || a.id - b.id;
    if (sortMode === "count") return cardCount(b.id) - cardCount(a.id) || a.id - b.id;
    return a.id - b.id;
  });
  return result;
}

function cardMetaLine(card) {
  return [card.kind, card.rule, card.category, card.type].filter(Boolean).join(" / ");
}

function renderCards() {
  const visible = filteredCards();
  document.getElementById("resultCount").textContent = `${visible.length}枚`;

  cardsTarget.replaceChildren(...visible.map((card) => {
    const button = document.createElement("article");
    button.className = "card";
    button.innerHTML = `
      <img loading="lazy" src="${card.image}" alt="${card.name}">
      <div class="card-title">
        <strong>${card.name}</strong>
        <span>#${card.id} ${card.expansion} ${card.collectionNumber}</span>
      </div>
      <div class="meta-row">${cardMetaLine(card)}</div>
      <div class="attack-line">${card.attacks[0]?.effect || card.attacks[0]?.name || ""}</div>
      <div class="card-actions">
        <button type="button" data-action="remove" aria-label="${card.name}を1枚減らす">−</button>
        <b>${cardCount(card.id)}</b>
        <button type="button" data-action="add" aria-label="${card.name}を1枚増やす">＋</button>
      </div>
    `;
    button.querySelector('[data-action="remove"]').addEventListener("click", () => removeCard(card.id));
    button.querySelector('[data-action="add"]').addEventListener("click", () => addCard(card.id));
    button.querySelector("img").addEventListener("click", () => showDetail(card.id));
    button.querySelector(".card-title").addEventListener("click", () => showDetail(card.id));
    return button;
  }));
}

function renderDeck() {
  const total = totalCards();
  document.getElementById("deckCount").textContent = `${total} / 60`;
  document.getElementById("deckMeter").style.width = `${Math.min(100, total / 60 * 100)}%`;
  downloadButton.disabled = total === 0;

  const entries = [...deck.entries()]
    .map(([id, count]) => ({ card: cardsById.get(id), id, count }))
    .sort((a, b) => (a.card?.id || a.id) - (b.card?.id || b.id));

  document.getElementById("deckList").replaceChildren(...entries.map(({ card, id, count }) => {
    const row = document.createElement("div");
    row.className = "deck-row";
    row.innerHTML = `
      <div>
        <strong>${card?.name || `#${id}`}</strong>
        <span>#${id}${card ? ` ${card.kind || ""} ${card.type || ""}` : ""}</span>
      </div>
      <div class="deck-row-controls">
        <button type="button" data-action="remove" aria-label="1枚減らす">−</button>
        <b>${count}</b>
        <button type="button" data-action="add" aria-label="1枚増やす">＋</button>
      </div>
    `;
    row.querySelector('[data-action="remove"]').addEventListener("click", () => removeCard(id));
    row.querySelector('[data-action="add"]').addEventListener("click", () => addCard(id));
    return row;
  }));

}

function renderAll() {
  renderCards();
  renderDeck();
}

function deckIds() {
  return [...deck.entries()]
    .sort(([a], [b]) => a - b)
    .flatMap(([id, count]) => Array(count).fill(id));
}

function isBasicEnergy(card) {
  return card?.kind === "基本エネルギー";
}

function validateDeck() {
  const total = totalCards();
  const errors = [];
  const warnings = [];
  const nameCounts = new Map();
  let aceSpecCount = 0;
  const unknownIds = [];

  if (total !== 60) {
    errors.push(`デッキ枚数が60枚ではありません。現在は${total}枚です。`);
  }

  for (const [id, count] of deck.entries()) {
    const card = cardsById.get(id);
    if (!card) {
      unknownIds.push(`#${id}`);
      continue;
    }
    if (!isBasicEnergy(card)) {
      nameCounts.set(card.name, (nameCounts.get(card.name) || 0) + count);
    }
    if (card.rule === "ACE SPEC") {
      aceSpecCount += count;
    }
  }

  for (const [name, count] of nameCounts.entries()) {
    if (count > 4) {
      errors.push(`「${name}」が${count}枚入っています。同名カードは4枚までです。`);
    }
  }

  if (aceSpecCount > 1) {
    errors.push(`ACE SPEC が${aceSpecCount}枚入っています。ACE SPEC は1枚までです。`);
  }

  if (unknownIds.length) {
    warnings.push(`カードデータにないIDがあります: ${unknownIds.join(", ")}`);
  }

  if (!total) {
    warnings.push("デッキが空です。");
  }

  return { errors, warnings };
}

function clearValidation() {
  validationResult.className = "validation-result";
  validationResult.replaceChildren();
}

function renderValidation() {
  const { errors, warnings } = validateDeck();
  const items = [
    ...errors.map((message) => ({ type: "error", message })),
    ...warnings.map((message) => ({ type: "warning", message })),
  ];

  if (!items.length) {
    validationResult.className = "validation-result ok";
    validationResult.innerHTML = "<strong>OK</strong><p>現在のデッキは制約チェックを通過しました。</p>";
    return;
  }

  validationResult.className = "validation-result ng";
  validationResult.innerHTML = `
    <strong>${errors.length ? "NG" : "注意"}</strong>
    <ul>${items.map((item) => `<li class="${item.type}">${item.message}</li>`).join("")}</ul>
    <p>チェック結果に関係なく deck.csv は保存できます。</p>
  `;
}

function downloadDeck() {
  const blob = new Blob([`${deckIds().join("\n")}\n`], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "deck.csv";
  link.click();
  URL.revokeObjectURL(url);
}

async function importDeck(file) {
  const text = await file.text();
  const ids = text.split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => Number(line.split(",")[0]));
  deck.clear();
  for (const id of ids) {
    if (Number.isInteger(id) && id > 0) deck.set(id, cardCount(id) + 1);
  }
  clearValidation();
  renderAll();
}

async function copyDeckList() {
  const lines = [...deck.entries()]
    .sort(([a], [b]) => a - b)
    .map(([id, count]) => `${count} ${cardsById.get(id)?.name || `#${id}`}`);
  await navigator.clipboard.writeText(lines.join("\n"));
}

function showDetail(id) {
  const card = cardsById.get(id);
  if (!card) return;
  const dialog = document.getElementById("detailDialog");
  const chips = [card.kind, card.rule, card.category, card.type, card.evolvesFrom && `進化前: ${card.evolvesFrom}`]
    .filter(Boolean)
    .map((value) => `<span class="chip">${value}</span>`)
    .join("");
  const attacks = card.attacks.length
    ? card.attacks.map((attack) => `
      <div class="attack">
        <strong>${attack.name || "効果"}</strong>
        <span>${[attack.cost, attack.damage].filter(Boolean).join(" / ")}</span>
        <p>${attack.effect || ""}</p>
      </div>
    `).join("")
    : '<div class="attack"><span>ワザなし</span></div>';

  document.getElementById("detailBody").innerHTML = `
    <div class="detail">
      <img src="${card.image}" alt="${card.name}">
      <div>
        <h3>${card.name}</h3>
        <div class="meta-row">#${card.id} ${card.expansion} ${card.collectionNumber}</div>
        <div class="detail-meta">${chips}</div>
        <div class="meta-row">HP ${card.hp || "-"} / 弱点 ${card.weakness || "-"} / 抵抗力 ${card.resistance || "-"} / にげる ${card.retreat || "-"}</div>
        ${attacks}
      </div>
    </div>
  `;
  dialog.showModal();
}

searchInput.addEventListener("input", renderCards);
for (const [id] of filterConfig) document.getElementById(id).addEventListener("change", renderCards);
document.getElementById("sortMode").addEventListener("change", renderCards);
document.getElementById("resetFilters").addEventListener("click", () => {
  searchInput.value = "";
  for (const [id] of filterConfig) document.getElementById(id).value = "";
  renderCards();
});
document.getElementById("clear").addEventListener("click", () => {
  deck.clear();
  clearValidation();
  renderAll();
});
document.getElementById("validateDeck").addEventListener("click", renderValidation);
downloadButton.addEventListener("click", downloadDeck);
deckFileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) importDeck(file);
});
document.getElementById("copyList").addEventListener("click", copyDeckList);
document.getElementById("closeDetail").addEventListener("click", () => document.getElementById("detailDialog").close());

fetch("/api/cards")
  .then((response) => response.json())
  .then((payload) => {
    cards = payload.cards;
    cardsById = new Map(cards.map((card) => [card.id, card]));
    setFilterOptions(payload.filters);
    renderAll();
  });
