const cardsTarget = document.getElementById("cards");
const deck = new Map();
let cards = [];

const imageUrl = (id) => `/cards/${String(id).padStart(4, "0")}.jpg`;

function add(id) {
  if ([...deck.values()].reduce((a, b) => a + b, 0) >= 60) return;
  deck.set(id, (deck.get(id) || 0) + 1);
  renderDeck();
}

function remove(id) {
  const count = deck.get(id) || 0;
  if (count <= 1) deck.delete(id); else deck.set(id, count - 1);
  renderDeck();
}

function renderCards() {
  const query = document.getElementById("search").value.trim().toLowerCase();
  const filtered = cards.filter(card => !query || String(card.id).includes(query) || card.name.toLowerCase().includes(query));
  document.getElementById("resultCount").textContent = `${filtered.length}枚`;
  cardsTarget.replaceChildren(...filtered.map(card => {
    const button = document.createElement("button");
    button.className = "card";
    button.innerHTML = `<img loading="lazy" src="${imageUrl(card.id)}" alt=""><span>#${card.id} ${card.name}</span><b>＋</b>`;
    button.addEventListener("click", () => add(card.id));
    return button;
  }));
}

function renderDeck() {
  const total = [...deck.values()].reduce((a, b) => a + b, 0);
  document.getElementById("deckCount").textContent = `${total} / 60`;
  document.getElementById("download").disabled = total !== 60;
  const list = document.getElementById("deckList");
  list.replaceChildren(...[...deck.entries()].map(([id, count]) => {
    const card = cards.find(item => item.id === id);
    const row = document.createElement("div");
    row.innerHTML = `<span>#${id} ${card?.name || ""}</span><div><button>−</button><b>${count}</b><button>＋</button></div>`;
    const buttons = row.querySelectorAll("button");
    buttons[0].addEventListener("click", () => remove(id));
    buttons[1].addEventListener("click", () => add(id));
    return row;
  }));
}

document.getElementById("search").addEventListener("input", renderCards);
document.getElementById("clear").addEventListener("click", () => { deck.clear(); renderDeck(); });
document.getElementById("download").addEventListener("click", () => {
  const ids = [...deck.entries()].flatMap(([id, count]) => Array(count).fill(id));
  const blob = new Blob([`${ids.join("\n")}\n`], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = "deck.csv"; link.click();
  URL.revokeObjectURL(link.href);
});

fetch("/api/meta").then(response => response.json()).then(meta => {
  cards = Object.values(meta.cards).sort((a, b) => a.id - b.id);
  renderCards(); renderDeck();
});
