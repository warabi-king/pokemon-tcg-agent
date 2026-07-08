const $ = (id) => document.getElementById(id);

function enterDuel(result) {
  sessionStorage.setItem("duelPlayerToken", result.playerToken);
  window.location.assign("/duel");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function showError(message) {
  $("errorMessage").textContent = message;
  $("errorMessage").classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setBusy(form, busy) {
  form.querySelector("button").disabled = busy;
}

async function readDeckFile(input) {
  if (!input.files.length) return null;
  const values = (await input.files[0].text()).split(/[\s,]+/).filter(Boolean);
  const ids = values.map(Number);
  if (ids.length !== 60 || ids.some(id => !Number.isInteger(id) || id <= 0)) {
    throw new Error("CSVにはカードIDを60件指定してください。");
  }
  return ids;
}

async function initialize() {
  try {
    const [meta, status] = await Promise.all([api("/api/meta"), api("/api/rooms/status")]);
    for (const name of meta.agents) {
      $("deckA").add(new Option(name, name));
      $("joinDeck").add(new Option(name, name));
    }
    $("capacity").textContent = `使用中 ${status.activeMatches} / ${status.maxRooms} 対戦`;
  } catch (error) {
    showError(error.message);
  }
}

$("roomId").addEventListener("input", (event) => {
  event.target.value = event.target.value.toUpperCase().replace(/[^A-Z0-9]/g, "");
});

$("createForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(event.currentTarget, true);
  try {
    const deckAIds = await readDeckFile($("deckAFile"));
    const result = await api("/api/rooms", {
      method: "POST",
      body: JSON.stringify({
        deckA: $("deckA").value,
        deckAIds,
      }),
    });
    enterDuel(result);
  } catch (error) {
    showError(error.message);
    setBusy(event.currentTarget, false);
  }
});

$("joinForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(event.currentTarget, true);
  try {
    const deckIds = await readDeckFile($("joinDeckFile"));
    const result = await api("/api/rooms/join", {
      method: "POST",
      body: JSON.stringify({ roomId: $("roomId").value, deck: $("joinDeck").value, deckIds }),
    });
    enterDuel(result);
  } catch (error) {
    showError(error.message);
    setBusy(event.currentTarget, false);
  }
});

initialize();
