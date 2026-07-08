const $ = (id) => document.getElementById(id);

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

async function initialize() {
  try {
    const [meta, status] = await Promise.all([api("/api/meta"), api("/api/rooms/status")]);
    for (const name of meta.agents) {
      $("deckA").add(new Option(name, name));
      $("deckB").add(new Option(name, name));
    }
    $("capacity").textContent = `使用中 ${status.activeRooms} / ${status.maxRooms} ルーム`;
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
    await api("/api/rooms", {
      method: "POST",
      body: JSON.stringify({
        deckA: $("deckA").value,
        deckB: $("deckB").value,
      }),
    });
    window.location.assign("/duel");
  } catch (error) {
    showError(error.message);
    setBusy(event.currentTarget, false);
  }
});

$("joinForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(event.currentTarget, true);
  try {
    await api("/api/rooms/join", {
      method: "POST",
      body: JSON.stringify({ roomId: $("roomId").value }),
    });
    window.location.assign("/duel");
  } catch (error) {
    showError(error.message);
    setBusy(event.currentTarget, false);
  }
});

initialize();
