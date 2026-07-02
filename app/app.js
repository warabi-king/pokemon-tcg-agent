const OPTION_TYPES = [
  "数を選ぶ", "はい", "いいえ", "カードを選ぶ", "ポケモンのどうぐ", "エネルギーカード",
  "エネルギー", "カードを使う", "カードをつける", "進化する", "特性を使う", "場から捨てる",
  "にげる", "ワザを使う", "番を終わる", "効果を選ぶ", "特殊状態を選ぶ"
];

const CONTEXTS = [
  "メインフェイズ", "バトル場に出す", "ベンチに出す", "いれかえる", "バトル場へ", "ベンチへ",
  "場に出す", "手札に加える", "トラッシュする", "山札へ戻す", "山札の下へ戻す", "サイドへ",
  "そのまま残す", "ダメカンをのせる", "ダメカンをのせる", "ダメージを与える", "ダメカンを取り除く",
  "回復する", "進化元を選ぶ", "進化先を選ぶ", "退化する", "つける先を選ぶ", "つけるカードを選ぶ",
  "外す元を選ぶ", "カードを見る", "効果の対象", "エネルギーをトラッシュ", "どうぐをトラッシュ",
  "エネルギーをいれかえる", "カードをトラッシュ", "エネルギーをトラッシュ", "エネルギーを手札へ",
  "エネルギーを山札へ", "エネルギーをいれかえる", "効果の順番", "ワザを選ぶ", "使えないワザを選ぶ",
  "進化を選ぶ", "引く枚数", "ダメカンの数", "取り除く数", "先攻を選ぶ", "引き直し", "効果を使う",
  "先に使う効果", "さらに退化", "コインの表", "特殊状態にする", "特殊状態を回復"
];

const LOG_TYPES = [
  "山札を切った", "たねポケモン確認", "番が始まった", "番を終えた", "カードを引いた", "相手がカードを引いた",
  "カードを移動した", "裏向きカードを移動した", "ポケモンをいれかえた", "ポケモンが変化した", "カードを使った",
  "カードをつけた", "進化した", "特性を使った", "ワザを使った", "ダメージ", "回復", "きぜつ", "サイドを取った"
];

let state = null;
let meta = { cards: {}, attacks: {} };
let selected = new Set();
let busy = false;

const $ = (id) => document.getElementById(id);
const padCardId = (id) => String(id).padStart(4, "0");
const cardId = (card) => card && (card.id ?? card.cardId);
const cardMeta = (id) => meta.cards[String(id)] || { id, name: `カード #${id}` };

function cardImageUrl(id) {
  return `/cards/${padCardId(id)}.jpg`;
}

function cardElement(card, options = {}) {
  const element = document.createElement("div");
  element.className = options.mini ? "mini-card" : "card";
  if (!card) {
    element.classList.add("empty");
    return element;
  }
  if (options.back) {
    element.innerHTML = '<div class="card-back"></div>';
    element.style.cursor = "default";
    return element;
  }

  const id = cardId(card);
  const image = document.createElement("img");
  image.src = cardImageUrl(id);
  image.alt = cardMeta(id).name;
  image.loading = "lazy";
  element.appendChild(image);
  element.title = `${cardMeta(id).name} (#${id})`;
  element.addEventListener("click", () => showPreview(id));

  if (!options.mini && card.hp != null) {
    const hp = document.createElement("span");
    hp.className = "hp-badge";
    hp.textContent = `HP ${card.hp}/${card.maxHp}`;
    element.appendChild(hp);
  }
  if (!options.mini) {
    const attachments = (card.energies?.length || 0) + (card.tools?.length || 0);
    if (attachments) {
      const badge = document.createElement("span");
      badge.className = "attach-badge";
      badge.textContent = `ENERGY ${card.energies?.length || 0} / TOOL ${card.tools?.length || 0}`;
      element.appendChild(badge);
    }
  }
  return element;
}

function renderSlots(target, cards, count) {
  target.replaceChildren();
  for (let i = 0; i < count; i += 1) target.appendChild(cardElement(cards?.[i] || null));
}

function renderPile(target, label, count, topCard = null, hidden = false) {
  target.replaceChildren();
  let visual;
  if (topCard) visual = cardElement(topCard);
  else if (hidden || count > 0) visual = cardElement({}, { back: true });
  else visual = cardElement(null);
  target.appendChild(visual);
  const copy = document.createElement("span");
  copy.innerHTML = `${label}<span class="pile-count">${count}</span>`;
  target.appendChild(copy);
}

function renderBoard(observation) {
  const current = observation?.current;
  if (!current) return;
  const human = current.players[0];
  const opponent = current.players[1];

  renderSlots($("opponentBench"), opponent.bench, opponent.benchMax || 5);
  renderSlots($("humanBench"), human.bench, human.benchMax || 5);
  renderSlots($("opponentActive"), opponent.active, 1);
  renderSlots($("humanActive"), human.active, 1);

  renderPile($("opponentDeck"), "山札", opponent.deckCount, null, true);
  renderPile($("humanDeck"), "山札", human.deckCount, null, true);
  renderPile($("opponentDiscard"), "トラッシュ", opponent.discard?.length || 0, opponent.discard?.at(-1));
  renderPile($("humanDiscard"), "トラッシュ", human.discard?.length || 0, human.discard?.at(-1));

  $("opponentStats").innerHTML = `<span>HAND ${opponent.handCount}</span><span>PRIZE ${opponent.prize?.length || 0}</span>`;
  $("humanStats").innerHTML = `<span>HAND ${human.handCount}</span><span>PRIZE ${human.prize?.length || 0}</span>`;

  const hand = $("humanHand");
  hand.replaceChildren();
  if (!human.hand?.length) hand.innerHTML = '<div class="empty-message">手札はありません</div>';
  else human.hand.forEach((card) => hand.appendChild(cardElement(card)));

  const stadium = $("stadium");
  stadium.replaceChildren();
  if (current.stadium?.[0]) stadium.appendChild(cardElement(current.stadium[0], { mini: true }));
}

function findCardForOption(option, observation) {
  const current = observation?.current;
  const selection = observation?.select;
  if (!current) return null;
  const playerIndex = option.playerIndex ?? current.yourIndex;
  const player = current.players[playerIndex];
  const area = option.area;
  const index = option.index ?? 0;
  if (area === 1) return selection?.deck?.[index] || null;
  if (area === 2) return player?.hand?.[index] || null;
  if (area === 3) return player?.discard?.[index] || null;
  if (area === 4) return player?.active?.[index] || null;
  if (area === 5) return player?.bench?.[index] || null;
  if (area === 6) return player?.prize?.[index] || null;
  if (area === 7) return current.stadium?.[index] || null;
  if (area === 12) return current.looking?.[index] || null;
  if ([7, 8, 9].includes(option.type)) return player?.hand?.[index] || null;
  if ([10, 11].includes(option.type)) {
    if (area === 4) return player?.active?.[index] || null;
    if (area === 5) return player?.bench?.[index] || null;
  }
  return option.cardId ? { id: option.cardId } : null;
}

function optionDescription(option, card) {
  if (option.type === 0) return `${option.number}`;
  if (option.type === 1) return "はい";
  if (option.type === 2) return "いいえ";
  if (option.type === 12) return "バトルポケモンをベンチといれかえる";
  if (option.type === 13) {
    const attack = meta.attacks[String(option.attackId)];
    return attack ? `${attack.name}${attack.damage ? ` — ${attack.damage}ダメージ` : ""}` : `ワザ #${option.attackId}`;
  }
  if (option.type === 14) return "自分の番を終える";
  if (card) return cardMeta(cardId(card)).name;
  return OPTION_TYPES[option.type] || `選択肢 ${option.type}`;
}

function renderActions(observation) {
  selected = new Set();
  const selection = observation?.select;
  const list = $("actionList");
  list.replaceChildren();
  if (!selection || !state?.humanTurn) {
    $("actionTitle").textContent = state?.finished ? "対戦終了" : "AIが選択中";
    $("selectionRule").textContent = "";
    $("submitAction").disabled = true;
    list.innerHTML = `<div class="empty-message">${state?.finished ? resultMessage() : "少々お待ちください"}</div>`;
    return;
  }

  $("actionTitle").textContent = CONTEXTS[selection.context] || "行動を選択";
  $("selectionRule").textContent = `${selection.minCount}〜${selection.maxCount}個`;
  selection.option.forEach((option, index) => {
    const card = findCardForOption(option, observation);
    const button = document.createElement("button");
    button.type = "button";
    button.className = `action-option${card ? "" : " no-card"}`;
    button.dataset.index = index;
    const heading = optionDescription(option, card);
    button.innerHTML = `<span class="option-index">${index + 1}</span><span class="option-copy"><strong>${escapeHtml(heading)}</strong><small>${escapeHtml(OPTION_TYPES[option.type] || "合法手")}</small></span>`;
    if (card) button.appendChild(cardElement(card, { mini: true }));
    button.addEventListener("click", () => toggleSelection(index, selection));
    list.appendChild(button);
  });

  const contextTarget = $("contextCard");
  contextTarget.replaceChildren();
  const context = selection.contextCard || selection.effect;
  if (context) {
    const row = document.createElement("div");
    row.appendChild(cardElement(context, { mini: true }));
    const label = document.createElement("span");
    label.innerHTML = `<strong>${escapeHtml(cardMeta(cardId(context)).name)}</strong><br><small>このカードの効果を処理中</small>`;
    row.appendChild(label);
    contextTarget.appendChild(row);
  }
  updateSubmit(selection);
}

function toggleSelection(index, selection) {
  if (selected.has(index)) selected.delete(index);
  else {
    if (selection.maxCount === 1) selected.clear();
    if (selected.size < selection.maxCount) selected.add(index);
  }
  document.querySelectorAll(".action-option").forEach((button) => {
    button.classList.toggle("selected", selected.has(Number(button.dataset.index)));
  });
  updateSubmit(selection);
}

function updateSubmit(selection) {
  const valid = selected.size >= selection.minCount && selected.size <= selection.maxCount;
  $("submitAction").disabled = busy || !valid;
  $("submitAction").textContent = `決定 (${selected.size}/${selection.maxCount})`;
}

function renderEvents(events) {
  const log = $("eventLog");
  log.replaceChildren();
  if (!events?.length) {
    log.innerHTML = '<div class="empty-message">対戦ログはまだありません</div>';
    return;
  }
  events.slice(-40).reverse().forEach((event) => {
    const item = document.createElement("div");
    item.className = `event-item${event.source === "AI" ? " ai" : ""}`;
    const id = event.cardId || event.cardIdAfter || event.cardIdTarget;
    const detail = id ? ` — ${cardMeta(id).name}` : event.value != null ? ` — ${event.value}` : "";
    item.innerHTML = `<strong>${escapeHtml(event.source)}</strong> ${escapeHtml(LOG_TYPES[event.type] || `イベント ${event.type}`)}${escapeHtml(detail)}`;
    log.appendChild(item);
  });
}

function resultMessage() {
  if (!state?.finished) return "";
  const reward = state.states?.[0]?.reward;
  if (reward > 0) return "YOU WIN — 勝利しました";
  if (reward < 0) return "YOU LOSE — AIの勝利です";
  return "DRAW — 引き分けです";
}

function render() {
  const observation = state?.observation;
  renderBoard(observation);
  renderActions(observation);
  renderEvents(state?.events);
  $("stepLabel").textContent = `STEP ${state?.step ?? 0}`;

  const badge = $("turnBadge");
  badge.classList.toggle("waiting", !state?.humanTurn);
  badge.textContent = state?.finished ? resultMessage() : state?.humanTurn ? "YOUR MOVE" : "AI THINKING";
  const bannerMessage = state?.error;
  $("errorBanner").textContent = bannerMessage || "";
  $("errorBanner").classList.toggle("hidden", !bannerMessage);
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

async function startNewGame() {
  if (busy) return;
  busy = true;
  $("newGameButton").disabled = true;
  $("turnBadge").textContent = "SHUFFLING";
  try {
    state = await api("/api/new", { method: "POST", body: "{}" });
    render();
  } catch (error) {
    showError(error.message);
  } finally {
    busy = false;
    $("newGameButton").disabled = false;
  }
}

async function submitAction() {
  if (busy) return;
  busy = true;
  $("submitAction").disabled = true;
  $("turnBadge").textContent = "RESOLVING";
  try {
    state = await api("/api/action", {
      method: "POST",
      body: JSON.stringify({ indices: [...selected].sort((a, b) => a - b) }),
    });
    render();
  } catch (error) {
    showError(error.message);
  } finally {
    busy = false;
    if (state?.observation?.select) updateSubmit(state.observation.select);
  }
}

function showPreview(id) {
  $("previewImage").src = cardImageUrl(id);
  $("cardPreview").classList.remove("hidden");
  $("cardPreview").setAttribute("aria-hidden", "false");
}

function showError(message) {
  $("errorBanner").textContent = message;
  $("errorBanner").classList.remove("hidden");
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[character]);
}

$("newGameButton").addEventListener("click", startNewGame);
$("submitAction").addEventListener("click", submitAction);
$("cardPreview").addEventListener("click", () => $("cardPreview").classList.add("hidden"));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("cardPreview").classList.add("hidden");
});

async function initialize() {
  try {
    meta = await api("/api/meta");
    state = await api("/api/state");
    if (!state.started) await startNewGame();
    else render();
  } catch (error) {
    showError(error.message);
  }
}

initialize();
