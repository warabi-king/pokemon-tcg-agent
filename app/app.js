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

const ENERGY_TYPES = ["無色", "草", "炎", "水", "雷", "超", "闘", "悪", "鋼", "ドラゴン", "レインボー", "ロケット団"];
const CONDITION_FIELDS = [
  ["poisoned", "どく"], ["burned", "やけど"], ["asleep", "ねむり"],
  ["paralyzed", "マヒ"], ["confused", "こんらん"]
];

let state = null;
let meta = { cards: {}, attacks: {} };
let selected = new Set();
let busy = false;
const watchMode = window.location.pathname === "/watch";
const duelMode = window.location.pathname === "/duel";
let autoPlaying = false;
let autoPlayTimer = null;
let duelPollTimer = null;

const $ = (id) => document.getElementById(id);
const padCardId = (id) => String(id).padStart(4, "0");
const cardId = (card) => card && (card.id ?? card.cardId);
const cardMeta = (id) => meta.cards[String(id)] || { id, name: `カード #${id}` };

function cardImageUrl(id) {
  return `/cards/${padCardId(id)}.jpg`;
}

function addActionIndex(map, key, index) {
  if (!map.has(key)) map.set(key, []);
  map.get(key).push(index);
}

function buildCardActionMap(observation) {
  const map = new Map();
  const selection = observation?.select;
  if (!state?.humanTurn || selection?.context !== 0) return map;

  selection.option.forEach((option, index) => {
    if ([7, 8, 9].includes(option.type) && option.index != null) {
      addActionIndex(map, `hand:${option.index}`, index);
    } else if ([10, 11].includes(option.type) && [4, 5].includes(option.area)) {
      addActionIndex(map, `field:${option.area}:${option.index}`, index);
    } else if ([12, 13].includes(option.type)) {
      addActionIndex(map, "field:4:0", index);
    }
  });
  return map;
}

function mappedActionIndices(observation) {
  return new Set([...buildCardActionMap(observation).values()].flat());
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

  const actionIndices = options.actionIndices || [];
  const inPlayDetails = Boolean(options.inPlayDetails);
  if (actionIndices.length || inPlayDetails) {
    element.classList.add(inPlayDetails ? "in-play-card" : "actionable");
    if (actionIndices.length) element.classList.add("actionable");
    element.dataset.actionIndices = actionIndices.join(",");
    element.setAttribute("role", "button");
    element.setAttribute("tabindex", "0");
    element.setAttribute("aria-label", inPlayDetails ? `${cardMeta(id).name}の詳細と行動を表示` : `${cardMeta(id).name}の行動を表示`);
    const openActions = () => {
      if (inPlayDetails) {
        openPokemonPopup(card, actionIndices, options.locationLabel, options.observation, options.playerState, options.isActive, options.ownerLabel);
      } else {
        openActionPopup(card, actionIndices, options.locationLabel, options.observation);
      }
    };
    element.addEventListener("click", openActions);
    element.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openActions();
      }
    });
    const actionBadge = document.createElement("span");
    actionBadge.className = actionIndices.length ? "action-badge" : "detail-badge";
    actionBadge.textContent = actionIndices.length ? (actionIndices.length > 1 ? `ACTION ${actionIndices.length}` : "ACTION") : "INFO";
    element.appendChild(actionBadge);
  } else {
    element.addEventListener("click", () => showPreview(id));
  }

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

function renderSlots(target, cards, count, optionsForIndex = () => ({})) {
  target.replaceChildren();
  for (let i = 0; i < count; i += 1) {
    const card = cards?.[i] || null;
    target.appendChild(cardElement(card, card ? optionsForIndex(i, card) : {}));
  }
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
  const actionMap = buildCardActionMap(observation);

  renderSlots($("opponentBench"), opponent.bench, opponent.benchMax || 5, (index) => ({
    inPlayDetails: true,
    locationLabel: `ベンチ${index + 1}`,
    observation,
    playerState: opponent,
    isActive: false,
    ownerLabel: "対戦AI",
  }));
  renderSlots($("humanBench"), human.bench, human.benchMax || 5, (index) => ({
    actionIndices: actionMap.get(`field:5:${index}`) || [],
    inPlayDetails: true,
    locationLabel: `ベンチ${index + 1}`,
    observation,
    playerState: human,
    isActive: false,
    ownerLabel: "あなた",
  }));
  renderSlots($("opponentActive"), opponent.active, 1, () => ({
    inPlayDetails: true,
    locationLabel: "バトル場",
    observation,
    playerState: opponent,
    isActive: true,
    ownerLabel: "対戦AI",
  }));
  renderSlots($("humanActive"), human.active, 1, (index) => ({
    actionIndices: actionMap.get(`field:4:${index}`) || [],
    inPlayDetails: true,
    locationLabel: "バトル場",
    observation,
    playerState: human,
    isActive: true,
    ownerLabel: "あなた",
  }));

  renderPile($("opponentDeck"), "山札", opponent.deckCount, null, true);
  renderPile($("humanDeck"), "山札", human.deckCount, null, true);
  renderPile($("opponentDiscard"), "トラッシュ", opponent.discard?.length || 0, opponent.discard?.at(-1));
  renderPile($("humanDiscard"), "トラッシュ", human.discard?.length || 0, human.discard?.at(-1));

  $("opponentStats").innerHTML = `<span>HAND ${opponent.handCount}</span><span>PRIZE ${opponent.prize?.length || 0}</span>`;
  $("humanStats").innerHTML = `<span>HAND ${human.handCount}</span><span>PRIZE ${human.prize?.length || 0}</span>`;

  const hand = $("humanHand");
  hand.replaceChildren();
  if (!human.hand?.length) hand.innerHTML = '<div class="empty-message">手札はありません</div>';
  else human.hand.forEach((card, index) => hand.appendChild(cardElement(card, {
    actionIndices: actionMap.get(`hand:${index}`) || [],
    locationLabel: "手札",
    observation,
  })));

  $("opponentHandPanel").classList.toggle("hidden", !watchMode);
  if (watchMode) {
    const opponentHand = $("opponentHand");
    opponentHand.replaceChildren();
    if (opponent.hand?.length) {
      opponent.hand.forEach((card) => opponentHand.appendChild(cardElement(card)));
    } else if (opponent.handCount) {
      for (let index = 0; index < opponent.handCount; index += 1) {
        opponentHand.appendChild(cardElement({}, { back: true }));
      }
    } else {
      opponentHand.innerHTML = '<div class="empty-message">手札はありません</div>';
    }
  }

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

function findInPlayTarget(option, observation) {
  if (option.type !== 8 || !observation?.current) return null;
  const playerIndex = option.playerIndex ?? observation.current.yourIndex;
  const player = observation.current.players[playerIndex];
  if (option.inPlayArea === 4) {
    return { card: player?.active?.[option.inPlayIndex], area: "バトル場" };
  }
  if (option.inPlayArea === 5) {
    return { card: player?.bench?.[option.inPlayIndex], area: `ベンチ${option.inPlayIndex + 1}` };
  }
  return null;
}

function optionDescription(option, card, observation) {
  if (option.type === 0) return `${option.number}`;
  if (option.type === 1) return "はい";
  if (option.type === 2) return "いいえ";
  if (option.type === 12) return "バトルポケモンをベンチといれかえる";
  if (option.type === 13) {
    const attack = meta.attacks[String(option.attackId)];
    return attack ? `${attack.name}${attack.damage ? ` — ${attack.damage}ダメージ` : ""}` : `ワザ #${option.attackId}`;
  }
  if (option.type === 14) return "自分の番を終える";
  if (option.type === 8 && card) {
    const target = findInPlayTarget(option, observation);
    if (target?.card) {
      return `${cardMeta(cardId(card)).name} → ${cardMeta(cardId(target.card)).name}（${target.area}）`;
    }
  }
  if (card) return cardMeta(cardId(card)).name;
  return OPTION_TYPES[option.type] || `選択肢 ${option.type}`;
}

function createActionOptionButton(index, option, observation, onClick) {
  const card = findCardForOption(option, observation);
  const button = document.createElement("button");
  button.type = "button";
  button.className = `action-option${card ? "" : " no-card"}`;
  button.dataset.index = index;
  const heading = optionDescription(option, card, observation);
  button.innerHTML = `<span class="option-index">${index + 1}</span><span class="option-copy"><strong>${escapeHtml(heading)}</strong><small>${escapeHtml(OPTION_TYPES[option.type] || "合法手")}</small></span>`;
  if (card) button.appendChild(cardElement(card, { mini: true }));
  button.addEventListener("click", onClick);
  return button;
}

function renderPopupActions(actionIndices, observation) {
  const selection = observation?.select;
  const options = $("actionPopupOptions");
  options.replaceChildren();
  actionIndices.forEach((index) => {
    const button = createActionOptionButton(index, selection.option[index], observation, () => executePopupAction(index, selection));
    button.classList.toggle("selected", selected.has(index));
    options.appendChild(button);
  });
  $("popupActionsHeading").classList.toggle("hidden", !actionIndices.length);
  $("popupNoActions").classList.toggle("hidden", Boolean(actionIndices.length));
}

async function executePopupAction(index, selection) {
  if (busy) return;
  if (!selected.has(index)) toggleSelection(index, selection);
  closeActionPopup();
  await submitAction();
}

function openActionPopup(card, actionIndices, locationLabel, observation) {
  if (!observation?.select) return;
  const id = cardId(card);
  $("actionPopupEyebrow").textContent = "CARD ACTIONS";
  $("actionPopupTitle").textContent = cardMeta(id).name;
  $("actionPopupLocation").textContent = `${locationLabel || "カード"}で選べる行動`;
  $("pokemonDetails").classList.add("hidden");

  const popupCard = $("actionPopupCard");
  popupCard.replaceChildren(cardElement(card));
  renderPopupActions(actionIndices, observation);

  $("actionPopup").classList.remove("hidden");
  $("actionPopup").setAttribute("aria-hidden", "false");
}

function renderAttachmentList(target, cards, fallbackEnergyTypes = []) {
  target.replaceChildren();
  if (cards?.length) {
    cards.forEach((attachedCard) => {
      const row = document.createElement("div");
      row.className = "attachment-item";
      row.appendChild(cardElement(attachedCard, { mini: true }));
      const label = document.createElement("span");
      label.textContent = cardMeta(cardId(attachedCard)).name;
      row.appendChild(label);
      target.appendChild(row);
    });
    return;
  }
  if (fallbackEnergyTypes.length) {
    fallbackEnergyTypes.forEach((type) => {
      const chip = document.createElement("span");
      chip.className = "energy-chip";
      chip.textContent = ENERGY_TYPES[type] || `タイプ${type}`;
      target.appendChild(chip);
    });
    return;
  }
  target.innerHTML = '<span class="detail-empty">なし</span>';
}

function renderPokemonDetails(card, playerState, isActive) {
  $("pokemonDetails").classList.remove("hidden");
  $("pokemonHp").innerHTML = `<span>現在HP</span><strong>${card.hp}/${card.maxHp}</strong>`;
  renderAttachmentList($("pokemonEnergy"), card.energyCards, card.energies || []);
  renderAttachmentList($("pokemonTools"), card.tools);

  const conditionTarget = $("pokemonConditions");
  conditionTarget.replaceChildren();
  const conditions = isActive ? CONDITION_FIELDS.filter(([field]) => playerState?.[field]).map(([, label]) => label) : [];
  if (!conditions.length) conditionTarget.innerHTML = '<span class="condition-chip normal">なし</span>';
  else conditions.forEach((condition) => {
    const chip = document.createElement("span");
    chip.className = "condition-chip";
    chip.textContent = condition;
    conditionTarget.appendChild(chip);
  });
}

function openPokemonPopup(card, actionIndices, locationLabel, observation, playerState, isActive, ownerLabel) {
  const id = cardId(card);
  $("actionPopupEyebrow").textContent = "POKÉMON DETAILS";
  $("actionPopupTitle").textContent = cardMeta(id).name;
  $("actionPopupLocation").textContent = `${ownerLabel || "プレイヤー"}・${locationLabel || "場"}`;
  $("actionPopupCard").replaceChildren(cardElement(card));
  renderPokemonDetails(card, playerState, isActive);
  renderPopupActions(actionIndices, observation);
  $("actionPopup").classList.remove("hidden");
  $("actionPopup").setAttribute("aria-hidden", "false");
}

function closeActionPopup() {
  $("actionPopup").classList.add("hidden");
  $("actionPopup").setAttribute("aria-hidden", "true");
}

function renderActions(observation) {
  selected = new Set();
  closeActionPopup();
  const selection = observation?.select;
  const list = $("actionList");
  list.replaceChildren();
  $("selectedAction").classList.add("hidden");
  if (watchMode) {
    $("contextCard").replaceChildren();
    $("actionTitle").textContent = state?.finished
      ? "対戦終了"
      : `Player ${Number(state?.nextPlayer) + 1} の行動待ち`;
    $("selectionRule").textContent = `ACTION ${state?.step ?? 0}`;
    $("submitAction").disabled = busy || state?.finished || Boolean(state?.error);
    $("submitAction").textContent = state?.finished ? "対戦終了" : "次の1行動";
    const latest = state?.actions?.at(-1);
    list.innerHTML = latest
      ? `<div class="empty-message">直前: Player ${latest.player + 1} / 選択 index [${latest.indices.join(", ")}]</div>`
      : '<div class="empty-message">ボタンを押すとエージェントが1行動します。</div>';
    return;
  }
  if (!selection || !state?.humanTurn) {
    if (duelMode) {
      $("actionTitle").textContent = state?.finished
        ? "対戦終了"
        : !state?.started
          ? "ルーム作成待ち"
          : !state?.opponentJoined ? "Player 2参加待ち" : "対戦相手の選択待ち";
      $("selectionRule").textContent = "";
      $("submitAction").disabled = true;
      list.innerHTML = `<div class="empty-message">${state?.finished ? resultMessage() : "少々お待ちください"}</div>`;
      return;
    }
    $("actionTitle").textContent = state?.finished ? "対戦終了" : "AIが選択中";
    $("selectionRule").textContent = "";
    $("submitAction").disabled = true;
    list.innerHTML = `<div class="empty-message">${state?.finished ? resultMessage() : "少々お待ちください"}</div>`;
    return;
  }

  $("actionTitle").textContent = CONTEXTS[selection.context] || "行動を選択";
  $("selectionRule").textContent = `${selection.minCount}〜${selection.maxCount}個`;
  const cardActionIndices = mappedActionIndices(observation);
  if (cardActionIndices.size) {
    const hint = document.createElement("div");
    hint.className = "card-action-hint";
    hint.innerHTML = '<strong>盤面から行動を選択</strong><span>「ACTION」が付いたカードを開き、行動を押すとすぐに実行されます。</span>';
    list.appendChild(hint);
  }
  selection.option.forEach((option, index) => {
    if (cardActionIndices.has(index)) return;
    list.appendChild(createActionOptionButton(index, option, observation, () => toggleSelection(index, selection)));
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
  updateSelectedAction(selection, observation);
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
  document.querySelectorAll(".card.actionable").forEach((card) => {
    const indices = (card.dataset.actionIndices || "").split(",").filter(Boolean).map(Number);
    card.classList.toggle("choice-selected", indices.some((optionIndex) => selected.has(optionIndex)));
  });
  updateSelectedAction(selection, state?.observation);
  updateSubmit(selection);
}

function updateSelectedAction(selection, observation) {
  const target = $("selectedAction");
  if (!selected.size) {
    target.classList.add("hidden");
    target.replaceChildren();
    return;
  }
  const labels = [...selected].map((index) => {
    const option = selection.option[index];
    return optionDescription(option, findCardForOption(option, observation), observation);
  });
  target.innerHTML = `<span>選択中</span><strong>${labels.map(escapeHtml).join(" / ")}</strong>`;
  target.classList.remove("hidden");
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
  if (duelMode) return reward > 0 ? "YOU WIN" : reward < 0 ? "YOU LOSE" : "DRAW";
  if (watchMode && reward > 0) return "PLAYER 1 WIN";
  if (watchMode && reward < 0) return "PLAYER 2 WIN";
  if (watchMode) return "DRAW";
  if (reward > 0) return "YOU WIN — 勝利しました";
  if (reward < 0) return "YOU LOSE — AIの勝利です";
  return "DRAW — 引き分けです";
}

function render() {
  closeActionPopup();
  const observation = state?.observation;
  if (watchMode && state?.agentNames?.length === 2) {
    document.querySelector(".human-zone .zone-heading strong").textContent = `Player 1 / ${state.agentNames[0]}`;
    document.querySelector(".opponent-zone .zone-heading strong").textContent = `Player 2 / ${state.agentNames[1]}`;
  }
  if (duelMode && state?.started) {
    const role = Number(state.role);
    document.querySelector(".human-zone .zone-heading strong").textContent = `あなた / Player ${role + 1}`;
    document.querySelector(".opponent-zone .zone-heading strong").textContent = `対戦相手 / Player ${2 - role}`;
    $("duelLobby").classList.remove("hidden");
    $("duelLobbyTitle").textContent = state.opponentJoined ? "対戦中" : "Player 2の参加を待っています";
    $("duelLobbyMessage").textContent = state.opponentJoined
      ? `ルームコード ${state.roomId}`
      : state.cloudMode
        ? `ルームコード ${state.roomId} をPlayer 2へ伝えてください。`
        : "下の招待リンクをPlayer 2へ共有してください。";
    if (state.invitePath) {
      $("inviteLink").value = `${window.location.origin}${state.invitePath}`;
      $("invitePanel").classList.remove("hidden");
    }
    $("newGameButton").classList.toggle("hidden", state.cloudMode || role !== 0);
    for (const select of [$("agentASelect"), $("agentBSelect")]) {
      select.closest("label").classList.add("hidden");
    }
  } else if (duelMode) {
    $("duelLobby").classList.remove("hidden");
  }
  renderBoard(observation);
  renderActions(observation);
  renderEvents(state?.events);
  $("stepLabel").textContent = `STEP ${state?.step ?? 0}`;

  const badge = $("turnBadge");
  badge.classList.toggle("waiting", watchMode ? false : !state?.humanTurn);
  badge.textContent = state?.finished
    ? resultMessage()
    : watchMode
      ? `PLAYER ${Number(state?.nextPlayer) + 1} TO MOVE`
      : duelMode && !state?.started ? "LOBBY"
      : duelMode && !state?.opponentJoined ? "WAITING FOR PLAYER 2"
      : state?.humanTurn ? "YOUR MOVE" : duelMode ? "OPPONENT'S MOVE" : "AI THINKING";
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
    stopAutoPlay();
    const agents = watchMode
      ? { agentA: $("agentASelect").value, agentB: $("agentBSelect").value }
      : duelMode
        ? { deckA: $("agentASelect").value, deckB: $("agentBSelect").value }
        : { agent: $("agentBSelect").value };
    const path = watchMode ? "/api/watch/new" : duelMode ? "/api/duel/create" : "/api/new";
    state = await api(path, {
      method: "POST",
      body: JSON.stringify(agents),
    });
    render();
  } catch (error) {
    showError(error.message);
  } finally {
    busy = false;
    $("newGameButton").disabled = false;
    if (watchMode && state?.started) renderActions(state.observation);
  }
}

async function submitAction() {
  if (busy) return;
  busy = true;
  $("submitAction").disabled = true;
  $("turnBadge").textContent = "RESOLVING";
  try {
    const path = watchMode ? "/api/watch/step" : duelMode ? "/api/duel/action" : "/api/action";
    state = await api(path, {
      method: "POST",
      body: watchMode
        ? "{}"
        : JSON.stringify({ indices: [...selected].sort((a, b) => a - b), step: state?.step }),
    });
    render();
  } catch (error) {
    showError(error.message);
  } finally {
    busy = false;
    if (!watchMode && state?.observation?.select) updateSubmit(state.observation.select);
    if (watchMode && state?.started) renderActions(state.observation);
    if (watchMode && autoPlaying && !state?.finished && !state?.error) {
      autoPlayTimer = window.setTimeout(submitAction, 650);
    } else if (watchMode && (state?.finished || state?.error)) {
      stopAutoPlay();
    }
  }
}

function stopAutoPlay() {
  autoPlaying = false;
  if (autoPlayTimer != null) window.clearTimeout(autoPlayTimer);
  autoPlayTimer = null;
  $("autoPlayButton").textContent = "連続再生";
}

function toggleAutoPlay() {
  if (!watchMode || state?.finished || state?.error) return;
  if (autoPlaying) {
    stopAutoPlay();
    return;
  }
  autoPlaying = true;
  $("autoPlayButton").textContent = "一時停止";
  submitAction();
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
$("autoPlayButton").addEventListener("click", toggleAutoPlay);
$("cardPreview").addEventListener("click", () => $("cardPreview").classList.add("hidden"));
$("closeActionPopup").addEventListener("click", closeActionPopup);
$("actionPopup").addEventListener("click", (event) => {
  if (event.target === $("actionPopup")) closeActionPopup();
});
$("copyInviteButton").addEventListener("click", async () => {
  const link = $("inviteLink").value;
  if (!link) return;
  try {
    await navigator.clipboard.writeText(link);
    $("copyInviteButton").textContent = "コピーしました";
    window.setTimeout(() => {
      $("copyInviteButton").textContent = "招待リンクをコピー";
    }, 1600);
  } catch (_error) {
    $("inviteLink").select();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    $("cardPreview").classList.add("hidden");
    closeActionPopup();
  }
});

async function initialize() {
  try {
    if (watchMode) {
      document.title = "PokeTCG Agent Match Viewer";
      document.querySelector("h1").textContent = "PokeTCG Agent Match Viewer";
      $("autoPlayButton").classList.remove("hidden");
      document.querySelector(".human-zone .zone-heading strong").textContent = "Player 1";
      document.querySelector(".opponent-zone .zone-heading strong").textContent = "Player 2";
      $("opponentHandPanel").querySelector(".section-label").textContent = "PLAYER 2 HAND";
      $("humanHand").parentElement.querySelector(".section-label").textContent = "PLAYER 1 HAND";
    }
    if (duelMode) {
      document.title = "PokeTCG 1対1 Online Battle";
      document.querySelector("h1").textContent = "PokeTCG 1対1 Online Battle";
      document.querySelector(".eyebrow").textContent = "HUMAN VS HUMAN / CABT";
      $("newGameButton").textContent = "ルーム作成";
    }
    meta = await api("/api/meta");
    for (const select of [$("agentASelect"), $("agentBSelect")]) {
      for (const agentName of meta.agents) {
        select.add(new Option(agentName, agentName));
      }
    }
    $("agentASelect").closest("label").classList.toggle("hidden", !watchMode && !duelMode);
    const statePath = watchMode ? "/api/watch/state" : duelMode ? "/api/duel/state" : "/api/state";
    state = await api(statePath);
    if (!state.started && !duelMode) await startNewGame();
    else render();
    if (duelMode) duelPollTimer = window.setInterval(pollDuelState, 1000);
  } catch (error) {
    showError(error.message);
  }
}

async function pollDuelState() {
  if (!duelMode || busy) return;
  try {
    const nextState = await api("/api/duel/state");
    const changed = nextState.started !== state?.started
      || nextState.step !== state?.step
      || nextState.opponentJoined !== state?.opponentJoined
      || nextState.finished !== state?.finished;
    state = nextState;
    if (changed) render();
  } catch (error) {
    showError(error.message);
  }
}

initialize();
