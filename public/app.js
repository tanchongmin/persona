const personasEl = document.querySelector("#personas");
const addPersonaEl = document.querySelector("#addPersona");
const scenarioEl = document.querySelector("#scenario");
const generatePersonasEl = document.querySelector("#generatePersonas");
const userNameEl = document.querySelector("#userName");
const userPersonaEl = document.querySelector("#userPersona");
const summaryEl = document.querySelector("#summary");
const narrationToggleEl = document.querySelector("#narrationToggle");
const modelEls = Array.from(document.querySelectorAll("input[name='model']"));
const messagesEl = document.querySelector("#messages");
const setupPanelEl = document.querySelector(".setup-panel");
const setupToggleEl = document.querySelector("#setupToggle");
const turnControlsEl = document.querySelector(".turn-controls");
const form = document.querySelector("#turnForm");
const inputEl = document.querySelector("#turnInput");
const imageInputEl = document.querySelector("#imageInput");
const imagePreviewEl = document.querySelector("#imagePreview");
const sendEl = document.querySelector("#send");
const takeActionEl = document.querySelector("#takeAction");
const fastForwardEl = document.querySelector("#fastForward");
const backtrackEl = document.querySelector("#backtrack");
const imageModalEl = document.querySelector("#imageModal");
const imageModalImgEl = document.querySelector("#imageModalImg");
const imageModalCloseEl = document.querySelector("#imageModalClose");
const tokenUsageBarEl = document.querySelector("#tokenUsageBar");
const tokenModalEl = document.querySelector("#tokenModal");
const tokenModalTitleEl = document.querySelector("#tokenModalTitle");
const tokenModalCloseEl = document.querySelector("#tokenModalClose");
const tokenInputTextEl = document.querySelector("#tokenInputText");
const tokenOutputTextEl = document.querySelector("#tokenOutputText");

const state = {
  messages: [],
  recentMessages: [],
  turnSnapshots: [],
  personas: [],
  pendingDeltas: new Map(),
  streamFinals: new Map(),
  suppressedStreamIds: new Set(),
  streamQueue: [],
  activeStreamId: null,
  streamTimer: null,
  pendingImages: [],
  tokenReports: [],
  serverTurnDone: false,
  turnInProgress: false,
  personaGenerationInProgress: false
};

const STREAM_FLUSH_INTERVAL_MS = 45;
const STREAM_CHARS_PER_TICK = 3;
const RECENT_CONTEXT_TURN_LIMIT = 10;
const MAX_PERSONAS = 10;
const MAX_IMAGES_PER_TURN = 4;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;

const defaults = {
  personas: [{
    id: crypto.randomUUID(),
    name: "Mara",
    gender: "Female",
    description: "A warm but direct companion who notices emotional shifts quickly. She speaks naturally, has clear boundaries, and responds like a real person rather than an assistant."
  }],
  scenario: "A quiet evening conversation at home where a small honest choice can shift the relationship.",
  userName: "John",
  userPersona: "A happy and optimistic individual",
  summary: "John and Mara are at home at the start of a quiet evening conversation. They know each other well enough for the exchange to feel comfortable, but nothing has happened yet."
};

state.personas = defaults.personas.map((persona) => ({ ...persona }));
scenarioEl.value = defaults.scenario;
userNameEl.value = defaults.userName;
userPersonaEl.value = defaults.userPersona;
summaryEl.value = defaults.summary;
renderPersonas();
render();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.turnInProgress) return;
  const content = inputEl.value.trim();
  if (!content && state.pendingImages.length === 0) return;
  inputEl.value = "";
  const imageAttachments = consumePendingImages();
  await submitTurn("chat", content, { imageAttachments });
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  event.preventDefault();
  if (state.turnInProgress) return;
  form.requestSubmit();
});

takeActionEl.addEventListener("click", () => {
  if (state.turnInProgress) return;
  const content = inputEl.value.trim();
  if (!content && state.pendingImages.length === 0) return;
  inputEl.value = "";
  const imageAttachments = consumePendingImages();
  submitTurn("action", content, { imageAttachments });
});

imageInputEl.addEventListener("change", async () => {
  try {
    await addPendingImages(Array.from(imageInputEl.files || []));
  } catch (error) {
    pushSystemError(error.message || "Could not attach image.");
  }
  imageInputEl.value = "";
});

fastForwardEl.addEventListener("click", () => {
  fastForwardVisibleStream();
});

backtrackEl.addEventListener("click", () => {
  backtrackTurn();
});

imageModalCloseEl.addEventListener("click", () => {
  closeImageModal();
});

imageModalEl.addEventListener("click", (event) => {
  if (event.target === imageModalEl) closeImageModal();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !imageModalEl.hidden) closeImageModal();
  if (event.key === "Escape" && !tokenModalEl.hidden) closeTokenModal();
});

tokenModalCloseEl.addEventListener("click", () => {
  closeTokenModal();
});

tokenModalEl.addEventListener("click", (event) => {
  if (event.target === tokenModalEl) closeTokenModal();
});

setupToggleEl.addEventListener("click", () => {
  const isCollapsed = setupPanelEl.classList.toggle("collapsed");
  setupToggleEl.setAttribute("aria-expanded", String(!isCollapsed));
  scrollMessagesToBottom();
});

narrationToggleEl.addEventListener("change", () => {
  render();
});

addPersonaEl.addEventListener("click", () => {
  if (state.personas.length >= MAX_PERSONAS) return;
  state.personas.push(createCustomPersona(state.personas.length));
  renderPersonas();
});

generatePersonasEl.addEventListener("click", () => {
  generatePersonasFromScenario();
});

function createCustomPersona(index) {
  return {
    id: crypto.randomUUID(),
    name: `Persona ${index + 1}`,
    gender: "Female",
    description: ""
  };
}

function createPersonaFromGenerated(persona, index) {
  return {
    id: crypto.randomUUID(),
    name: persona.name || `Persona ${index + 1}`,
    gender: persona.gender === "Male" ? "Male" : "Female",
    description: persona.description || ""
  };
}

async function submitTurn(type, content, extraBody = {}) {
  if (state.turnInProgress) return;
  pushTurnSnapshot();
  state.serverTurnDone = false;
  state.turnInProgress = true;
  setBusy(true);
  try {
    const response = await fetch("/api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type,
        content,
        imageAttachments: extraBody.imageAttachments || [],
        personas: currentPersonas(),
        scenario: scenarioEl.value,
        userName: userNameEl.value,
        userPersona: userPersonaEl.value,
        summary: summaryEl.value,
        model: selectedModel(),
        messages: state.recentMessages
      })
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || "Request failed.");
    }

    await readTurnStream(response);
  } catch (error) {
    state.messages.push({
      id: crypto.randomUUID(),
      role: "system",
      kind: "error",
      content: error.message,
      createdAt: new Date().toISOString()
    });
    render();
    state.turnInProgress = false;
    setBusy(false);
  } finally {
    unlockComposerIfReady();
  }
}

async function addPendingImages(files) {
  const remainingSlots = MAX_IMAGES_PER_TURN - state.pendingImages.length;
  if (remainingSlots <= 0) {
    pushSystemError(`Attach up to ${MAX_IMAGES_PER_TURN} images per turn.`);
    return;
  }

  const accepted = files
    .filter((file) => {
      if (!file.type.startsWith("image/")) {
        pushSystemError(`${file.name} is not an image.`);
        return false;
      }
      if (file.size > MAX_IMAGE_BYTES) {
        pushSystemError(`${file.name} is larger than 5 MB.`);
        return false;
      }
      return true;
    })
    .slice(0, remainingSlots);

  const attachments = await Promise.all(accepted.map(readImageFile));
  state.pendingImages.push(...attachments);
  renderImagePreview();
}

function readImageFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      resolve({
        id: crypto.randomUUID(),
        name: file.name,
        type: file.type,
        size: file.size,
        dataUrl: String(reader.result || "")
      });
    });
    reader.addEventListener("error", () => reject(new Error(`Could not read ${file.name}.`)));
    reader.readAsDataURL(file);
  });
}

function consumePendingImages() {
  const images = state.pendingImages;
  state.pendingImages = [];
  renderImagePreview();
  return images;
}

function renderImagePreview() {
  imagePreviewEl.innerHTML = "";
  imagePreviewEl.hidden = state.pendingImages.length === 0;

  for (const image of state.pendingImages) {
    const item = document.createElement("div");
    item.className = "image-preview-item";

    const img = document.createElement("img");
    img.src = image.dataUrl;
    img.alt = image.name || "Attached image";

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "image-remove";
    remove.textContent = "x";
    remove.title = "Remove image";
    remove.setAttribute("aria-label", `Remove ${image.name || "image"}`);
    remove.addEventListener("click", () => {
      state.pendingImages = state.pendingImages.filter((itemImage) => itemImage.id !== image.id);
      renderImagePreview();
    });

    item.append(img, remove);
    imagePreviewEl.append(item);
  }
}

async function generatePersonasFromScenario() {
  if (state.personaGenerationInProgress || state.turnInProgress) return;
  const scenario = scenarioEl.value.trim();
  if (!scenario) {
    pushSystemError("Type a scenario before generating personas.");
    return;
  }

  state.personaGenerationInProgress = true;
  setBusyControls();
  try {
    const response = await fetch("/api/generate-personas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scenario,
        userName: userNameEl.value,
        userPersona: userPersonaEl.value,
        model: selectedModel()
      })
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || "Persona generation failed.");
    if (!Array.isArray(payload.personas) || payload.personas.length === 0) {
      throw new Error("Persona generation returned no personas.");
    }

    state.personas = payload.personas.slice(0, MAX_PERSONAS).map(createPersonaFromGenerated);
    summaryEl.value = payload.summary || "";
    state.recentMessages = [];
    state.messages = [];
    state.turnSnapshots = [];
    state.tokenReports = [];
    renderPersonas();
    render();
    renderTokenUsageBar();
  } catch (error) {
    pushSystemError(error.message);
  } finally {
    state.personaGenerationInProgress = false;
    setBusyControls();
  }
}

function currentPersonas() {
  return state.personas.map((persona) => ({
    id: persona.id,
    name: persona.nameEl?.value.trim() || persona.name || "Persona",
    gender: persona.genderEl?.value || persona.gender || "",
    description: persona.descriptionEl?.value.trim() || persona.description || ""
  }));
}

function selectedModel() {
  return modelEls.find((input) => input.checked)?.value || "gemma-4";
}

function pushTurnSnapshot() {
  state.turnSnapshots.push({
    messages: cloneMessages(state.messages),
    recentMessages: cloneMessages(state.recentMessages),
    tokenReports: cloneMessages(state.tokenReports),
    personas: snapshotPersonas(),
    summary: summaryEl.value
  });
  updateTurnControls();
}

function cloneMessages(messages) {
  return JSON.parse(JSON.stringify(Array.isArray(messages) ? messages : []));
}

function snapshotPersonas() {
  return state.personas.map((persona) => ({
    id: persona.id,
    name: persona.nameEl?.value.trim() || persona.name || "Persona",
    gender: persona.genderEl?.value || persona.gender || "",
    description: persona.descriptionEl?.value.trim() || persona.description || ""
  }));
}

function backtrackTurn() {
  if (state.turnInProgress || state.personaGenerationInProgress || state.turnSnapshots.length === 0) return;
  const snapshot = state.turnSnapshots.pop();
  restoreTurnSnapshot(snapshot);
}

function restoreTurnSnapshot(snapshot) {
  clearActiveStreamTimer();
  state.messages = cloneMessages(snapshot.messages);
  state.recentMessages = cloneMessages(snapshot.recentMessages);
  state.tokenReports = cloneMessages(snapshot.tokenReports);
  state.personas = snapshot.personas.map((persona) => ({ ...persona }));
  summaryEl.value = snapshot.summary;
  state.pendingDeltas.clear();
  state.streamFinals.clear();
  state.suppressedStreamIds.clear();
  state.streamQueue = [];
  state.activeStreamId = null;
  state.serverTurnDone = false;
  renderPersonas();
  render();
  renderTokenUsageBar();
  setBusyControls();
}

function renderPersonas() {
  personasEl.innerHTML = "";
  state.personas.forEach((persona, index) => {
    const item = document.createElement("section");
    item.className = "persona-editor";

    const header = document.createElement("div");
    header.className = "persona-editor-header";

    const title = document.createElement("span");
    title.textContent = `Persona ${index + 1}`;

    const meta = document.createElement("div");
    meta.className = "persona-editor-meta";
    const genderBadge = document.createElement("span");
    genderBadge.className = "gender-badge";
    genderBadge.textContent = persona.gender || "Gender";
    meta.append(title, genderBadge);

    const deleteButton = document.createElement("button");
    deleteButton.className = "secondary icon-button";
    deleteButton.type = "button";
    deleteButton.textContent = "x";
    deleteButton.title = "Delete persona";
    deleteButton.setAttribute("aria-label", "Delete persona");
    deleteButton.addEventListener("click", () => {
      state.personas = state.personas.filter((itemPersona) => itemPersona.id !== persona.id);
      renderPersonas();
      render();
    });

    header.append(meta, deleteButton);

    const grid = document.createElement("div");
    grid.className = "persona-grid";

    const nameField = createField("Name", "input", persona.name, "Persona name");
    const genderField = createGenderField(persona.gender);
    const descriptionField = createField(
      "Description",
      "textarea",
      persona.description,
      "Describe who this persona is, how they speak, boundaries, preferences, and goals."
    );
    descriptionField.field.classList.add("persona-description-field");
    descriptionField.input.rows = 4;

    persona.nameEl = nameField.input;
    persona.genderEl = genderField.input;
    persona.descriptionEl = descriptionField.input;

    const syncPersona = () => {
      persona.name = persona.nameEl.value;
      persona.gender = persona.genderEl.value;
      persona.description = persona.descriptionEl.value;
      genderBadge.textContent = persona.gender || "Gender";
    };
    persona.nameEl.addEventListener("input", syncPersona);
    persona.genderEl.addEventListener("change", syncPersona);
    persona.descriptionEl.addEventListener("input", syncPersona);

    grid.append(nameField.field, genderField.field, descriptionField.field);
    item.append(header, grid);
    personasEl.append(item);
  });

  setBusyControls();
}

function createField(labelText, elementType, value, placeholder) {
  const field = document.createElement("label");
  field.className = "field";

  const label = document.createElement("span");
  label.textContent = labelText;

  const input = elementType === "textarea" ? document.createElement("textarea") : document.createElement("input");
  if (elementType !== "textarea") input.type = "text";
  input.value = value || "";
  input.placeholder = placeholder;

  field.append(label, input);
  return { field, input };
}

function createGenderField(value) {
  const field = document.createElement("label");
  field.className = "field";

  const label = document.createElement("span");
  label.textContent = "Gender";

  const input = document.createElement("select");
  for (const gender of ["Female", "Male"]) {
    const option = document.createElement("option");
    option.value = gender;
    option.textContent = gender;
    input.append(option);
  }
  input.value = value === "Male" ? "Male" : "Female";

  field.append(label, input);
  return { field, input };
}

async function readTurnStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\n\n/);
    buffer = events.pop() || "";

    for (const eventText of events) {
      handleSseEvent(eventText);
    }
  }

  if (buffer.trim()) handleSseEvent(buffer);
}

function handleSseEvent(eventText) {
  const lines = eventText.split(/\r?\n/);
  const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
  const data = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("\n");

  if (!event || !data) return;
  const payload = JSON.parse(data);

  if (event === "message") {
    upsertMessage(payload);
    if (shouldSuppressStreamMessage(payload)) {
      state.suppressedStreamIds.add(payload.id);
    } else if (shouldQueueStreamMessage(payload)) {
      queueStreamMessage(payload.id);
    }
    render();
    return;
  }

  if (event === "delta") {
    if (state.suppressedStreamIds.has(payload.id)) {
      appendSuppressedMessageDelta(payload.id, payload.delta);
      return;
    }
    queueMessageDelta(payload.id, payload.delta);
    return;
  }

  if (event === "messageDone") {
    if (state.suppressedStreamIds.has(payload.id)) {
      finishSuppressedMessageStream(payload);
      return;
    }
    finishMessageStream(payload);
    return;
  }

  if (event === "summary") {
    summaryEl.value = payload.summary || summaryEl.value;
    return;
  }

  if (event === "tokenUsage") {
    upsertTokenReport(payload);
    renderTokenUsageBar();
    return;
  }

  if (event === "summaryError" || event === "error") {
    state.messages.push({
      id: crypto.randomUUID(),
      role: "system",
      kind: "error",
      content: payload.error,
      createdAt: new Date().toISOString()
    });
    render();
    return;
  }

  if (event === "done") {
    state.recentMessages = sliceLastTurns(payload.recentMessages || state.recentMessages, RECENT_CONTEXT_TURN_LIMIT);
    state.serverTurnDone = true;
    render();
    unlockComposerIfReady();
  }
}

function sliceLastTurns(messages, turnLimit) {
  if (!Array.isArray(messages) || messages.length <= turnLimit) return Array.isArray(messages) ? messages : [];

  const turnKeys = [];
  const seen = new Set();
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index] || {};
    const key = message.turnId || message.id || `index:${index}`;
    if (seen.has(key)) continue;
    seen.add(key);
    turnKeys.push(key);
    if (turnKeys.length >= turnLimit) break;
  }

  const keep = new Set(turnKeys);
  return messages.filter((message, index) => {
    const key = message?.turnId || message?.id || `index:${index}`;
    return keep.has(key);
  });
}

function upsertMessage(message) {
  const index = state.messages.findIndex((item) => item.id === message.id);
  if (index === -1) {
    state.messages.push(message);
    return;
  }
  state.messages[index] = message;
}

function queueMessageDelta(messageId, delta) {
  if (!messageId || !delta) return;
  const current = state.pendingDeltas.get(messageId) || "";
  state.pendingDeltas.set(messageId, current + delta);
  if (messageId === state.activeStreamId) ensureActiveStreamTimer();
}

function appendSuppressedMessageDelta(messageId, delta) {
  if (!messageId || !delta) return;
  const message = state.messages.find((item) => item.id === messageId);
  if (message) message.content += delta;
}

function finishSuppressedMessageStream(message) {
  if (!message?.id) return;
  upsertMessage(message);
  state.suppressedStreamIds.delete(message.id);
  render();
  unlockComposerIfReady();
}

function queueStreamMessage(messageId) {
  if (!messageId || state.pendingDeltas.has(messageId)) return;
  state.pendingDeltas.set(messageId, "");
  state.streamQueue.push(messageId);
  startNextQueuedStream();
}

function startNextQueuedStream() {
  if (state.activeStreamId) return;

  while (state.streamQueue.length > 0) {
    const messageId = state.streamQueue.shift();
    if (!state.pendingDeltas.has(messageId) && !state.streamFinals.has(messageId)) continue;
    state.activeStreamId = messageId;
    render();
    if (state.pendingDeltas.get(messageId)) {
      ensureActiveStreamTimer();
    } else {
      applyStreamFinalIfReady(messageId);
    }
    return;
  }
}

function ensureActiveStreamTimer() {
  if (state.streamTimer || !state.activeStreamId) return;
  state.streamTimer = setInterval(flushActiveMessageDelta, STREAM_FLUSH_INTERVAL_MS);
}

function flushActiveMessageDelta() {
  const messageId = state.activeStreamId;
  if (!messageId) {
    clearActiveStreamTimer();
    return;
  }

  const pending = state.pendingDeltas.get(messageId) || "";
  if (!pending) {
    clearActiveStreamTimer();
    applyStreamFinalIfReady(messageId);
    return;
  }

  const delta = pending.slice(0, STREAM_CHARS_PER_TICK);
  state.pendingDeltas.set(messageId, pending.slice(delta.length));

  const message = state.messages.find((item) => item.id === messageId);
  if (message) {
    message.content += delta;
    render();
  }
}

function finishMessageStream(message) {
  if (!message?.id) return;
  if (!state.pendingDeltas.has(message.id) && message.id !== state.activeStreamId) {
    upsertMessage(message);
    render();
    return;
  }
  state.streamFinals.set(message.id, message);
  if (message.id === state.activeStreamId && !state.pendingDeltas.get(message.id)) applyStreamFinalIfReady(message.id);
  startNextQueuedStream();
}

function applyStreamFinalIfReady(messageId) {
  const finalMessage = state.streamFinals.get(messageId);
  if (!finalMessage || state.pendingDeltas.get(messageId) || messageId !== state.activeStreamId) return;
  clearActiveStreamTimer();
  upsertMessage(finalMessage);
  state.streamFinals.delete(messageId);
  state.pendingDeltas.delete(messageId);
  state.activeStreamId = null;
  render();
  startNextQueuedStream();
  unlockComposerIfReady();
}

function fastForwardVisibleStream() {
  if (!state.serverTurnDone || !hasPendingVisibleStream()) return;
  clearActiveStreamTimer();

  const messageIds = [
    state.activeStreamId,
    ...state.streamQueue,
    ...state.pendingDeltas.keys(),
    ...state.streamFinals.keys()
  ].filter(Boolean);

  for (const messageId of [...new Set(messageIds)]) {
    const finalMessage = state.streamFinals.get(messageId);
    if (finalMessage) {
      upsertMessage(finalMessage);
      continue;
    }

    const pending = state.pendingDeltas.get(messageId);
    if (!pending) continue;
    const message = state.messages.find((item) => item.id === messageId);
    if (message) message.content += pending;
  }

  state.pendingDeltas.clear();
  state.streamFinals.clear();
  state.streamQueue = [];
  state.activeStreamId = null;
  render();
  unlockComposerIfReady();
}

function clearActiveStreamTimer() {
  if (state.streamTimer) clearInterval(state.streamTimer);
  state.streamTimer = null;
}

function pushSystemError(content) {
  state.messages.push({
    id: crypto.randomUUID(),
    role: "system",
    kind: "error",
    content,
    createdAt: new Date().toISOString()
  });
  render();
}

function setBusy(isBusy) {
  state.turnInProgress = isBusy;
  setBusyControls();
}

function setBusyControls() {
  const isTurnBusy = state.turnInProgress;
  const isGenerating = state.personaGenerationInProgress;
  sendEl.disabled = isTurnBusy || isGenerating;
  takeActionEl.disabled = isTurnBusy || isGenerating;
  imageInputEl.disabled = isTurnBusy || isGenerating;
  generatePersonasEl.disabled = isTurnBusy || isGenerating;
  addPersonaEl.disabled = isTurnBusy || isGenerating || state.personas.length >= MAX_PERSONAS;
  sendEl.textContent = isTurnBusy ? "Sending" : "Send";
  generatePersonasEl.textContent = isGenerating ? "Generating" : "Generate Personas";
  updateTurnControls();
}

function hasPendingVisibleStream() {
  return Boolean(state.activeStreamId || state.streamQueue.length > 0 || state.pendingDeltas.size > 0 || state.streamFinals.size > 0);
}

function updateTurnControls() {
  const canFastForward = state.serverTurnDone && hasPendingVisibleStream();
  const canBacktrack = !state.turnInProgress && !state.personaGenerationInProgress && state.turnSnapshots.length > 0;
  turnControlsEl.hidden = !canFastForward && !canBacktrack;
  fastForwardEl.hidden = !canFastForward;
  fastForwardEl.disabled = !canFastForward;
  backtrackEl.disabled = !canBacktrack;
}

function unlockComposerIfReady() {
  if (!state.turnInProgress || hasPendingVisibleStream()) return;
  state.turnInProgress = false;
  setBusy(false);
}

function render() {
  messagesEl.innerHTML = "";

  if (state.messages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Start a chat.";
    messagesEl.append(empty);
  }

  for (const message of state.messages) {
    const images = imageAttachmentsForDisplay(message);
    const hasContent = Boolean(message.content.trim());
    const hasImages = images.length > 0;
    if (message.kind === "narration" && !narrationToggleEl.checked) continue;
    if (message.kind === "narration" && !hasContent && !hasImages) continue;
    if (message.role === "persona" && !hasContent && !hasImages && !isMessageStreamingVisible(message.id)) continue;
    if (message.role === "system" && !hasContent && !hasImages && !isMessageStreamingVisible(message.id)) continue;

    const bubble = document.createElement("div");
    bubble.className = `bubble ${message.role} ${message.kind}`;

    if (message.kind !== "narration") {
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = labelFor(message);
      bubble.append(label);
    }

    const content = document.createElement("span");
    content.className = "message-content";
    content.textContent = displayContentFor(message);

    bubble.append(content);
    if (images.length > 0) bubble.append(createMessageImages(images));
    messagesEl.append(bubble);
  }

  scrollMessagesToBottom();
  updateTurnControls();
}

function upsertTokenReport(report) {
  if (!report?.turnId) return;
  const normalized = {
    turnId: report.turnId,
    inputTokens: Number(report.inputTokens) || 0,
    outputTokens: Number(report.outputTokens) || 0,
    inputText: String(report.inputText || ""),
    outputText: String(report.outputText || "")
  };
  const index = state.tokenReports.findIndex((item) => item.turnId === normalized.turnId);
  if (index === -1) {
    state.tokenReports.push(normalized);
  } else {
    state.tokenReports[index] = normalized;
  }
  state.tokenReports = state.tokenReports.slice(-1);
}

function renderTokenUsageBar() {
  tokenUsageBarEl.innerHTML = "";
  tokenUsageBarEl.hidden = state.tokenReports.length === 0;

  const report = state.tokenReports.at(-1);
  if (!report) return;

  const button = document.createElement("button");
  button.type = "button";
  button.className = "token-usage-item";
  button.textContent = `Latest: in ${report.inputTokens} / out ${report.outputTokens}`;
  button.title = "Show input and output text";
  button.addEventListener("click", () => openTokenModal(report));
  tokenUsageBarEl.append(button);
}

function openTokenModal(report) {
  tokenModalTitleEl.textContent = `Latest Turn Tokens: input ${report.inputTokens}, output ${report.outputTokens}`;
  tokenInputTextEl.value = report.inputText;
  tokenOutputTextEl.value = report.outputText;
  tokenModalEl.hidden = false;
  tokenModalCloseEl.focus();
}

function closeTokenModal() {
  tokenModalEl.hidden = true;
  tokenInputTextEl.value = "";
  tokenOutputTextEl.value = "";
}

function imageAttachmentsForDisplay(message) {
  return Array.isArray(message?.imageAttachments)
    ? message.imageAttachments.filter((image) => image?.dataUrl)
    : [];
}

function createMessageImages(images) {
  const wrap = document.createElement("div");
  wrap.className = "message-images";

  for (const image of images) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "message-image-button";
    button.title = image.name || "Open image";
    button.setAttribute("aria-label", `Open ${image.name || "image"}`);
    button.addEventListener("click", () => openImageModal(image));

    const img = document.createElement("img");
    img.src = image.dataUrl;
    img.alt = image.name || "Uploaded image";
    button.append(img);
    wrap.append(button);
  }

  return wrap;
}

function openImageModal(image) {
  if (!image?.dataUrl) return;
  imageModalImgEl.src = image.dataUrl;
  imageModalImgEl.alt = image.name || "Uploaded image";
  imageModalEl.hidden = false;
  imageModalCloseEl.focus();
}

function closeImageModal() {
  imageModalEl.hidden = true;
  imageModalImgEl.removeAttribute("src");
  imageModalImgEl.alt = "";
}

function scrollMessagesToBottom() {
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

function labelFor(message) {
  if (message.kind === "narration") return "";
  if (message.kind === "environment") return "environment";
  if (message.kind === "context") return "context";
  if (message.kind === "location") return "location";
  if (message.kind === "attire") return "attire";
  if (message.kind === "action") return "action";
  if (message.kind === "error") return "error";
  if (message.role === "persona") return message.personaName || personaNameForId(message.personaId) || "persona";
  if (message.role === "user") return userNameEl.value.trim() || "user";
  return message.role;
}

function personaNameForId(personaId) {
  return state.personas.find((persona) => persona.id === personaId)?.nameEl?.value.trim() ||
    state.personas.find((persona) => persona.id === personaId)?.name ||
    "";
}

function displayContentFor(message) {
  if (message.kind !== "narration") return message.content;
  return message.content
}

function shouldQueueStreamMessage(message) {
  return Boolean(
    message?.id &&
      !message.content &&
      (message.role === "persona" || message.role === "narrator")
  );
}

function shouldSuppressStreamMessage(message) {
  return Boolean(message?.id && message.kind === "narration" && !narrationToggleEl.checked);
}

function isMessageStreamingVisible(messageId) {
  return state.activeStreamId === messageId;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
