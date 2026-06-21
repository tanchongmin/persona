import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { pathToFileURL } from "node:url";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

const ROOT = process.cwd();
loadEnv();

const PORT = Number(process.env.PORT || 3000);
const PUBLIC_DIR = join(ROOT, "public");
const QUANTIZED_GEMMA_MODEL = join(ROOT, "models", "gemma-4-E2B-it-q4", "gemma-4-E2B_q4_0-it.gguf");
const GEMMA_RUNNER = join(ROOT, "gemma_runner.py");
const LOCAL_ANACONDA_PYTHON = "/opt/anaconda3/bin/python3";
const GEMMA_PYTHON = process.env.GEMMA_PYTHON ||
  (existsSync(LOCAL_ANACONDA_PYTHON) ? LOCAL_ANACONDA_PYTHON : "python3");
const TRUST_PROXY_HEADERS = process.env.TRUST_PROXY_HEADERS === "true";
const SSE_HEARTBEAT_MS = Number(process.env.SSE_HEARTBEAT_MS || 15000);
const SERVER_MAX_CONNECTIONS = Number(process.env.SERVER_MAX_CONNECTIONS || 200);
const SERVER_REQUEST_TIMEOUT_MS = Number(process.env.SERVER_REQUEST_TIMEOUT_MS || 0);
const SERVER_HEADERS_TIMEOUT_MS = Number(process.env.SERVER_HEADERS_TIMEOUT_MS || 60000);
const SERVER_KEEP_ALIVE_TIMEOUT_MS = Number(process.env.SERVER_KEEP_ALIVE_TIMEOUT_MS || 5000);
const MODEL_GEMMA_4 = "gemma-4";
const MODEL_GPT_5_MINI = "gpt-5-mini";
const MODEL_GPT_5_NANO = "gpt-5-nano";
const OPENAI_RESPONSES_URL = process.env.OPENAI_RESPONSES_URL || "https://api.openai.com/v1/responses";
const MAX_IMAGES_PER_TURN = Number(process.env.MAX_IMAGES_PER_TURN || 4);
const MAX_IMAGE_BYTES = Number(process.env.MAX_IMAGE_BYTES || 5 * 1024 * 1024);
const RECENT_CONTEXT_TURN_LIMIT = Number(process.env.RECENT_CONTEXT_TURN_LIMIT || 10);
const MAX_JSON_BODY_BYTES = Number(
  process.env.MAX_JSON_BODY_BYTES ||
  ((RECENT_CONTEXT_TURN_LIMIT + 1) * MAX_IMAGES_PER_TURN * MAX_IMAGE_BYTES) + (5 * 1024 * 1024)
);
let gemmaProcess = null;
let gemmaReadline = null;
let gemmaReady = null;
let gemmaNextRequestId = 1;
const gemmaPendingRequests = new Map();

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8"
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", `http://${req.headers.host}`);

    if (req.method === "POST" && url.pathname === "/api/turn") {
      const body = await readJson(req);
      await handleTurnStream(body, req, res);
      return;
    }

    if (req.method === "POST" && url.pathname === "/api/generate-personas") {
      const body = await readJson(req);
      await handlePersonaGeneration(body, req, res);
      return;
    }

    if (req.method === "GET") {
      await serveStatic(url.pathname, res);
      return;
    }

    sendJson(res, 405, { error: "Method not allowed" });
  } catch (error) {
    console.error(error);
    if (res.headersSent) {
      sendSse(res, "error", {
        error: error instanceof Error ? error.message : "Unexpected server error"
      });
      res.end();
      return;
    }
    sendJson(res, 500, {
      error: error instanceof Error ? error.message : "Unexpected server error"
    });
  }
});

server.maxConnections = SERVER_MAX_CONNECTIONS;
server.requestTimeout = SERVER_REQUEST_TIMEOUT_MS;
server.headersTimeout = SERVER_HEADERS_TIMEOUT_MS;
server.keepAliveTimeout = SERVER_KEEP_ALIVE_TIMEOUT_MS;

if (isMainModule()) {
  server.listen(PORT, () => {
    console.log(`Persona Chat is running at http://localhost:${PORT}`);
  });
}

function isMainModule() {
  return process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
}

function loadEnv() {
  const envPath = join(ROOT, ".env");
  if (!existsSync(envPath)) return;

  const lines = readFileSync(envPath, "utf8").split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
    const index = trimmed.indexOf("=");
    const key = trimmed.slice(0, index).trim();
    let value = trimmed.slice(index + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (key && process.env[key] === undefined) process.env[key] = value;
  }
}

async function readJson(req) {
  let raw = "";
  for await (const chunk of req) {
    raw += chunk;
    if (Buffer.byteLength(raw, "utf8") > MAX_JSON_BODY_BYTES) {
      throw new Error("Request body is too large.");
    }
  }
  if (!raw) return {};
  return JSON.parse(raw);
}

async function serveStatic(pathname, res) {
  const requested = pathname === "/" ? "/index.html" : pathname;
  const safePath = normalize(requested).replace(/^(\.\.[/\\])+/, "");
  const filePath = join(PUBLIC_DIR, safePath);

  if (!filePath.startsWith(PUBLIC_DIR)) {
    sendJson(res, 403, { error: "Forbidden" });
    return;
  }

  try {
    const file = await readFile(filePath);
    res.writeHead(200, {
      "Content-Type": mimeTypes[extname(filePath)] || "application/octet-stream"
    });
    res.end(file);
  } catch {
    sendJson(res, 404, { error: "Not found" });
  }
}

function sendJson(res, status, payload) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(payload));
}

function getClientIp(req) {
  if (TRUST_PROXY_HEADERS) {
    const forwarded = req.headers["x-forwarded-for"];
    const value = Array.isArray(forwarded) ? forwarded[0] : forwarded;
    const firstForwardedIp = clean(String(value || "").split(",")[0]);
    if (firstForwardedIp) return firstForwardedIp;
  }
  return req.socket?.remoteAddress || "unknown";
}

async function handleTurnStream(body, req, res) {
  const personas = normalizePersonas(body);
  const scenario = clean(body.scenario);
  const userName = clean(body.userName) || "John";
  const userPersona = clean(body.userPersona) || "A friendly, charming male friend";
  const summary = clean(body.summary);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const recentMessages = messages;
  const type = body.type;
  const content = clean(body.content);
  const imageAttachments = normalizeImageAttachments(body.imageAttachments);
  const model = normalizeModelSelection(body.model);

  if (!personas.some((persona) => persona.description)) throw new Error("At least one persona description is required.");
  if (!content && imageAttachments.length === 0) throw new Error("Message content or an image is required.");

  const clientIp = getClientIp(req);
  startSse(res);
  try {
    let newMessages;
    const turnId = crypto.randomUUID();

    if (type === "chat") {
      const userMessage = makeMessage("user", "chat", content, turnId, {
        imageAttachments
      });
      sendSse(res, "message", userMessage);

      const storyResult = await streamPersonaReply({
        personas,
        scenario,
        userName,
        userPersona,
        summary,
        recentMessages: [...recentMessages, userMessage],
        userMessage,
        extraInstruction: `If ${userName}'s message implies an action, visible change, or emotion, let the narration show only the immediate visible beat around the relevant persona responses.`,
        turnId,
        clientIp,
        model,
        res
      });
      if (storyResult.summary) sendSse(res, "summary", { summary: storyResult.summary });
      newMessages = [...messages, userMessage, ...storyResult.messages];
    } else if (type === "action") {
      const actionMessage = makeMessage("system", "action", content, turnId, {
        imageAttachments
      });
      sendSse(res, "message", actionMessage);

      const storyResult = await streamPersonaReply({
        personas,
        scenario,
        userName,
        userPersona,
        summary,
        recentMessages: [...recentMessages, actionMessage],
        userMessage: actionMessage,
        extraInstruction: "The user has taken this explicit action. Treat it as already carried out in the scene and let only relevant personas respond from after the action happens.",
        turnId,
        clientIp,
        model,
        res
      });
      if (storyResult.summary) sendSse(res, "summary", { summary: storyResult.summary });
      newMessages = [...messages, actionMessage, ...storyResult.messages];
    } else {
      throw new Error("Unknown turn type.");
    }

    sendSse(res, "done", { recentMessages: sliceLastTurns(newMessages, RECENT_CONTEXT_TURN_LIMIT) });
    res.end();
  } catch (error) {
    throw error;
  }
}

function normalizeImageAttachments(value) {
  if (!Array.isArray(value)) return [];

  return value.slice(0, MAX_IMAGES_PER_TURN).map((image, index) => {
    const dataUrl = clean(image?.dataUrl);
    const match = dataUrl.match(/^data:(image\/(?:png|jpe?g|webp|gif));base64,([A-Za-z0-9+/=]+)$/i);
    if (!match) throw new Error("Image attachments must be PNG, JPEG, WebP, or GIF data URLs.");

    const size = Number(image?.size) || Math.floor(match[2].length * 3 / 4);
    if (size > MAX_IMAGE_BYTES) throw new Error("One or more images are too large.");

    return {
      id: clean(image?.id) || `image-${index + 1}`,
      name: clean(image?.name) || `image-${index + 1}`,
      type: match[1].toLowerCase().replace("image/jpg", "image/jpeg"),
      size,
      dataUrl
    };
  });
}

async function handlePersonaGeneration(body, req, res) {
  const scenario = clean(body.scenario);
  const userName = clean(body.userName) || "John";
  const userPersona = clean(body.userPersona) || "A friendly, charming male friend";
  const model = normalizeModelSelection(body.model);

  if (!scenario) throw new Error("Scenario is required to generate personas.");

  const prompt = buildPersonaGenerationPrompt({ scenario, userName, userPersona });
  const raw = await generateText(prompt, {
    maxOutputTokens: personaGenerationTokenLimit(model),
    clientIp: getClientIp(req),
    model
  });
  const generated = parseGeneratedPersonaPayload(raw, userName);
  const personas = generated.personas;

  if (personas.length === 0) throw new Error("No usable personas were generated.");
  sendJson(res, 200, generated);
}

function clean(value) {
  return typeof value === "string" ? value.trim() : "";
}

function normalizePersonas(body) {
  const source = Array.isArray(body.personas) && body.personas.length > 0
    ? body.personas
    : [{
        id: "persona-1",
        name: body.personaName,
        gender: body.personaGender,
        description: body.persona
      }];

  return normalizePersonaList(source);
}

function normalizePersonaList(personas = []) {
  return (Array.isArray(personas) ? personas : []).slice(0, 10).map((persona, index) => ({
    id: clean(persona.id) || `persona-${index + 1}`,
    name: normalizePersonaName(persona.name, `Persona ${index + 1}`),
    gender: normalizeGender(persona.gender),
    description: clean(persona.description)
  }));
}

function normalizePersonaName(value, fallback = "Persona") {
  const name = clean(value) || fallback;
  const unbolded = name.match(/^\*\*(.*?)\*\*$/)?.[1] || name;
  if (clean(unbolded).toLowerCase() === "narration") return "Narration_Persona";
  return name;
}

function normalizeGender(value) {
  const gender = clean(value).toLowerCase();
  if (gender === "male") return "Male";
  if (gender === "female") return "Female";
  return "";
}

function normalizeModelSelection(value) {
  const model = clean(value);
  if (model === MODEL_GPT_5_MINI) return MODEL_GPT_5_MINI;
  if (model === MODEL_GPT_5_NANO) return MODEL_GPT_5_NANO;
  return MODEL_GEMMA_4;
}

function isOpenAIModel(model) {
  return model === MODEL_GPT_5_MINI || model === MODEL_GPT_5_NANO;
}

function personaGenerationTokenLimit(model) {
  if (isOpenAIModel(model)) return Number(process.env.OPENAI_PERSONA_GENERATION_TOKENS || 4000);
  return Number(process.env.GEMMA_PERSONA_GENERATION_TOKENS || 700);
}

function streamTokenLimit(model) {
  if (isOpenAIModel(model)) return Number(process.env.OPENAI_STREAM_TOKENS || 4000);
  return Number(process.env.GEMMA_STREAM_TOKENS || 32000);
}

function buildPersonaGenerationPrompt({ scenario, userName, userPersona }) {
  return [
    "Generate personas for a roleplay chat scene.",
    `Scenario:\n${scenario}`,
    `User name:\n${userName}`,
    `User description:\n${userPersona}`,
    `In the scenario text, "you", "your", "I", "me", and "my" refer to the user, ${userName}. Do not generate a persona for those pronouns.`,
    "Create the smallest number of personas needed for this scenario, up to a hard maximum of 10 personas. Usually create 1 or 2 personas. Create more only when the scenario clearly needs that many distinct people present at the start. Never create extra background, duplicate, decorative, or nice-to-have personas.",
    "If the scenario mentions a person's name, generate that named person as a persona unless the name is clearly the user's own name. Preserve the mentioned name exactly, including the full name when given. For example, use Donald Trump instead of Donald if the scenario says Donald Trump.",
    "Never name a persona Narration because **Narration**: is a reserved prose prefix. If the scenario requires someone named Narration, name that persona Narration_Persona instead.",
    "Do not create side characters, bystanders, staff, relatives, friends, rivals, witnesses, or other extra people unless the scenario explicitly asks for them.",
    "Choose only personas who have a necessary reason to be present, create tension, help move the story, reveal information, oppose the user, support the user, or complicate the scene.",
    "Use common modern English first names. Each gender must be exactly Male or Female.",
    "Each persona should feel like a human character, not an assistant, helper, narrator, or generic NPC.",
    "Descriptions should include temperament, voice, boundaries, likely behavior in the scenario, and what role they fulfil in the story.",
    "Also generate a concise summary of the starting situation, including where the scene begins, who is present, what has happened so far, and any important relationship, consent, tension, or pending-question context.",
    `Return strict JSON only, with no markdown, in this exact shape:
{
  "summary": "Concise current-state summary of what has happened so far.",
  "personas": [
    {
      "name": "Common first name",
      "gender": "Male",
      "description": "Human character description tied to the scenario."
    }
  ]
}`,
    "Do not include any top-level field except summary and personas. Do not include any persona field except name, gender, and description."
  ].join("\n\n");
}

function parseGeneratedPersonaPayload(raw, userName = "John") {
  const payload = parseGeneratedPersonaJson(raw);
  return {
    personas: parseGeneratedPersonasFromPayload(payload, userName),
    summary: clean(payload?.summary) || generatedScenarioSummary(userName)
  };
}

function parseGeneratedPersonas(raw, userName = "John") {
  const payload = parseGeneratedPersonaJson(raw);
  return parseGeneratedPersonasFromPayload(payload, userName);
}

function parseGeneratedPersonasFromPayload(payload, userName = "John") {
  const source = Array.isArray(payload) ? payload : Array.isArray(payload?.personas) ? payload.personas : [];

  return source.slice(0, 10).map((persona, index) => {
    const name = normalizePersonaName(persona?.name, `Persona ${index + 1}`);
    const gender = normalizeGender(persona?.gender) || (index % 2 === 0 ? "Female" : "Male");
    const description = clean(persona?.description);

    return {
      name,
      gender,
      description
    };
  }).filter((persona) => persona.description);
}

function parseGeneratedPersonaJson(raw) {
  const text = String(raw || "").trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  try {
    return JSON.parse(text);
  } catch {
    const objectStart = text.indexOf("{");
    const objectEnd = text.lastIndexOf("}");
    if (objectStart !== -1 && objectEnd > objectStart) {
      try {
        return JSON.parse(text.slice(objectStart, objectEnd + 1));
      } catch {
        // Fall through to array extraction.
      }
    }

    const arrayStart = text.indexOf("[");
    const arrayEnd = text.lastIndexOf("]");
    if (arrayStart !== -1 && arrayEnd > arrayStart) {
      return JSON.parse(text.slice(arrayStart, arrayEnd + 1));
    }

    throw new Error("Generated personas were not valid JSON.");
  }
}

function generatedScenarioSummary(userName) {
  return `${userName} is present in the scenario. The conversation has just begun, and no prior events have happened yet.`;
}

function makeMessage(role, kind, content, turnId = null, extra = {}) {
  const message = {
    id: crypto.randomUUID(),
    role,
    kind,
    content,
    createdAt: new Date().toISOString(),
    ...extra
  };
  if (turnId) message.turnId = turnId;
  return message;
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

function startSse(res) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no"
  });
  res.write(": connected\n\n");
  if (SSE_HEARTBEAT_MS > 0) {
    const heartbeat = setInterval(() => {
      res.write(": keep-alive\n\n");
    }, SSE_HEARTBEAT_MS);
    res.on("close", () => clearInterval(heartbeat));
  }
}

function sendSse(res, event, payload) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(payload)}\n\n`);
}

async function streamPersonaReply({ personas, personaName, persona, scenario = "", userName, userPersona, summary = "", recentMessages, userMessage, extraInstruction = "", turnId, clientIp = "unknown", model = MODEL_GEMMA_4, res }) {
  const normalizedPersonas = normalizePersonaList(personas || [{
    id: "persona-1",
    name: personaName || "Persona",
    description: persona || ""
  }]);
  const prompt = buildPersonaReplyPrompt({
    personas: normalizedPersonas,
    scenario,
    userName,
    userPersona,
    summary,
    recentMessages,
    userMessage,
    extraInstruction
  });

  const segmentStreamer = createTaggedSegmentStreamer({ res, turnId, personas: normalizedPersonas });
  const llmImageAttachments = imageAttachmentsForLlmContext(recentMessages, userMessage);
  const raw = await generateStream(prompt, {
    maxOutputTokens: streamTokenLimit(model),
    clientIp,
    model,
    imageAttachments: llmImageAttachments,
    onTextDelta: (delta) => {
      segmentStreamer.feed(delta);
    }
  });
  sendSse(res, "tokenUsage", tokenUsagePayload({
    turnId,
    inputText: tokenUsageInputText(prompt, llmImageAttachments),
    outputText: raw
  }));
  const streamedResult = segmentStreamer.finish();
  if (streamedResult.messages.length > 0) {
    if (!streamedResult.summary) {
      streamedResult.summary = messagesAndSummaryFromTaggedSequence(raw, turnId, normalizedPersonas).summary;
    }
    return streamedResult;
  }

  const parsed = messagesAndSummaryFromTaggedSequence(raw || silentFallbackSequence(normalizedPersonas[0]?.name || "Persona"), turnId, normalizedPersonas);
  const messages = parsed.messages;
  for (const message of messages) {
    sendSse(res, "message", message);
    sendSse(res, "messageDone", message);
  }
  return parsed;
}

function buildPersonaReplyPrompt({ personas, personaName, persona, scenario = "", userName, userPersona, summary = "", recentMessages, userMessage, extraInstruction = "" }) {
  const normalizedPersonas = normalizePersonaList(personas || [{
    id: "persona-1",
    name: personaName || "Persona",
    description: persona || ""
  }]);
  const personaNames = normalizedPersonas.map((item) => item.name).join(", ");
  const personaLineGuide = normalizedPersonas
    .map((item) => `**${item.name}**: spoken dialogue here`)
    .join("\n");
  const userPronounInstruction = normalizedPersonas.length > 1
    ? `User pronoun rule: In every User: line, "you", "your", and "yourself" are never ${userName}. They refer to the persona being addressed by name, or the most relevant nearby persona if no name is given. For example, "Emma, can you move?" means Emma should move, not ${userName}.`
    : `User pronoun rule: In every User: line, "you", "your", and "yourself" are never ${userName}. They refer to ${normalizedPersonas[0]?.name || "the persona"}. For example, "can you move?" means ${normalizedPersonas[0]?.name || "the persona"} should move, not ${userName}.`;
  const multiPersonaNarrationInstruction = normalizedPersonas.length > 1
    ? "Because more than one persona is present, include **Narration**: lines whenever staging, movement, facial expression, silence, turn-taking, or spatial relationship matters. Use **Narration**: to connect persona replies so the exchange does not become a bare script."
    : "";
  const narrationModeBlock = [
    `Write the next natural conversation for the relevant personas only. Available personas: ${personaNames}.`,
    "If the persona is performing an action, describe it in detail till completion. Do not describe any action that has already happened in the past.",
    "Not every persona needs to respond. Include only personas who are directly addressed, affected, nearby, or naturally motivated to react. Let irrelevant personas stay silent.",
    `Use bold persona-name prefixes for spoken dialogue, and use this format exactly:\n${personaLineGuide}`,
    "Use **Narration**: for prose when possible. If a model line does not start with **Narration**:, a bold available persona-name prefix, or a **Summary**: line, it will be treated as visible narration. Use **Summary**: only for the hidden current-state summary.",
    "Only **Summary**: lines are hidden from the chat. If the model accidentally writes Summary: without bolding, it is still treated as hidden summary. **Narration**: lines are shown as narration; if the model accidentally writes Narration: without bolding, it is still treated as narration. Unprefixed prose, unbolded persona labels, unknown labels, and any other non-persona output are shown as narration.",
    `Never write a User:, ${userName}:, <user>:, or any other user-prefixed line in the generated response. Do not speak, act, answer, decide, or continue the conversation on ${userName}'s behalf.`,
    userPronounInstruction,
    "After all visible narration and persona dialogue for this turn, return exactly one **Summary**: line. The summary line is not dialogue or narration and must not repeat the visible reply verbatim.",
    "The **Summary**: line must be a detailed standalone continuity memory that can replace older conversation messages. A future response should be possible using only this summary, the persona descriptions, and the next User: line or [state/action] line.",
    "Update the prior summary after this same turn. Carry forward every durable fact still needed for continuity: current location, who is present, what each person is doing or holding, visible images or objects that still matter, clothing or physical state if relevant, relationship dynamics, emotional state, promises, decisions, conflicts, consent, boundaries, unresolved questions, active plans, and why the current moment is happening.",
    "Then add the latest user message/action, persona dialogue, and narration from this turn with concrete cause and effect. Include enough detail that another model could continue the scene naturally without seeing the older transcript.",
    "Do not write only the latest beat, a vague current-state note, or a short recap. The summary is not a verbatim transcript, but it should be dense, specific, and multi-sentence when needed rather than compressed into one generic sentence.",
    "If the scene benefits from it, use multiple dialogue lines from one or more relevant personas separated by **Narration**: lines so the exchange feels cinematic and alive.",
    "Before writing any persona dialogue or **Narration**: prose, compare it against the most recent messages and avoid repeating already generated wording, actions, gestures, locations, descriptions, emotions, or conclusions.",
    "Minimize questions from personas. Do not end every turn with a question; it is often better to end with a statement, reaction, decision, action, silence, or narration that lets the conversation breathe.",
    "Never put any questions in **Narration**:. If there is a question, a persona must ask it on that speaker's own line.",
    "Never put any speech lines or quotation marks in **Narration**:. Put all spoken words on named persona lines.",
    "Narration should be cinematic third party prose, and should include inner thoughts of relevant personas when useful. Use concrete sensory details, movement, expression, posture, silence, pacing, and environment to immerse the user in the scene without overexplaining.",
    multiPersonaNarrationInstruction,
    `When narration refers to ${userName}, write it in second person as "you", "your", or "yourself". Do not use ${userName}, "the user", "the player", "John", or any other third-person user label in **Narration**:. Use ${userName} only in persona dialogue or the hidden **Summary**: line.`,
    "Refer to personas by name in narration when needed for clarity.",
  ].filter(Boolean).join("\n");

  return [
    contextBlockForPersonas(normalizedPersonas, userName, userPersona, summary, recentMessages, scenario),
    narrationModeBlock,
    "Dialogue must be spoken only by an available persona named in the line prefix. Stay consistent with each persona description and the conversation summary. Do not break character.",
    "Treat the most recent messages as already seen and already said. Do not repeat prior narration, prior dialogue, the same gesture, the same location/attire statement, the same sensory description, or the same emotions unless the user explicitly asks to revisit it.",
    `If ${userName} has already consented to an action in the conversation history or summary, treat that consent as remembered and active. Do not ask for the same consent again unless ${userName} revoked it, limited it, or the new action is materially different.`,
    `When ${userName}'s latest User: message says "you", "your", or "yourself", do not make ${userName} perform that referenced action or possess that referenced thing. The addressed persona should respond or act as the target of that second-person wording.`,
    `If ${userName} asks a persona to decide, says anything is good, says either option is fine, or otherwise leaves the choice to a persona, that persona should make a concrete choice and continue from that choice. Do not ask ${userName} the same choice question again.`,
    "Dialogue is already labeled in the UI. After a bold persona prefix, write only the speaker's spoken words. Do not repeat the name, 'says', or quoted attribution inside the content.",
    "Do not describe a persona's inner thoughts, emotional state, facial expression, body language, or motivation inside that persona's dialogue line. Put those details in **Narration**: lines before or after the spoken line.",
    "Only ask a question when a persona genuinely needs an answer to continue or when asking fits the character and moment. Prefer implied invitations and concrete statements over repeated check-ins.",
    `Dialogue segments are persona replies, not ${userName}'s. Each dialogue segment should be human and in-character; use natural phrasing, contractions when they fit, pauses, interruptions, and concrete reactions when the scene calls for them.`,
    "Each persona should act according to their own description, current mood, relationship state, boundaries, and situation. Character fidelity matters more than being cooperative, friendly, or agreeable.",
    "Do not make personas helpful, compliant, agreeable, or responsive by default. A persona may disagree, hesitate, ignore a request, refuse, tease, deflect, become guarded, or stay silent if that fits the character and scene.",
    "If a relevant persona refuses to speak or chooses silence, do not force a dialogue line. Use **Narration**: to show the silence, expression, posture, or action instead. The persona should still remain part of the scene unless the user changes the scene.",
    extraInstruction,
    `Current user turn:\n${formatMessage(userMessage, { focusImages: true })}`
  ].filter(Boolean).join("\n\n");
}

function silentFallbackSequence(personaName) {
  return `**Narration**: ${personaName} does not answer.`;
}

function contextBlock(personaName, persona, userName, userPersona, summary, recentMessages) {
  return [
    `Persona name:\n${personaName}`,
    `Persona description:\n${persona}`,
    `User name:\n${userName}`,
    `User description:\n${userPersona}`,
    perspectiveGuide(personaName, userName),
    `Conversation summary:\n${summary || "No summary yet."}`,
    `Most recent messages:\n${recentMessages.map((message) => formatMessage(message, { boldUserLabel: true })).join("\n") || "No prior messages."}`
  ].filter(Boolean).join("\n\n");
}

function contextBlockForPersonas(personas, userName, userPersona, summary, recentMessages, scenario = "") {
  return [
    `Personas:\n${personas.map((persona) => `${persona.name}${persona.gender ? ` (${persona.gender})` : ""}:\n${persona.description || "No description provided."}`).join("\n\n")}`,
    scenario ? `Scenario:\n${scenario}\nUse this as the current story premise, setting, and pressure. Generate the scene and persona behavior so the conversation grows out of this scenario unless the latest turn clearly changes it.` : "",
    `User name:\n${userName}`,
    `User description:\n${userPersona}`,
    perspectiveGuideForPersonas(personas, userName),
    `Conversation summary:\n${summary || "No summary yet."}`,
    `Most recent messages:\n${recentMessages.map((message) => formatMessage(message, { boldUserLabel: true })).join("\n") || "No prior messages."}`
  ].filter(Boolean).join("\n\n");
}

function perspectiveGuide(personaName, userName) {
  return [
    "Perspective guide:",
    `- In User: lines, ${userName} is speaking. "I", "me", and "my" refer to ${userName}; "you" and "your" never refer to ${userName} and instead refer to ${personaName}.`,
    `- In [state/action], ${userName} is the actor unless another actor is explicitly named. "I", "me", and "my" refer to ${userName}; "you" and "your" never refer to ${userName} and instead refer to ${personaName}.`,
    `- In **${personaName}**: lines, ${personaName} is speaking. "I", "me", and "my" refer to ${personaName}; "you" and "your" refer to ${userName}.`,
    `- In **Narration**: lines, the story is written to ${userName} in second person. "you" and "your" always refer to ${userName}, not ${personaName}.`
  ].join("\n");
}

function perspectiveGuideForPersonas(personas, userName) {
  const personaNames = personas.map((persona) => persona.name).join(", ");
  return [
    "Perspective guide:",
    `- In User: lines, ${userName} is speaking. "I", "me", and "my" refer to ${userName}; "you" and "your" never refer to ${userName} and instead refer to the addressed persona or personas.`,
    `- In [state/action], ${userName} is the actor unless another actor is explicitly named. "I", "me", and "my" refer to ${userName}; "you" and "your" never refer to ${userName} and instead refer to the addressed persona or personas.`,
    `- In **persona-name**: lines, the named persona is speaking. "I", "me", and "my" refer to that speaker; "you" and "your" refer to ${userName} unless another persona is clearly addressed.`,
    `- In **Narration**: lines, the story is written to ${userName} in second person. "you" and "your" always refer to ${userName}, not ${personaNames}.`
  ].join("\n");
}

function formatMessage(message, options = {}) {
  const imageText = imageAttachmentPromptText(message?.imageAttachments, options);
  const content = [message.content, imageText].filter(Boolean).join("\n");
  if (message.role === "user" && message.kind === "chat") {
    return `${options.boldUserLabel ? "**User**" : "User"}: ${content}`;
  }
  if (message.kind === "narration") return `**Narration**: ${content}`;
  if (message.kind === "location") return `[state/location] ${content}`;
  if (message.kind === "attire") return `[state/attire] ${content}`;
  if (message.kind === "environment") return `[environment] ${content}`;
  if (message.kind === "context") return `[state/context] ${content}`;
  if (message.kind === "action") return `[state/action] ${content}`;
  if (message.role === "persona" && message.personaName) return `**${message.personaName}**: ${content}`;
  if (message.role === "persona" && message.kind === "chat") return `**Persona**: ${content}`;
  return `[${message.role}/${message.kind}] ${content}`;
}

function imageAttachmentPromptText(imageAttachments = [], options = {}) {
  const images = Array.isArray(imageAttachments) ? imageAttachments : [];
  if (images.length === 0) return "";

  const lines = images.map((image, index) => {
    const name = clean(image?.name) || `image ${index + 1}`;
    const type = clean(image?.type) || "image";
    const size = Number(image?.size);
    const sizeText = Number.isFinite(size) && size > 0 ? `, ${Math.round(size / 1024)} KB` : "";
    const availability = image?.dataUrl
      ? "The selected model can inspect the image if it supports image input."
      : "This is a prior uploaded image retained as transcript metadata.";
    return `- ${name} (${type}${sizeText}). ${availability}`;
  }).join("\n");

  const heading = options.focusImages
    ? "[latest uploaded images - focus on these for this turn]"
    : "[uploaded images]";
  return `${heading}\n${lines}`;
}

function imageAttachmentsForLlmContext(messages = [], focusMessage = null) {
  const focusImageIds = new Set();
  const focusImages = imageAttachmentsFromMessage(focusMessage);
  for (const image of focusImages) {
    if (image?.id) focusImageIds.add(image.id);
  }

  const priorImages = (Array.isArray(messages) ? messages : [])
    .flatMap((message) => imageAttachmentsFromMessage(message))
    .filter((image) => !image?.id || !focusImageIds.has(image.id));

  return [...focusImages, ...priorImages]
    .filter((image) => typeof image?.dataUrl === "string" && image.dataUrl.startsWith("data:image/"));
}

function imageAttachmentsFromMessage(message) {
  return Array.isArray(message?.imageAttachments) ? message.imageAttachments : [];
}

function tokenUsageInputText(prompt, imageAttachments = []) {
  const imageCount = (Array.isArray(imageAttachments) ? imageAttachments : [])
    .filter((image) => typeof image?.dataUrl === "string" && image.dataUrl.startsWith("data:image/"))
    .length;
  if (imageCount === 0) return prompt;

  return [
    prompt,
    "[image inputs]",
    ...Array.from({ length: imageCount }, () => "<image>")
  ].join("\n");
}

function tokenUsagePayload({ turnId, inputText = "", outputText = "" }) {
  return {
    turnId,
    inputTokens: estimateTokenCount(inputText),
    outputTokens: estimateTokenCount(outputText),
    inputText,
    outputText
  };
}

function estimateTokenCount(value) {
  const text = String(value || "").trim();
  if (!text) return 0;
  const pieces = text.match(/<image>|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]/g) || [];
  return pieces.length;
}

async function gemmaText(input, options = {}) {
  return gemmaRequest({
    input,
    maxOutputTokens: options.maxOutputTokens || 500,
    stream: false,
    clientIp: options.clientIp,
    imageAttachments: options.imageAttachments
  });
}

async function generateText(input, options = {}) {
  if (isOpenAIModel(options.model)) {
    return openaiText(input, options);
  }
  return gemmaText(input, options);
}

function extractTaggedText(text, tagName) {
  const pattern = new RegExp(`<${tagName}>([\\s\\S]*?)<\\/${tagName}>`, "i");
  const match = String(text || "").match(pattern);
  return match ? clean(match[1]) : "";
}

async function gemmaStream(input, options = {}) {
  return gemmaRequest({
    input,
    maxOutputTokens: options.maxOutputTokens || 900,
    stream: true,
    clientIp: options.clientIp,
    onTextDelta: options.onTextDelta,
    imageAttachments: options.imageAttachments
  });
}

async function generateStream(input, options = {}) {
  if (isOpenAIModel(options.model)) {
    return openaiStream(input, options);
  }
  return gemmaStream(input, options);
}

function gemmaRequestBody({ input, maxOutputTokens, stream = false, clientIp = "unknown", imageAttachments = [] }) {
  return {
    input,
    image_urls: gemmaImageUrls(imageAttachments),
    client_ip: clientIp,
    max_new_tokens: maxOutputTokens,
    stream,
    temperature: Number(process.env.GEMMA_TEMPERATURE || 1),
    top_p: Number(process.env.GEMMA_TOP_P || 0.95),
    top_k: Number(process.env.GEMMA_TOP_K || 64)
  };
}

function gemmaImageUrls(imageAttachments = []) {
  return (Array.isArray(imageAttachments) ? imageAttachments : [])
    .map((image) => image?.dataUrl)
    .filter((value) => typeof value === "string" && value.startsWith("data:image/"));
}

function openaiRequestBody({ input, maxOutputTokens, stream = false, model = MODEL_GPT_5_MINI, imageAttachments = [] }) {
  return {
    model: isOpenAIModel(model) ? model : MODEL_GPT_5_MINI,
    input: openaiInput(input, imageAttachments),
    max_output_tokens: maxOutputTokens,
    stream,
    reasoning: {
      effort: process.env.OPENAI_REASONING_EFFORT || "minimal"
    }
  };
}

async function openaiText(input, options = {}) {
  return openaiRequest({
    input,
    maxOutputTokens: options.maxOutputTokens || 900,
    stream: false,
    model: options.model,
    imageAttachments: options.imageAttachments
  });
}

async function openaiStream(input, options = {}) {
  return openaiRequest({
    input,
    maxOutputTokens: options.maxOutputTokens || 900,
    stream: true,
    onTextDelta: options.onTextDelta,
    model: options.model,
    imageAttachments: options.imageAttachments
  });
}

function openaiInput(input, imageAttachments = []) {
  const images = (Array.isArray(imageAttachments) ? imageAttachments : []).filter((image) => image?.dataUrl);
  if (images.length === 0) return input;

  return [{
    role: "user",
    content: [
      { type: "input_text", text: input },
      ...images.map((image) => ({
        type: "input_image",
        image_url: image.dataUrl
      }))
    ]
  }];
}

async function openaiRequest({ input, maxOutputTokens, stream = false, onTextDelta, model = MODEL_GPT_5_MINI, imageAttachments = [] }) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error("OPENAI_API_KEY is not set.");

  const response = await fetch(OPENAI_RESPONSES_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`
    },
    body: JSON.stringify(openaiRequestBody({ input, maxOutputTokens, stream, model, imageAttachments }))
  });

  if (!response.ok) {
    throw new Error(await openaiErrorMessage(response));
  }

  if (stream) return readOpenAIStream(response, onTextDelta);

  const data = await response.json();
  const text = extractOpenAIOutputText(data);
  if (!text) throw new Error(openaiEmptyResponseMessage(data));
  return text.trim();
}

async function openaiErrorMessage(response) {
  const fallback = `OpenAI request failed with status ${response.status}.`;
  const body = await response.text().catch(() => "");
  if (!body) return fallback;
  try {
    const parsed = JSON.parse(body);
    return parsed?.error?.message || fallback;
  } catch {
    return fallback;
  }
}

async function readOpenAIStream(response, onTextDelta) {
  if (!response.body) throw new Error("OpenAI streaming response did not include a body.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let output = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let eventEnd = buffer.indexOf("\n\n");
    while (eventEnd !== -1) {
      const eventText = buffer.slice(0, eventEnd);
      buffer = buffer.slice(eventEnd + 2);
      const eventOutput = handleOpenAIStreamEvent(eventText, onTextDelta);
      output += eventOutput;
      eventEnd = buffer.indexOf("\n\n");
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) output += handleOpenAIStreamEvent(buffer, onTextDelta);
  if (!output) throw new Error("OpenAI returned an empty response.");
  return output.trim();
}

function handleOpenAIStreamEvent(eventText, onTextDelta) {
  const dataLines = eventText
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim());

  let output = "";
  for (const dataLine of dataLines) {
    if (!dataLine || dataLine === "[DONE]") continue;
    const event = JSON.parse(dataLine);
    if (event.type === "response.output_text.delta" && event.delta) {
      output += event.delta;
      onTextDelta?.(event.delta);
    }
    if (event.type === "response.error") {
      throw new Error(event.error?.message || "OpenAI streaming request failed.");
    }
  }
  return output;
}

function extractOpenAIOutputText(data) {
  if (typeof data?.output_text === "string") return data.output_text;
  const chunks = [];
  for (const item of data?.output || []) {
    for (const content of item?.content || []) {
      if (typeof content?.text === "string") chunks.push(content.text);
    }
  }
  return chunks.join("");
}

function openaiEmptyResponseMessage(data) {
  if (data?.status === "incomplete") {
    const reason = data?.incomplete_details?.reason || "unknown reason";
    return `OpenAI returned an incomplete response before producing visible text (${reason}). Increase OPENAI_PERSONA_GENERATION_TOKENS or OPENAI_STREAM_TOKENS if this persists.`;
  }
  if (data?.status === "failed") {
    return data?.error?.message || "OpenAI response failed before producing visible text.";
  }
  return "OpenAI returned an empty response.";
}

async function gemmaRequest({ input, maxOutputTokens, stream = false, clientIp = "unknown", onTextDelta, imageAttachments = [] }) {
  await ensureGemmaReady();

  const id = String(gemmaNextRequestId++);
  const payload = gemmaRequestPayload({ id, input, maxOutputTokens, stream, clientIp, imageAttachments });

  return new Promise((resolve, reject) => {
    gemmaPendingRequests.set(id, {
      output: "",
      onTextDelta,
      resolve: (text) => {
        if (!text) {
          reject(new Error("Gemma returned an empty response."));
          return;
        }
        resolve(text.trim());
      },
      reject
    });

    gemmaProcess.stdin.write(`${JSON.stringify(payload)}\n`, (error) => {
      if (!error) return;
      gemmaPendingRequests.delete(id);
      reject(error);
    });
  });
}

function gemmaRequestPayload({ id, input, maxOutputTokens, stream = false, clientIp = "unknown", imageAttachments = [] }) {
  return {
    id,
    ...gemmaRequestBody({ input, maxOutputTokens, stream, clientIp, imageAttachments })
  };
}

function ensureGemmaReady() {
  if (gemmaReady) return gemmaReady;

  if (!existsSync(GEMMA_RUNNER)) {
    throw new Error(`Gemma runner is missing at ${GEMMA_RUNNER}.`);
  }
  if (!existsSync(QUANTIZED_GEMMA_MODEL)) {
    throw new Error(`Quantized Gemma model is missing at ${QUANTIZED_GEMMA_MODEL}. Run \`npm run download:model:vps\` first.`);
  }

  gemmaProcess = spawn(GEMMA_PYTHON, [GEMMA_RUNNER], {
    cwd: ROOT,
    env: process.env,
    stdio: ["pipe", "pipe", "pipe"]
  });

  gemmaReadline = createInterface({ input: gemmaProcess.stdout });
  gemmaProcess.stderr.on("data", (chunk) => {
    process.stderr.write(`[gemma] ${chunk}`);
  });
  gemmaReadline.on("line", handleGemmaLine);
  gemmaProcess.on("exit", (code, signal) => {
    const error = new Error(`Gemma runner stopped${signal ? ` by ${signal}` : ""}${code === null ? "" : ` with code ${code}`}.`);
    for (const pending of gemmaPendingRequests.values()) pending.reject(error);
    gemmaPendingRequests.clear();
    gemmaProcess = null;
    gemmaReadline = null;
    gemmaReady = null;
  });

  gemmaReady = new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error("Timed out while starting the Gemma runner."));
    }, 10 * 60 * 1000);

    gemmaPendingRequests.set("__ready__", {
      resolve: () => {
        clearTimeout(timer);
        resolve();
      },
      reject: (error) => {
        clearTimeout(timer);
        reject(error);
      }
    });
  });

  return gemmaReady;
}

function handleGemmaLine(line) {
  let event;
  try {
    event = JSON.parse(line);
  } catch {
    console.error(`[gemma] ${line}`);
    return;
  }

  if (event.type === "ready") {
    const ready = gemmaPendingRequests.get("__ready__");
    gemmaPendingRequests.delete("__ready__");
    ready?.resolve();
    return;
  }

  if (event.type === "load_error") {
    const ready = gemmaPendingRequests.get("__ready__");
    gemmaPendingRequests.delete("__ready__");
    ready?.reject(new Error([
      event.error || "Gemma runner failed to load.",
      `Python interpreter: ${GEMMA_PYTHON}`,
      "Install the llama.cpp dependencies with `python3 -m pip install -r requirements-vps.txt`, or start with `GEMMA_PYTHON=/path/to/python npm run dev`."
    ].join("\n")));
    return;
  }

  const pending = gemmaPendingRequests.get(String(event.id));
  if (!pending) return;

  if (event.type === "delta") {
    const text = event.text || "";
    pending.output += text;
    pending.onTextDelta?.(text);
    return;
  }

  gemmaPendingRequests.delete(String(event.id));
  if (event.type === "error") {
    pending.reject(new Error(event.error || "Gemma request failed."));
    return;
  }

  if (event.type === "done") {
    pending.resolve(event.text || pending.output);
  }
}

function createTaggedSegmentStreamer({ res, turnId, personas = [], personaName = "" }) {
  const messages = [];
  let buffer = "";
  let rawText = "";
  let current = null;
  let summary = "";

  const openSegment = (type, speaker = null) => {
    if (type === "summary") {
      current = { type, content: "" };
      return;
    }

    const message = type === "dialogue"
      ? makeMessage("persona", "chat", "", turnId, { personaId: speaker.id, personaName: speaker.name })
      : makeMessage("narrator", "narration", "", turnId);
    current = { type, message, speakerName: speaker?.name || personaName };
    messages.push(message);
    sendSse(res, "message", message);
  };

  const emitNarration = (text) => {
    const content = clean(text);
    if (!content) return;
    openSegment("narration");
    appendCurrent(content);
    closeCurrent();
  };

  const appendCurrent = (text) => {
    if (!current || !text) return;
    if (current.type === "summary") {
      current.content += text;
      return;
    }
    if (!current.message) return;
    current.message.content += text;
    if (current.hidden) return;
    sendSse(res, "delta", { id: current.message.id, delta: text });
  };

  const closeCurrent = () => {
    if (!current) return;
    if (current.type === "summary") {
      summary = cleanSummaryContent(current.content) || summary;
      current = null;
      return;
    }
    if (!current.message) {
      current = null;
      return;
    }
    if (current.type === "dialogue") {
      current.message.content = cleanDialogueContent(current.message.content, current.speakerName || personaName);
    }
    if (!current.hidden) sendSse(res, "messageDone", current.message);
    current = null;
  };

  const process = (flush = false) => {
    while (buffer) {
      if (!current) {
        const prefix = findLinePrefix(buffer, personas);
        if (!prefix) {
          if (!flush) return;
          const text = buffer;
          buffer = "";
          emitNarration(text);
          return;
        }

        if (prefix.index > 0) {
          emitNarration(buffer.slice(0, prefix.index));
          buffer = buffer.slice(prefix.index);
          continue;
        }

        buffer = buffer.slice(prefix.index + prefix.length);
        openSegment(prefix.type, prefix.speaker);
        continue;
      }

      const nextPrefix = findLinePrefix(buffer, personas);
      if (nextPrefix) {
        appendCurrent(buffer.slice(0, nextPrefix.index).trimEnd());
        buffer = buffer.slice(nextPrefix.index);
        closeCurrent();
        continue;
      }

      const lineBreakIndex = buffer.indexOf("\n");
      if (lineBreakIndex !== -1) {
        appendCurrent(buffer.slice(0, lineBreakIndex).trimEnd());
        buffer = buffer.slice(lineBreakIndex + 1);
        closeCurrent();
        continue;
      }

      if (flush) {
        appendCurrent(buffer);
        buffer = "";
        closeCurrent();
        return;
      }

      const holdLength = trailingLinePrefixCandidateLength(buffer, personas);
      if (buffer.length <= holdLength) return;
      appendCurrent(buffer.slice(0, buffer.length - holdLength));
      buffer = buffer.slice(buffer.length - holdLength);
      return;
    }

    if (flush && current) closeCurrent();
  };

  return {
    feed(delta) {
      rawText += delta;
      buffer += delta;
      process(false);
    },
    finish() {
      process(true);
      return { messages: messages.filter((message) => clean(message.content)), summary: summary || extractSummaryLine(rawText) };
    }
  };
}

function messagesFromTaggedSequence(sequence, turnId, personaNameOrPersonas = "") {
  return messagesAndSummaryFromTaggedSequence(sequence, turnId, personaNameOrPersonas).messages;
}

function messagesAndSummaryFromTaggedSequence(sequence, turnId, personaNameOrPersonas = "") {
  const personas = Array.isArray(personaNameOrPersonas)
    ? personaNameOrPersonas
    : [{ id: "", name: personaNameOrPersonas || "" }];
  const streamer = createTaggedSegmentStreamer({
    res: { write() {} },
    turnId,
    personas
  });
  streamer.feed(typeof sequence === "string" ? sequence : "");
  return streamer.finish();
}

function extractSummaryLine(text) {
  const value = typeof text === "string" ? text : "";
  const pattern = /(?:^|\n)[ \t]*(?:[-*]\s*)?(?:\*\*Summary\*\*|Summary)\s*:[ \t]*(.*?)(?=\n|$)/gi;
  let match;
  let summary = "";
  while ((match = pattern.exec(value)) !== null) {
    summary = clean(match[1]) || summary;
  }
  return summary;
}

function cleanSummaryContent(content) {
  return clean(content).replace(/^\*+\s*/, "").replace(/\s*\*+$/g, "");
}

function cleanDialogueContent(content, personaName = "") {
  let output = clean(content);
  const escapedName = escapeRegExp(clean(personaName));
  if (!escapedName) return output;

  const attributionPatterns = [
    new RegExp(`^${escapedName}\\s*:\\s*`, "i"),
    new RegExp(`^${escapedName}\\s+(?:says|said|asks|asked|replies|replied|responds|responded)\\s*,?\\s*`, "i")
  ];

  for (const pattern of attributionPatterns) {
    output = output.replace(pattern, "");
  }

  const quoted = output.match(/^["']([\s\S]*)["']$/);
  return clean(quoted ? quoted[1] : output);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function findLinePrefix(value, personas = []) {
  const pattern = /(^|\n)[ \t]*([^:\n]{1,120}):[ \t]*/g;
  let match;
  while ((match = pattern.exec(value)) !== null) {
    const label = clean(match[2]);
    const prefix = linePrefixForLabel(label, personas);
    if (!prefix) continue;
    return {
      ...prefix,
      index: match.index + match[1].length,
      length: pattern.lastIndex - (match.index + match[1].length)
    };
  }
  return null;
}

function linePrefixForLabel(label, personas = []) {
  const normalizedLabel = clean(label).replace(/^[-*]\s+/, "");
  const unboldedLabel = normalizedLabel.toLowerCase();
  if (unboldedLabel === "narration") return { type: "narration" };
  if (unboldedLabel === "summary") return { type: "summary" };

  const boldMatch = normalizedLabel.match(/^\*\*(.*?)\*\*$/);
  if (!boldMatch) return null;

  const normalized = clean(boldMatch[1]).toLowerCase();
  if (normalized === "narration") return { type: "narration" };
  if (normalized === "summary") return { type: "summary" };
  const speaker = (personas || []).find((persona) => clean(persona.name).toLowerCase() === normalized);
  return speaker ? { type: "dialogue", speaker } : null;
}

function trailingLinePrefixCandidateLength(value, personas = []) {
  const lastNewline = value.lastIndexOf("\n");
  const start = lastNewline === -1 ? 0 : lastNewline + 1;
  const suffix = value.slice(start);
  if (!suffix || suffix.includes(":")) return 0;
  const trimmed = suffix.trimStart().toLowerCase();
  if (!trimmed) return suffix.length;
  const labels = ["narration", "summary", "**narration**", "**summary**", ...(personas || []).map((persona) => `**${clean(persona.name).toLowerCase()}**`)];
  return labels.some((label) => label.startsWith(trimmed)) ? suffix.length : 0;
}

export {
  buildPersonaGenerationPrompt,
  buildPersonaReplyPrompt,
  contextBlock,
  createTaggedSegmentStreamer,
  formatMessage,
  gemmaRequestBody,
  gemmaRequestPayload,
  handleOpenAIStreamEvent,
  imageAttachmentsForLlmContext,
  makeMessage,
  messagesAndSummaryFromTaggedSequence,
  messagesFromTaggedSequence,
  normalizeModelSelection,
  openaiEmptyResponseMessage,
  openaiRequestBody,
  parseGeneratedPersonaPayload,
  parseGeneratedPersonas,
  personaGenerationTokenLimit,
  silentFallbackSequence,
  streamTokenLimit,
  sliceLastTurns,
  tokenUsageInputText,
  tokenUsagePayload
};
