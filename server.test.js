import assert from "node:assert/strict";
import test from "node:test";

import {
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
} from "./server.js";

test("past narration remains in recent conversation context", () => {
  const turnId = "turn-1";
  const userMessage = makeMessage("user", "chat", "Meet me at the cafe.", turnId);
  const storyMessages = messagesFromTaggedSequence(
    "**Narration**: Mara arrives at the cafe and spots you near the window.\n**Mara**: I picked the quiet corner for us.",
    turnId,
    "Mara"
  );
  const recentMessages = sliceLastTurns([userMessage, ...storyMessages], 10);
  const context = contextBlock(
    "Mara",
    "Warm and decisive.",
    "John",
    "Optimistic.",
    "Mara:\n- Location: at the cafe\n\nJohn:\n- Location: at the cafe",
    recentMessages
  );

  assert.match(context, /\*\*User\*\*: Meet me at the cafe\./);
  assert.doesNotMatch(context, /^User: Meet me at the cafe\./m);
  assert.doesNotMatch(context, /\[user\/chat\] Meet me at the cafe\./);
  assert.match(context, /\*\*Narration\*\*: Mara arrives at the cafe/);
  assert.match(context, /\*\*Mara\*\*: I picked the quiet corner for us\./);
  assert.doesNotMatch(context, /\[narration\] Mara arrives at the cafe/);
  assert.doesNotMatch(context, /\[chat:Mara\] I picked the quiet corner for us\./);
});

test("context explains who you refers to in each message perspective", () => {
  const context = contextBlock(
    "Mara",
    "Warm and decisive.",
    "John",
    "Optimistic.",
    "Mara:\n- Location: at home\n\nJohn:\n- Location: at home",
    [
      makeMessage("user", "chat", "Can you meet me?", "turn-1"),
      makeMessage("persona", "chat", "I'll meet you there.", "turn-1"),
      makeMessage("narrator", "narration", "You step into the cafe beside Mara.", "turn-1")
    ]
  );

  assert.match(context, /In User: lines, John is speaking/);
  assert.match(context, /"you" and "your" never refer to John and instead refer to Mara/);
  assert.match(context, /In \*\*Mara\*\*: lines, Mara is speaking/);
  assert.match(context, /"you" and "your" refer to John/);
  assert.match(context, /In \*\*Narration\*\*: lines.*"you" and "your" always refer to John/s);
});

test("persona prompt remembers prior consent and waits for user answers", () => {
  const turnId = "turn-2";
  const priorConsent = makeMessage("user", "chat", "Yes, you can hold my hand.", turnId);
  const priorQuestion = makeMessage("persona", "chat", "Do you want to sit outside?", turnId);
  const userMessage = makeMessage("user", "chat", "Let's keep walking.", "turn-3");
  const prompt = buildPersonaReplyPrompt({
    personaName: "Mara",
    persona: "Warm and decisive.",
    userName: "John",
    userPersona: "Optimistic.",
    summary: "Mara:\n- Active consent: John consented to hand-holding\n\nJohn:\n- Active consent: Mara may hold his hand",
    recentMessages: [priorConsent, priorQuestion, userMessage],
    userMessage
  });

  assert.match(prompt, /User pronoun rule: In every User: line, "you", "your", and "yourself" are never John/);
  assert.match(prompt, /They refer to Mara/);
  assert.match(prompt, /"can you move\?" means Mara should move, not John/);
  assert.match(prompt, /When John's latest User: message says "you", "your", or "yourself"/);
  assert.match(prompt, /do not make John perform that referenced action/);
  assert.match(prompt, /The addressed persona should respond or act as the target of that second-person wording/);
});

test("line parser keeps loose text as narration when prefixed output exists", () => {
  const messages = messagesFromTaggedSequence(
    "plain setup\n**Mara**: Hello there.\ntrailing note",
    "turn-plain",
    "Mara"
  );

  assert.equal(messages.length, 3);
  assert.equal(messages[0].role, "narrator");
  assert.equal(messages[0].kind, "narration");
  assert.equal(messages[0].content, "plain setup");
  assert.equal(messages[1].role, "persona");
  assert.equal(messages[1].content, "Hello there.");
  assert.equal(messages[2].role, "narrator");
  assert.equal(messages[2].kind, "narration");
  assert.equal(messages[2].content, "trailing note");
});

test("fully unprefixed model output is parsed as narration", () => {
  const messages = messagesFromTaggedSequence(
    "plain setup with no prefixes",
    "turn-plain",
    "Mara"
  );

  assert.equal(messages.length, 1);
  assert.equal(messages[0].role, "narrator");
  assert.equal(messages[0].kind, "narration");
  assert.equal(messages[0].content, "plain setup with no prefixes");
});

test("unbolded persona labels are parsed as narration while unbolded summary is hidden state", () => {
  const result = messagesAndSummaryFromTaggedSequence(
    "Mara: Hello there.\nSummary: John and Mara spoke.",
    "turn-unbolded-labels",
    "Mara"
  );

  assert.equal(result.messages.length, 1);
  assert.equal(result.messages[0].role, "narrator");
  assert.equal(result.messages[0].kind, "narration");
  assert.equal(result.messages[0].content, "Mara: Hello there.");
  assert.equal(result.summary, "John and Mara spoke.");
});

test("unbolded narration labels are repaired as narration", () => {
  const messages = messagesFromTaggedSequence(
    "Narration: Mara looks at you.\n**Mara**: I saw that.",
    "turn-unbolded-narration",
    "Mara"
  );

  assert.equal(messages.length, 2);
  assert.equal(messages[0].role, "narrator");
  assert.equal(messages[0].kind, "narration");
  assert.equal(messages[0].content, "Mara looks at you.");
  assert.equal(messages[1].role, "persona");
  assert.equal(messages[1].content, "I saw that.");
});

test("unknown model labels are parsed as narration", () => {
  const messages = messagesFromTaggedSequence(
    "System: The room goes quiet.\n**Mara**: I heard that.",
    "turn-unknown-label",
    "Mara"
  );

  assert.equal(messages.length, 2);
  assert.equal(messages[0].role, "narrator");
  assert.equal(messages[0].kind, "narration");
  assert.equal(messages[0].content, "System: The room goes quiet.");
  assert.equal(messages[1].role, "persona");
  assert.equal(messages[1].content, "I heard that.");
});

test("line parser removes persona name attribution from dialogue content", () => {
  const messages = messagesFromTaggedSequence(
    "**Mara**: Mara says 'Hey how are you'\n**Mara**: Mara: Let's go.",
    "turn-dialogue",
    "Mara"
  );

  assert.equal(messages[0].role, "persona");
  assert.equal(messages[0].content, "Hey how are you");
  assert.equal(messages[1].role, "persona");
  assert.equal(messages[1].content, "Let's go.");
});

test("line parser keeps named persona speakers and narration", () => {
  const messages = messagesFromTaggedSequence(
    "**Narration**: Mara looks toward the door.\n**Mara**: Mara: I will check.\n**Nia**: I will wait here.",
    "turn-speakers",
    [
      { id: "p1", name: "Mara" },
      { id: "p2", name: "Nia" }
    ]
  );

  assert.equal(messages.length, 3);
  assert.equal(messages[0].role, "narrator");
  assert.equal(messages[0].content, "Mara looks toward the door.");
  assert.equal(messages[1].personaId, "p1");
  assert.equal(messages[1].personaName, "Mara");
  assert.equal(messages[1].content, "I will check.");
  assert.equal(messages[2].personaId, "p2");
  assert.equal(messages[2].personaName, "Nia");
  assert.equal(messages[2].content, "I will wait here.");
  assert.equal(formatMessage(messages[2]), "**Nia**: I will wait here.");
});

test("persona prompt allows only relevant personas to respond", () => {
  const userMessage = makeMessage("user", "chat", "Emma, can you take a look?", "turn-multi");
  const prompt = buildPersonaReplyPrompt({
    personas: [
      { id: "p1", name: "Emma", gender: "Female", description: "Warm and decisive." },
      { id: "p2", name: "Daniel", gender: "Male", description: "Quiet observer." }
    ],
    scenario: "A stalled train leaves everyone waiting in a storm-dark station.",
    userName: "John",
    userPersona: "Optimistic.",
    summary: "Emma:\n- Location: at home\n\nDaniel:\n- Location: at home\n\nJohn:\n- Location: at home",
    recentMessages: [userMessage],
    userMessage
  });

  assert.match(prompt, /Available personas: Emma, Daniel/);
  assert.match(prompt, /Emma \(Female\):/);
  assert.match(prompt, /Daniel \(Male\):/);
  assert.match(prompt, /Scenario:\nA stalled train leaves everyone waiting in a storm-dark station\./);
  assert.match(prompt, /Use this as the current story premise, setting, and pressure/);
  assert.match(prompt, /Not every persona needs to respond/);
  assert.match(prompt, /\*\*Emma\*\*: spoken dialogue here/);
  assert.match(prompt, /\*\*Daniel\*\*: spoken dialogue here/);
  assert.match(prompt, /If a model line does not start with \*\*Narration\*\*:, a bold available persona-name prefix, or a \*\*Summary\*\*: line, it will be treated as visible narration/);
  assert.match(prompt, /Only \*\*Summary\*\*: lines are hidden from the chat/);
  assert.match(prompt, /unknown labels, and any other non-persona output are shown as narration/);
  assert.match(prompt, /Never write a User:, John:, <user>:, or any other user-prefixed line in the generated response/);
  assert.match(prompt, /Do not speak, act, answer, decide, or continue the conversation on John's behalf/);
  assert.match(prompt, /User pronoun rule: In every User: line, "you", "your", and "yourself" are never John/);
  assert.match(prompt, /They refer to the persona being addressed by name, or the most relevant nearby persona if no name is given/);
  assert.match(prompt, /"Emma, can you move\?" means Emma should move, not John/);
  assert.match(prompt, /Because more than one persona is present, include \*\*Narration\*\*: lines/);
  assert.match(prompt, /Use \*\*Narration\*\*: to connect persona replies so the exchange does not become a bare script/);
  assert.doesNotMatch(prompt, /<dialogue/);
});

test("line parser maps full-name persona prefixes", () => {
  const messages = messagesFromTaggedSequence(
    "**Donald Trump**: Donald Trump: We need to move quickly.",
    "turn-full-name",
    [
      { id: "p1", name: "Donald Trump" }
    ]
  );

  assert.equal(messages.length, 1);
  assert.equal(messages[0].personaId, "p1");
  assert.equal(messages[0].personaName, "Donald Trump");
  assert.equal(messages[0].content, "We need to move quickly.");
});

test("streaming parser keeps full-name persona prefixes", () => {
  const res = {
    write() {}
  };
  const streamer = createTaggedSegmentStreamer({
    res,
    turnId: "turn-stream-full-name",
    personas: [{ id: "p1", name: "Sam Altman" }]
  });

  streamer.feed("**Sam Altman**: Hi there. It's good to meet you.\n");
  streamer.feed("**Summary**: John greeted Sam Altman in a sleek tech hub.");
  const result = streamer.finish();

  assert.equal(result.messages.length, 1);
  assert.equal(result.messages[0].personaId, "p1");
  assert.equal(result.messages[0].personaName, "Sam Altman");
  assert.equal(result.messages[0].content, "Hi there. It's good to meet you.");
  assert.equal(result.summary, "John greeted Sam Altman in a sleek tech hub.");
});

test("streaming parser removes repeated speaker attribution after prefix", () => {
  const res = {
    write() {}
  };
  const streamer = createTaggedSegmentStreamer({
    res,
    turnId: "turn-stream-leading-speaker",
    personas: [{ id: "p1", name: "Sam Altman" }]
  });

  streamer.feed("**Sam Altman**: Sam Altman: Hi there.");
  const result = streamer.finish();

  assert.equal(result.messages.length, 1);
  assert.equal(result.messages[0].role, "persona");
  assert.equal(result.messages[0].personaName, "Sam Altman");
  assert.equal(result.messages[0].content, "Hi there.");
});

test("streaming parser keeps unprefixed text as narration", () => {
  const taggedRes = {
    write() {}
  };
  const taggedStreamer = createTaggedSegmentStreamer({
    res: taggedRes,
    turnId: "turn-stream-loose-text",
    personas: [{ id: "p1", name: "Mara" }]
  });

  taggedStreamer.feed("plain setup\n");
  taggedStreamer.feed("**Mara**: Hello there.\ntrailing note");
  const taggedResult = taggedStreamer.finish();

  assert.equal(taggedResult.messages.length, 3);
  assert.equal(taggedResult.messages[0].role, "narrator");
  assert.equal(taggedResult.messages[0].content, "plain setup");
  assert.equal(taggedResult.messages[1].role, "persona");
  assert.equal(taggedResult.messages[1].content, "Hello there.");
  assert.equal(taggedResult.messages[2].role, "narrator");
  assert.equal(taggedResult.messages[2].content, "trailing note");

  const untaggedRes = {
    write() {}
  };
  const untaggedStreamer = createTaggedSegmentStreamer({
    res: untaggedRes,
    turnId: "turn-stream-no-tags",
    personas: [{ id: "p1", name: "Mara" }]
  });

  untaggedStreamer.feed("plain setup ");
  untaggedStreamer.feed("with no prefixes");
  const untaggedResult = untaggedStreamer.finish();

  assert.equal(untaggedResult.messages.length, 1);
  assert.equal(untaggedResult.messages[0].role, "narrator");
  assert.equal(untaggedResult.messages[0].kind, "narration");
  assert.equal(untaggedResult.messages[0].content, "plain setup with no prefixes");
});

test("streaming parser handles narration, dialogue, and summary prefixes", () => {
  const res = {
    write() {}
  };
  const streamer = createTaggedSegmentStreamer({
    res,
    turnId: "turn-stream-square-tags",
    personas: [{ id: "p1", name: "Sam Altman" }]
  });

  streamer.feed("**Narration**: Sam looks up.\n**Sam Altman**: Hi there.\n**Summ");
  streamer.feed("ary**: John greeted Sam Altman.");
  const result = streamer.finish();

  assert.equal(result.messages.length, 2);
  assert.equal(result.messages[0].role, "narrator");
  assert.equal(result.messages[0].content, "Sam looks up.");
  assert.equal(result.messages[1].role, "persona");
  assert.equal(result.messages[1].personaName, "Sam Altman");
  assert.equal(result.messages[1].content, "Hi there.");
  assert.equal(result.summary, "John greeted Sam Altman.");
});

test("streaming parser handles narration between multiple personas", () => {
  const streamer = createTaggedSegmentStreamer({
    res: { write() {} },
    turnId: "turn-stream-multi-narration",
    personas: [
      { id: "p1", name: "Emma" },
      { id: "p2", name: "Daniel" }
    ]
  });

  streamer.feed("**Narration**: Emma glances toward Daniel before answering.\n");
  streamer.feed("**Emma**: I'll check the hallway.\n**Narration**: Daniel shifts closer to the elevator panel.\n");
  streamer.feed("**Daniel**: Give me a second.\n**Summary**: Emma and Daniel coordinate while John waits nearby.");
  const result = streamer.finish();

  assert.equal(result.messages.length, 4);
  assert.equal(result.messages[0].role, "narrator");
  assert.equal(result.messages[0].content, "Emma glances toward Daniel before answering.");
  assert.equal(result.messages[1].personaId, "p1");
  assert.equal(result.messages[1].personaName, "Emma");
  assert.equal(result.messages[1].content, "I'll check the hallway.");
  assert.equal(result.messages[2].role, "narrator");
  assert.equal(result.messages[2].content, "Daniel shifts closer to the elevator panel.");
  assert.equal(result.messages[3].personaId, "p2");
  assert.equal(result.messages[3].personaName, "Daniel");
  assert.equal(result.messages[3].content, "Give me a second.");
  assert.equal(result.summary, "Emma and Daniel coordinate while John waits nearby.");
});

test("streaming parser extracts formatted summary lines", () => {
  const bulletStreamer = createTaggedSegmentStreamer({
    res: { write() {} },
    turnId: "turn-stream-bullet-summary",
    personas: [{ id: "p1", name: "Mara" }]
  });

  bulletStreamer.feed("**Mara**: Let's go.\n- **Summary**: John and Mara decide to go outside.");
  const bulletResult = bulletStreamer.finish();

  assert.equal(bulletResult.messages.length, 1);
  assert.equal(bulletResult.messages[0].content, "Let's go.");
  assert.equal(bulletResult.summary, "John and Mara decide to go outside.");

  const markdownStreamer = createTaggedSegmentStreamer({
    res: { write() {} },
    turnId: "turn-stream-markdown-summary",
    personas: [{ id: "p1", name: "Mara" }]
  });

  markdownStreamer.feed("**Mara**: Good.\n**Summary**: John and Mara agree to continue.");
  const markdownResult = markdownStreamer.finish();

  assert.equal(markdownResult.messages.length, 1);
  assert.equal(markdownResult.messages[0].content, "Good.");
  assert.equal(markdownResult.summary, "John and Mara agree to continue.");

  const unboldedStreamer = createTaggedSegmentStreamer({
    res: { write() {} },
    turnId: "turn-stream-unbolded-summary",
    personas: [{ id: "p1", name: "Mara" }]
  });

  unboldedStreamer.feed("**Mara**: Fine.\nSummary: John and Mara pause.");
  const unboldedResult = unboldedStreamer.finish();

  assert.equal(unboldedResult.messages.length, 1);
  assert.equal(unboldedResult.messages[0].content, "Fine.");
  assert.equal(unboldedResult.summary, "John and Mara pause.");
});

test("streaming parser closes the final prefixed line on finish", () => {
  const res = {
    write() {}
  };
  const streamer = createTaggedSegmentStreamer({
    res,
    turnId: "turn-stream-unclosed-tag",
    personas: [{ id: "p1", name: "Mara" }]
  });

  streamer.feed("**Mara**: Mara: We should go.");
  const result = streamer.finish();

  assert.equal(result.messages.length, 1);
  assert.equal(result.messages[0].role, "persona");
  assert.equal(result.messages[0].personaName, "Mara");
  assert.equal(result.messages[0].content, "We should go.");
});

test("persona generation prompt asks for scenario-fit common-name personas", () => {
  const prompt = buildPersonaGenerationPrompt({
    scenario: "A hotel elevator stops between floors during a blackout.",
    userName: "John",
    userPersona: "Optimistic."
  });

  assert.match(prompt, /Generate personas for a roleplay chat scene/);
  assert.match(prompt, /A hotel elevator stops between floors during a blackout/);
  assert.match(prompt, /"you", "your", "I", "me", and "my" refer to the user, \w+/);
  assert.match(prompt, /Do not generate a persona for those pronouns/);
  assert.match(prompt, /Create the smallest number of personas needed/);
  assert.match(prompt, /up to a hard maximum of 10 personas/);
  assert.match(prompt, /Usually create 1 or 2 personas/);
  assert.match(prompt, /Create more only when the scenario clearly needs that many distinct people/);
  assert.match(prompt, /Never create extra background, duplicate, decorative, or nice-to-have personas/);
  assert.match(prompt, /If the scenario mentions a person's name, generate that named person as a persona/);
  assert.match(prompt, /Preserve the mentioned name exactly, including the full name when given/);
  assert.match(prompt, /Never name a persona Narration because \*\*Narration\*\*: is a reserved prose prefix/);
  assert.match(prompt, /name that persona Narration_Persona instead/);
  assert.match(prompt, /use Donald Trump instead of Donald if the scenario says Donald Trump/);
  assert.match(prompt, /Do not create side characters, bystanders, staff, relatives, friends, rivals, witnesses, or other extra people unless the scenario explicitly asks for them/);
  assert.doesNotMatch(prompt, /Create 2 to 5 personas/);
  assert.match(prompt, /Use common modern English first names/);
  assert.match(prompt, /Each gender must be exactly Male or Female/);
  assert.match(prompt, /Return strict JSON only/);
  assert.match(prompt, /"summary"/);
  assert.match(prompt, /"personas"/);
  assert.match(prompt, /Also generate a concise summary of the starting situation/);
  assert.match(prompt, /Do not include any persona field except name, gender, and description/);
  assert.doesNotMatch(prompt, /userKeyDetails/);
  assert.doesNotMatch(prompt, /keyDetails/);
});

test("generated persona parser accepts fenced object JSON and normalizes gender", () => {
  const payload = parseGeneratedPersonaPayload(`\`\`\`json
{
  "summary": "John, Sarah, and Mark are stuck in a hotel elevator during a blackout while waiting for help.",
  "personas": [
    {
      "name": "Sarah",
      "gender": "female",
      "description": "A cautious hotel manager who knows the building and hates losing control."
    },
    {
      "name": "Mark",
      "gender": "MALE",
      "description": "A tired electrician with a dry voice and useful knowledge of the outage."
    }
  ]
}
\`\`\``, "John");
  const personas = payload.personas;

  assert.match(payload.summary, /stuck in a hotel elevator/);
  assert.equal(personas.length, 2);
  assert.equal(personas[0].name, "Sarah");
  assert.equal(personas[0].gender, "Female");
  assert.equal("keyDetails" in personas[0], false);
  assert.equal(personas[1].name, "Mark");
  assert.equal(personas[1].gender, "Male");
});

test("generated persona payload falls back to scenario-ready user summary", () => {
  const payload = parseGeneratedPersonaPayload(`{"personas":[{"name":"Sarah","gender":"Female","description":"A cautious hotel manager."}]}`, "Alex");

  assert.equal(payload.personas.length, 1);
  assert.match(payload.summary, /Alex is present in the scenario/);
  assert.match(payload.summary, /conversation has just begun/);
});

test("generated persona parser reserves Narration as a system label", () => {
  const payload = parseGeneratedPersonaPayload(`{"personas":[{"name":"Narration","gender":"Female","description":"A person whose given name would collide with the narration prefix."}]}`, "Alex");

  assert.equal(payload.personas.length, 1);
  assert.equal(payload.personas[0].name, "Narration_Persona");
});

test("persona prompt renames submitted Narration personas", () => {
  const userMessage = makeMessage("user", "chat", "Say something.", "turn-reserved-name");
  const prompt = buildPersonaReplyPrompt({
    personas: [
      { id: "p1", name: "Narration", gender: "Female", description: "Reserved-name collision." }
    ],
    userName: "Alex",
    userPersona: "Curious.",
    summary: "The conversation has just begun.",
    recentMessages: [],
    userMessage
  });

  assert.match(prompt, /Available personas: Narration_Persona/);
  assert.match(prompt, /\*\*Narration_Persona\*\*: spoken dialogue here/);
  assert.doesNotMatch(prompt, /Available personas: Narration\b/);
});

test("persona prompt always requests narration even if a caller passes a display toggle", () => {
  const userMessage = makeMessage("user", "chat", "Come here.", "turn-narration-display-off");
  const prompt = buildPersonaReplyPrompt({
    personaName: "Mara",
    persona: "Guarded and direct.",
    userName: "John",
    userPersona: "Optimistic.",
    summary: "Mara:\n- Location: at home\n\nJohn:\n- Location: at home",
    recentMessages: [userMessage],
    userMessage
  });

  assert.match(prompt, /relevant personas only/);
  assert.match(prompt, /Use \*\*Narration\*\*: for prose/);
  assert.match(prompt, /Narration should be cinematic/);
  assert.match(prompt, /immerse the user in the scene without overexplaining/);
  assert.match(prompt, /When narration refers to John, write it in second person as "you", "your", or "yourself"/);
  assert.match(prompt, /Do not use John, "the user", "the player", "John", or any other third-person user label in \*\*Narration\*\*:/);
  assert.match(prompt, /Use John only in persona dialogue or the hidden \*\*Summary\*\*: line/);
  assert.doesNotMatch(prompt, /immerse John in the scene/);
  assert.match(prompt, /Do not describe a persona's inner thoughts, emotional state, facial expression, body language, or motivation inside that persona's dialogue line/);
  assert.match(prompt, /Put those details in \*\*Narration\*\*: lines before or after the spoken line/);
  assert.doesNotMatch(prompt, /Because more than one persona is present/);
  assert.doesNotMatch(prompt, /using dialogue only/);
  assert.doesNotMatch(prompt, /Do not use <nar> tags/);
});

test("empty persona fallback preserves silence instead of forced agreement", () => {
  assert.equal(silentFallbackSequence("Mara"), "**Narration**: Mara does not answer.");
});

test("parser always keeps narration as normal conversation context", () => {
  const messages = messagesFromTaggedSequence(
    "**Narration**: Mara looks away.\n**Mara**: Not now.",
    "turn-no-narration",
    "Mara"
  );

  assert.equal(messages.length, 2);
  assert.equal(messages[0].role, "narrator");
  assert.equal(messages[0].kind, "narration");
  assert.equal(messages[0].content, "Mara looks away.");
  assert.equal(formatMessage(messages[0]), "**Narration**: Mara looks away.");
  assert.equal(messages[1].role, "persona");
  assert.equal(messages[1].content, "Not now.");
});

test("persona prompt requires summary in the same model response", () => {
  const userMessage = makeMessage("user", "chat", "Let's go outside.", "turn-1");
  const prompt = buildPersonaReplyPrompt({
    personaName: "Mara",
    persona: "Warm and decisive.",
    userName: "John",
    userPersona: "Optimistic.",
    summary: "Mara:\n- Location: at home\n\nJohn:\n- Location: at home",
    recentMessages: [userMessage],
    userMessage
  });

  assert.match(prompt, /After all visible narration and persona dialogue for this turn, return exactly one \*\*Summary\*\*: line/);
  assert.match(prompt, /The \*\*Summary\*\*: line must be a detailed standalone continuity memory that can replace older conversation messages/);
  assert.match(prompt, /A future response should be possible using only this summary, the persona descriptions, and the next User: line or \[state\/action\] line/);
  assert.match(prompt, /Update the prior summary after this same turn/);
  assert.match(prompt, /Carry forward every durable fact still needed for continuity/);
  assert.match(prompt, /current location, who is present, what each person is doing or holding, visible images or objects that still matter/);
  assert.match(prompt, /relationship dynamics, emotional state, promises, decisions, conflicts, consent, boundaries, unresolved questions, active plans/);
  assert.match(prompt, /Then add the latest user message\/action, persona dialogue, and narration from this turn with concrete cause and effect/);
  assert.match(prompt, /another model could continue the scene naturally without seeing the older transcript/);
  assert.match(prompt, /Do not write only the latest beat, a vague current-state note, or a short recap/);
  assert.match(prompt, /dense, specific, and multi-sentence when needed rather than compressed into one generic sentence/);
  assert.doesNotMatch(prompt, /Summary update cadence:/);
  assert.doesNotMatch(prompt, /separate summary update/);
});

test("summary lines are parsed as hidden state, not visible messages", () => {
  const messages = messagesFromTaggedSequence(
    "**Mara**: Let's go.\n**Summary**: John and Mara decide to go outside.",
    "turn-summary",
    "Mara"
  );

  assert.equal(messages.length, 1);
  assert.equal(messages[0].role, "persona");
  assert.equal(messages[0].content, "Let's go.");
});

test("sliceLastTurns keeps the latest turn groups", () => {
  const messages = Array.from({ length: 15 }, (_, index) => (
    makeMessage("user", "chat", `turn ${index + 1}`, `turn-${index + 1}`)
  ));
  const recentMessages = sliceLastTurns(messages, 13);

  assert.equal(recentMessages.length, 13);
  assert.equal(recentMessages[0].content, "turn 3");
  assert.equal(recentMessages.at(-1).content, "turn 15");
});

test("sliceLastTurns retains image data for kept turn groups only", () => {
  const messages = Array.from({ length: 12 }, (_, index) => (
    makeMessage("user", "chat", `turn ${index + 1}`, `turn-${index + 1}`, {
      imageAttachments: [{
        name: `turn-${index + 1}.png`,
        type: "image/png",
        size: 4,
        dataUrl: `data:image/png;base64,${index + 1}`
      }]
    })
  ));
  const recentMessages = sliceLastTurns(messages, 10);

  assert.equal(recentMessages.length, 10);
  assert.equal(recentMessages[0].content, "turn 3");
  assert.equal(recentMessages[0].imageAttachments[0].dataUrl, "data:image/png;base64,3");
  assert.equal(recentMessages.at(-1).imageAttachments[0].dataUrl, "data:image/png;base64,12");
});

test("LLM image context prioritizes current images before retained prior images", () => {
  const currentMessage = makeMessage("user", "chat", "current image", "turn-2", {
    imageAttachments: [
      { id: "current-image", dataUrl: "data:image/jpeg;base64,BBBB" }
    ]
  });
  const images = imageAttachmentsForLlmContext([
    makeMessage("user", "chat", "previous image", "turn-1", {
      imageAttachments: [
        { id: "previous-image", dataUrl: "data:image/png;base64,AAAA" },
        { dataUrl: "not-an-image" }
      ]
    }),
    makeMessage("persona", "chat", "I can see it.", "turn-1"),
    currentMessage
  ], currentMessage);

  assert.deepEqual(images.map((image) => image.dataUrl), [
    "data:image/jpeg;base64,BBBB",
    "data:image/png;base64,AAAA"
  ]);
});

test("Gemma request body sends local runner generation options only", () => {
  const body = gemmaRequestBody({
    input: "hello",
    maxOutputTokens: 50,
    stream: true,
    clientIp: "127.0.0.1"
  });

  assert.equal("model" in body, false);
  assert.equal(body.client_ip, "127.0.0.1");
  assert.equal(body.stream, true);
  assert.equal(body.max_new_tokens, 50);
  assert.deepEqual(Object.keys(body).sort(), [
    "client_ip",
    "image_urls",
    "input",
    "max_new_tokens",
    "stream",
    "temperature",
    "top_k",
    "top_p"
  ]);
});

test("Gemma request body uses recommended sampling defaults", () => {
  const body = gemmaRequestBody({
    input: "hello",
    maxOutputTokens: 50
  });

  assert.equal(body.temperature, 1);
  assert.equal(body.top_p, 0.95);
  assert.equal(body.top_k, 64);
});

test("Gemma request body sends uploaded image data URLs to the runner", () => {
  const body = gemmaRequestBody({
    input: "look",
    maxOutputTokens: 50,
    imageAttachments: [
      { dataUrl: "data:image/png;base64,AAAA" },
      { dataUrl: "not-an-image" },
      {}
    ]
  });

  assert.deepEqual(body.image_urls, ["data:image/png;base64,AAAA"]);
});

test("token usage input text hides image payloads behind placeholders", () => {
  const inputText = tokenUsageInputText("Prompt text", [
    { dataUrl: "data:image/png;base64,AAAA" },
    { dataUrl: "not-an-image" },
    { dataUrl: "data:image/jpeg;base64,BBBB" }
  ]);

  assert.equal(inputText, "Prompt text\n[image inputs]\n<image>\n<image>");
  assert.doesNotMatch(inputText, /base64/);
});

test("token usage payload includes counts and input output text", () => {
  const payload = tokenUsagePayload({
    turnId: "turn-1",
    inputText: "Look at <image>.",
    outputText: "**Narration**: You look closer."
  });

  assert.equal(payload.turnId, "turn-1");
  assert.equal(payload.inputText, "Look at <image>.");
  assert.equal(payload.outputText, "**Narration**: You look closer.");
  assert.equal(payload.inputTokens > 0, true);
  assert.equal(payload.outputTokens > 0, true);
});

test("Gemma stdin payload keeps uploaded images for the Python runner", () => {
  const payload = gemmaRequestPayload({
    id: "request-1",
    input: "look",
    maxOutputTokens: 50,
    stream: true,
    clientIp: "127.0.0.1",
    imageAttachments: [
      { dataUrl: "data:image/jpeg;base64,BBBB" }
    ]
  });

  assert.equal(payload.id, "request-1");
  assert.equal(payload.input, "look");
  assert.equal(payload.stream, true);
  assert.deepEqual(payload.image_urls, ["data:image/jpeg;base64,BBBB"]);
});

test("model selection defaults to Gemma and accepts GPT 5 OpenAI models", () => {
  assert.equal(normalizeModelSelection(undefined), "gemma-4");
  assert.equal(normalizeModelSelection("unknown"), "gemma-4");
  assert.equal(normalizeModelSelection("gpt-5-mini"), "gpt-5-mini");
  assert.equal(normalizeModelSelection("gpt-5-nano"), "gpt-5-nano");
});

test("OpenAI request body uses selected model without local rate-limit fields", () => {
  const body = openaiRequestBody({
    input: "hello",
    maxOutputTokens: 50,
    stream: true,
    model: "gpt-5-nano"
  });

  assert.deepEqual(body, {
    model: "gpt-5-nano",
    input: "hello",
    max_output_tokens: 50,
    stream: true,
    reasoning: {
      effort: "minimal"
    }
  });
  assert.equal("client_ip" in body, false);
  assert.equal("clientIp" in body, false);
  assert.equal("per_ip_min_turn_interval_ms" in body, false);
  assert.equal("per_ip_max_queued_turns" in body, false);
});

test("OpenAI request body sends images as multimodal input parts", () => {
  const body = openaiRequestBody({
    input: "Describe this.",
    maxOutputTokens: 50,
    model: "gpt-5-mini",
    imageAttachments: [{
      name: "scene.png",
      type: "image/png",
      size: 1234,
      dataUrl: "data:image/png;base64,AAAA"
    }]
  });

  assert.deepEqual(body.input, [{
    role: "user",
    content: [
      { type: "input_text", text: "Describe this." },
      { type: "input_image", image_url: "data:image/png;base64,AAAA" }
    ]
  }]);
});

test("formatMessage includes uploaded image context", () => {
  const message = makeMessage("user", "chat", "", "turn-image", {
    imageAttachments: [{
      name: "room.webp",
      type: "image/webp",
      size: 2048
    }]
  });

  const formatted = formatMessage(message);

  assert.match(formatted, /^User: \[uploaded images\]/);
  assert.match(formatted, /room\.webp \(image\/webp, 2 KB\)/);
  assert.match(formatted, /prior uploaded image retained as transcript metadata/);
});

test("formatMessage marks current-turn images as the focus", () => {
  const message = makeMessage("user", "chat", "Look at this one.", "turn-image", {
    imageAttachments: [{
      name: "new-room.png",
      type: "image/png",
      size: 1024,
      dataUrl: "data:image/png;base64,AAAA"
    }]
  });

  const formatted = formatMessage(message, { focusImages: true });

  assert.match(formatted, /^User: Look at this one\.\n\[latest uploaded images - focus on these for this turn\]/);
  assert.match(formatted, /new-room\.png \(image\/png, 1 KB\)/);
});

test("OpenAI empty response errors explain incomplete output", () => {
  const message = openaiEmptyResponseMessage({
    status: "incomplete",
    incomplete_details: { reason: "max_output_tokens" }
  });

  assert.match(message, /incomplete response before producing visible text/);
  assert.match(message, /max_output_tokens/);
  assert.match(message, /OPENAI_PERSONA_GENERATION_TOKENS/);
});

test("OpenAI stream events emit output text deltas", () => {
  const deltas = [];
  const output = handleOpenAIStreamEvent(
    [
      "event: response.output_text.delta",
      "data: {\"type\":\"response.output_text.delta\",\"delta\":\"**Narration**: Mara\"}",
      "",
      "event: response.output_text.delta",
      "data: {\"type\":\"response.output_text.delta\",\"delta\":\" looks up.\"}"
    ].join("\n"),
    (delta) => deltas.push(delta)
  );

  assert.equal(output, "**Narration**: Mara looks up.");
  assert.deepEqual(deltas, ["**Narration**: Mara", " looks up."]);
});

test("model-specific token limits use separate environment knobs", () => {
  const previousOpenAiPersonaTokens = process.env.OPENAI_PERSONA_GENERATION_TOKENS;
  const previousOpenAiStreamTokens = process.env.OPENAI_STREAM_TOKENS;
  const previousGemmaPersonaTokens = process.env.GEMMA_PERSONA_GENERATION_TOKENS;
  const previousGemmaStreamTokens = process.env.GEMMA_STREAM_TOKENS;

  process.env.OPENAI_PERSONA_GENERATION_TOKENS = "123";
  process.env.OPENAI_STREAM_TOKENS = "456";
  process.env.GEMMA_PERSONA_GENERATION_TOKENS = "789";
  process.env.GEMMA_STREAM_TOKENS = "987";

  try {
    assert.equal(personaGenerationTokenLimit("gpt-5-mini"), 123);
    assert.equal(streamTokenLimit("gpt-5-mini"), 456);
    assert.equal(personaGenerationTokenLimit("gpt-5-nano"), 123);
    assert.equal(streamTokenLimit("gpt-5-nano"), 456);
    assert.equal(personaGenerationTokenLimit("gemma-4"), 789);
    assert.equal(streamTokenLimit("gemma-4"), 987);
  } finally {
    restoreEnv("OPENAI_PERSONA_GENERATION_TOKENS", previousOpenAiPersonaTokens);
    restoreEnv("OPENAI_STREAM_TOKENS", previousOpenAiStreamTokens);
    restoreEnv("GEMMA_PERSONA_GENERATION_TOKENS", previousGemmaPersonaTokens);
    restoreEnv("GEMMA_STREAM_TOKENS", previousGemmaStreamTokens);
  }
});

test("OpenAI persona generation uses a larger default output budget", () => {
  const previousOpenAiPersonaTokens = process.env.OPENAI_PERSONA_GENERATION_TOKENS;
  delete process.env.OPENAI_PERSONA_GENERATION_TOKENS;

  try {
    assert.equal(personaGenerationTokenLimit("gpt-5-mini"), 4000);
    assert.equal(personaGenerationTokenLimit("gpt-5-nano"), 4000);
  } finally {
    restoreEnv("OPENAI_PERSONA_GENERATION_TOKENS", previousOpenAiPersonaTokens);
  }
});

function restoreEnv(key, value) {
  if (value === undefined) {
    delete process.env[key];
    return;
  }
  process.env[key] = value;
}
