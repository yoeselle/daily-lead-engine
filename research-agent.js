#!/usr/bin/env node
/**
 * Daily Lead Engine — Research Agent v2
 * ----------------------------------------
 * Runs real, web-verified lead research via Claude API + web search.
 * Outputs runs.js which is loaded automatically by index.html.
 *
 * Setup:
 *   1.  npm install
 *   2.  cp profile.example.json profile.json  &&  edit profile.json
 *   3.  export ANTHROPIC_API_KEY=sk-ant-...
 *   4.  node research-agent.js
 *
 * Schedule daily:
 *   crontab -e  →  0 6 * * *  cd /path/to/daily-lead-engine && node research-agent.js
 */

'use strict';

const Anthropic = require('@anthropic-ai/sdk');
const fs = require('fs');
const path = require('path');

// ─── Models & config ─────────────────────────────────────────────────────────

const DISCOVERY_MODEL = 'claude-sonnet-4-6';
const DEEP_MODEL       = 'claude-sonnet-4-6';   // swap to claude-opus-4-7 for higher quality
const GEO_MIX          = { local: 16, interstate: 6, international: 3 };
const DEEP_DIVE_COUNT  = 10;
const MAX_HISTORY_RUNS = 30;
const DIR              = __dirname;

// ─── Anthropic client ────────────────────────────────────────────────────────

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  defaultHeaders: { 'anthropic-beta': 'web-search-2025-03-05' },
});

const WEB_SEARCH_TOOL = { type: 'web_search_20250305', name: 'web_search' };

// ─── Utilities ───────────────────────────────────────────────────────────────

const sleep = ms => new Promise(r => setTimeout(r, ms));

function parseJSON(text) {
  const attempts = [];

  const cb = text.match(/```(?:json)?\s*\n([\s\S]+?)\n?```/);
  if (cb) attempts.push(cb[1]);

  const arr = text.match(/(\[\s*\{[\s\S]*\}\s*\])/);
  if (arr) attempts.push(arr[1]);

  const obj = text.match(/(\{\s*"(?:rank|name)"[\s\S]*\})/);
  if (obj) attempts.push(obj[1]);

  attempts.push(text.trim());

  for (const raw of attempts) {
    try {
      const fixed = raw.replace(/,(\s*[}\]])/g, '$1');   // strip trailing commas
      return JSON.parse(fixed);
    } catch { /* try next */ }
  }
  throw new Error('No valid JSON found in response');
}

function loadProfile() {
  const p = path.join(DIR, 'profile.json');
  if (!fs.existsSync(p)) {
    console.error('\n❌  profile.json not found.');
    console.error('   Run:  cp profile.example.json profile.json  and fill in your details.\n');
    process.exit(1);
  }
  const profile = JSON.parse(fs.readFileSync(p, 'utf8'));
  if (!profile.businessName) {
    console.error('❌  profile.json is missing the "businessName" field.\n');
    process.exit(1);
  }
  return profile;
}

function loadExistingRuns() {
  const p = path.join(DIR, 'runs.js');
  if (!fs.existsSync(p)) return [];
  try {
    const raw = fs.readFileSync(p, 'utf8')
      .replace(/^\s*window\.RUNS\s*=\s*/, '')
      .replace(/;\s*$/, '');
    return JSON.parse(raw);
  } catch { return []; }
}

function saveRuns(runs) {
  const js = 'window.RUNS = ' + JSON.stringify(runs, null, 2) + ';\n';
  fs.writeFileSync(path.join(DIR, 'runs.js'), js, 'utf8');
}

// ─── Claude agentic loop (handles multi-turn web search) ─────────────────────

async function runClaude(model, systemPrompt, userPrompt, maxTokens = 8192) {
  const messages = [{ role: 'user', content: userPrompt }];
  let finalText = '';

  for (let turn = 0; turn < 12; turn++) {
    const response = await client.messages.create({
      model,
      max_tokens: maxTokens,
      system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }],
      tools: [WEB_SEARCH_TOOL],
      messages,
    });

    for (const block of response.content) {
      if (block.type === 'text') finalText += block.text;
    }

    if (response.stop_reason === 'end_turn') break;

    if (response.stop_reason === 'tool_use') {
      messages.push({ role: 'assistant', content: response.content });
      const acks = response.content
        .filter(b => b.type === 'tool_use')
        .map(b => ({ type: 'tool_result', tool_use_id: b.id, content: '' }));
      if (acks.length) messages.push({ role: 'user', content: acks });
    } else {
      break;
    }
  }

  return finalText;
}

// ─── Phase 1 — Lead discovery ─────────────────────────────────────────────────

async function discoverLeads(profile, location, geo, count) {
  process.stdout.write(`  → ${location} (${geo}) — searching for ${count}...`);

  const system = `You are a lead researcher for ${profile.businessName} (${profile.services}).
They serve: ${profile.whoYouServe}. Based in: ${profile.location}.

CRITICAL RULES — NO EXCEPTIONS:
1. Use web_search for EVERY business you include — never rely on training data
2. Verify each business exists: search "[name] [city]" and confirm a real web result
3. Find their Instagram: search "[name] [city] instagram" — only include if you find it
4. Get Google rating/reviews: search "[name] [city] reviews" — only include real numbers
5. If you cannot verify a field, set it to null — NEVER GUESS OR INVENT DATA
6. Skip any business you cannot confirm exists via search`;

  const user = `Use web_search to find ${count} real, existing businesses in ${location} that would benefit from: ${profile.services}.

Search in these verticals: ${profile.targetVerticals.join(', ')}

For every business you include, you MUST:
a) Search "[business name] [suburb/city]" to confirm it exists
b) Search "[business name] instagram [city]" to find their IG handle
c) Search "[business name] google reviews" to get rating and count

Return ONLY a JSON array. Each object:
{
  "name": "Exact Business Name (as it appears online)",
  "vertical": "Hospitality | Fitness | Beauty | Retail | Health & Wellness",
  "location": "Suburb or area",
  "region": "${location}",
  "geo": "${geo}",
  "website": "domain.com (verified live, or null)",
  "instagram": "@handle (verified exists, or null)",
  "facebook": "Page name (or null)",
  "tiktok": "@handle (or null)",
  "gbpRating": 4.8,
  "gbpReviews": 240,
  "venueCount": 1,
  "staffEstimate": "~15 staff",
  "presenceScore": 2,
  "fitScore": 78,
  "tag": "3–5 word gap summary",
  "why": "One sentence why they need ${profile.services}",
  "searchedUrls": ["url1", "url2"]
}

fitScore 0–100: 80+ = strong gap + clear ability to pay; 60–79 = good fit; 40–59 = lower priority
presenceScore 1–5: 1=none, 2=minimal, 3=basic, 4=good, 5=strong

Only include verified, real businesses. Do not invent any.`;

  try {
    const text = await runClaude(DISCOVERY_MODEL, system, user, 6000);
    const leads = parseJSON(text);
    const arr = Array.isArray(leads) ? leads : [];
    console.log(` ✓ ${arr.length} found`);
    return arr;
  } catch (e) {
    console.log(` ⚠️  failed (${e.message.slice(0, 60)})`);
    return [];
  }
}

// ─── Phase 3 — Deep dive dossier ─────────────────────────────────────────────

async function deepDive(lead, profile, rank) {
  process.stdout.write(`  → #${rank} ${lead.name}...`);

  const today = new Date().toLocaleDateString('en-AU', {
    day: 'numeric', month: 'long', year: 'numeric',
  });

  const system = `You are a senior business intelligence analyst writing a full client dossier for ${profile.businessName}.
${profile.businessName} offers: ${profile.services}
Proof point: ${profile.proofPoint || 'N/A'}
Pricing: ${profile.pricing || 'N/A'}

NON-NEGOTIABLE RULES:
- Search the web for every single field — do not fill from training memory
- Social followers/counts: search the platform handle and read the actual number
- Competitors: search "[vertical] [city]" — only name businesses you find in results
- News/signals: search "[business name] news 2025 2026" — only include real results
- Mark anything you cannot verify as "not found" or null
- Never invent review numbers, follower counts, or contact details`;

  const user = `Write a full research dossier on ${lead.name} (${lead.vertical}, ${lead.location}, ${lead.region}).

Already known: ${JSON.stringify({ website: lead.website, instagram: lead.instagram, gbpRating: lead.gbpRating, gbpReviews: lead.gbpReviews })}

Search the web to find everything listed below, then return a single JSON object:
{
  "rank": ${rank},
  "name": "${lead.name}",
  "vertical": "${lead.vertical}",
  "location": "${lead.location}",
  "region": "${lead.region}",
  "geo": "${lead.geo}",
  "size": "X venue(s) · ~Y staff",
  "presence": ${lead.presenceScore || 2},
  "tag": "${lead.tag}",
  "fit": ${lead.fitScore || 75},
  "tier": "A",
  "deep": true,
  "founded": "Est. YYYY",
  "priceTier": "$$ (use $ /$$ /$$$ based on menu/pricing found)",
  "overview": "2–3 sentences from your research. Real facts only.",
  "why": "One specific sentence why they need ${profile.services}.",
  "web": {
    "url": "domain found via search",
    "status": "Live · mobile-optimised | Live · not mobile-optimised | not found",
    "note": "What the site does/lacks based on search results"
  },
  "socials": [
    {
      "p": "Instagram",
      "handle": "@handle (search '[name] instagram ${lead.location}' to verify) or 'not found'",
      "followers": "X,XXX (search and read actual count) or '—'",
      "activity": "posts X/week | monthly | rarely",
      "note": "What they post and what's missing"
    },
    {
      "p": "Facebook",
      "handle": "Page name or 'not found'",
      "followers": "X,XXX or '—'",
      "activity": "active | inactive | running ads",
      "note": "What's notable"
    },
    {
      "p": "TikTok",
      "handle": "@handle or 'not found'",
      "followers": "X,XXX or '—'",
      "activity": "posts X/week | rarely | none",
      "note": "Content type or 'No account found'"
    }
  ],
  "gbp": {
    "rating": "X.X",
    "reviews": "XXX+",
    "sentiment": "What reviewers say, based on what you found in your search"
  },
  "competitors": [
    {
      "name": "Real Competitor Name (Suburb) — found via '[vertical] ${lead.location}' search",
      "edge": "Specific thing they do better online, from your search"
    }
  ],
  "opportunities": [
    {
      "title": "Gap title",
      "detail": "Specific detail with evidence from your research",
      "source": "Platform name / URL · ${today}"
    }
  ],
  "ability": [
    {
      "t": "Signal they can afford this service (pricing, ads, expansion etc.)",
      "source": "Where you found this"
    }
  ],
  "timing": [
    {
      "t": "Specific reason why right now is a good moment to pitch",
      "source": "Where you found this"
    }
  ],
  "dmIntel": "Owner/manager name if publicly visible (website about page, press mentions). Say 'not publicly listed' if not found. Public info only — no personal data.",
  "strategy": "Specific pitch approach based on what you found in your research.",
  "pkg": {
    "name": "Package name tailored to their vertical",
    "scope": "What is included",
    "price": "${profile.pricing || '$1,500–4,000 / mo'}"
  },
  "risks": [
    "Specific potential objection based on what you found"
  ],
  "service": "The specific service from '${profile.services}' that fits best",
  "pitch": "One punchy tagline tailored to what you found",
  "hook": "2–3 sentence outreach opener using specific details from your research. Make it feel informed, not generic.",
  "value": "Estimated monthly deal value",
  "dm": "Contact person name if found publicly, else 'unknown'",
  "channel": "Best contact method — e.g. 'IG DM @handle' or 'contact form at domain.com/contact'",
  "sources": ["Every URL you searched and used in this dossier"]
}`;

  try {
    const text = await runClaude(DEEP_MODEL, system, user, 10000);
    const dossier = parseJSON(text);
    console.log(' ✓');
    return dossier;
  } catch (e) {
    console.log(` ⚠️  failed (${e.message.slice(0, 60)})`);
    return null;
  }
}

// ─── Quick profile (leads 11–25, no deep dive) ───────────────────────────────

function makeQuickProfile(lead, profile, rank) {
  const tier = lead.fitScore >= 80 ? 'A' : lead.fitScore >= 60 ? 'B' : 'C';
  const venues = lead.venueCount || 1;
  return {
    rank,
    name: lead.name,
    vertical: lead.vertical,
    location: lead.location,
    region: lead.region,
    geo: lead.geo,
    size: `${venues} venue${venues > 1 ? 's' : ''} · ${lead.staffEstimate || 'small team'}`,
    presence: lead.presenceScore || 2,
    tag: lead.tag,
    fit: lead.fitScore || 65,
    tier,
    deep: false,
    why: lead.why,
    signals: [
      { t: lead.why, s: (lead.searchedUrls || [])[0] || '#' },
      ...(lead.website ? [{ t: `Website: ${lead.website}`, s: `https://${lead.website}` }] : []),
      ...(lead.instagram ? [{ t: `Instagram: ${lead.instagram}`, s: `https://instagram.com/${lead.instagram.replace('@', '')}` }] : []),
    ],
    service: profile.services,
    pitch: `Close the gap: ${lead.tag}.`,
    hook: `Hi ${lead.name} — ${lead.why}`,
    value: profile.pricing || '$1,500–4,000/mo',
    dm: 'unknown',
    channel: lead.instagram
      ? `IG DM ${lead.instagram}`
      : (lead.website ? `contact form at ${lead.website}` : 'website contact form'),
    sources: lead.searchedUrls || [],
  };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('\n❌  ANTHROPIC_API_KEY is not set.');
    console.error('   export ANTHROPIC_API_KEY=sk-ant-...\n');
    process.exit(1);
  }

  const profile = loadProfile();
  const today = new Date();
  const todayISO = today.toISOString().split('T')[0];
  const todayLabel = today.toLocaleDateString('en-AU', { day: 'numeric', month: 'long', year: 'numeric' });

  console.log('\n Daily Lead Engine — Research Agent');
  console.log(' ====================================');
  console.log(` Business  : ${profile.businessName}`);
  console.log(` Home base : ${profile.location}`);
  console.log(` Targets   : ${(profile.targetLocations || [profile.location]).join(', ')}`);
  console.log(` Date      : ${todayLabel}`);
  console.log('');

  // ── Phase 1: Discover candidates ───────────────────────────────────────────

  console.log('Phase 1 — Lead discovery\n');
  const allCandidates = [];

  // Local
  const localLeads = await discoverLeads(
    profile, profile.location, 'local', GEO_MIX.local + 4
  );
  allCandidates.push(...localLeads);
  await sleep(2500);

  // Interstate
  const interstateLocations = (profile.targetLocations || [])
    .filter(l => l !== profile.location);

  for (const loc of interstateLocations.slice(0, 3)) {
    const perLoc = Math.ceil((GEO_MIX.interstate + 3) / Math.min(interstateLocations.length, 3));
    const leads = await discoverLeads(profile, loc, 'interstate', perLoc);
    allCandidates.push(...leads);
    await sleep(2500);
  }

  // International
  for (const loc of (profile.internationalLocations || []).slice(0, 2)) {
    const leads = await discoverLeads(profile, loc, 'international', GEO_MIX.international + 2);
    allCandidates.push(...leads);
    await sleep(2500);
  }

  console.log(`\nTotal candidates: ${allCandidates.length}`);

  // ── Phase 2: Score & select top 25 ─────────────────────────────────────────

  console.log('\nPhase 2 — Selection & geo-mix\n');

  // Deduplicate by name
  const seen = new Set();
  const unique = allCandidates.filter(l => {
    if (!l.name || seen.has(l.name)) return false;
    seen.add(l.name);
    return true;
  });

  // Split by geo and sort each group
  const byGeo = { local: [], interstate: [], international: [] };
  unique.forEach(l => {
    const g = l.geo && byGeo[l.geo] ? l.geo : 'local';
    byGeo[g].push(l);
  });
  Object.keys(byGeo).forEach(g => {
    byGeo[g].sort((a, b) => (b.fitScore || 0) - (a.fitScore || 0));
  });

  const selected = [
    ...byGeo.local.slice(0, GEO_MIX.local),
    ...byGeo.interstate.slice(0, GEO_MIX.interstate),
    ...byGeo.international.slice(0, GEO_MIX.international),
  ]
    .sort((a, b) => (b.fitScore || 0) - (a.fitScore || 0))
    .slice(0, 25);

  console.log(`  Selected ${selected.length} leads (target 25)`);
  console.log(`  Local: ${selected.filter(l=>l.geo==='local').length}  Interstate: ${selected.filter(l=>l.geo==='interstate').length}  International: ${selected.filter(l=>l.geo==='international').length}`);

  // ── Phase 3: Deep dives (top 10) ───────────────────────────────────────────

  const forDeepDive = selected.slice(0, DEEP_DIVE_COUNT);
  const forQuick    = selected.slice(DEEP_DIVE_COUNT);

  console.log(`\nPhase 3 — Deep dives (${forDeepDive.length} leads)\n`);

  const finalLeads = [];
  let deepCount = 0;

  for (const lead of forDeepDive) {
    const rank = finalLeads.length + 1;
    const dossier = await deepDive(lead, profile, rank);
    if (dossier) {
      dossier.rank = rank;
      dossier.deep = true;
      finalLeads.push(dossier);
      deepCount++;
    } else {
      // Fall back to quick profile if deep dive parsing fails
      finalLeads.push(makeQuickProfile(lead, profile, rank));
    }
    await sleep(2500);
  }

  // ── Phase 4: Quick profiles for leads 11–25 ────────────────────────────────

  console.log(`\nPhase 4 — Quick profiles (${forQuick.length} leads)\n`);
  for (const lead of forQuick) {
    finalLeads.push(makeQuickProfile(lead, profile, finalLeads.length + 1));
  }

  // ── Phase 5: Append to run history & save ──────────────────────────────────

  const existingRuns = loadExistingRuns();

  const newRun = {
    date: todayISO,
    label: `${todayLabel} — Live research`,
    meta: {
      verified: finalLeads.length,
      researched: deepCount,
      sample: false,
      location: profile.location,
    },
    leads: finalLeads,
  };

  const allRuns = [newRun, ...existingRuns].slice(0, MAX_HISTORY_RUNS);
  saveRuns(allRuns);

  const localCount     = finalLeads.filter(l => l.geo === 'local').length;
  const interstateCount = finalLeads.filter(l => l.geo === 'interstate').length;
  const intlCount      = finalLeads.filter(l => l.geo === 'international').length;

  console.log('\n ====================================');
  console.log(` ✅  Done!`);
  console.log(`     ${finalLeads.length} leads  ·  ${deepCount} deep-dived`);
  console.log(`     ${localCount} local  ·  ${interstateCount} interstate  ·  ${intlCount} international`);
  console.log(`     Saved to runs.js — open index.html to view.`);
  console.log('');
}

main().catch(err => {
  console.error('\n❌  Research agent failed:', err.message);
  if (err.status === 401) {
    console.error('   Invalid ANTHROPIC_API_KEY — check your key.');
  } else if (err.status === 400) {
    console.error('   Bad request — the web search beta may not be enabled for your API tier.');
    console.error('   Visit https://console.anthropic.com to check beta feature access.');
  } else if (err.code === 'ENOENT') {
    console.error('   File not found — did you run "npm install" first?');
  }
  process.exit(1);
});
