// Phase 38 — pre-compute MiniLM features for a small AG News-style corpus.
// Runs once at build time; output lives in public/data/{features,labels}.bin
// and is fetched by both the head-classifier DOs (test split) and the
// browser workers (train split).
//
// Why hand-curated 100 examples rather than 120K real AG News samples:
// (a) self-contained — no HuggingFace dataset download in CI
// (b) small enough to ship as a 150 KB static asset (no R2 needed)
// (c) MiniLM embedding is the only thing that needs the network
//
// Replace EXAMPLES below with a full AG News slice when scaling up.

import { pipeline } from "@huggingface/transformers";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

// 25 examples per class × 4 classes = 100 total.
// Classes: 0=World, 1=Sports, 2=Business, 3=Sci-Tech (matches AG News convention)
const EXAMPLES = [
  // === 0: World ===
  ["UN Security Council meets to address ongoing conflict in eastern Europe", 0],
  ["Prime minister announces new diplomatic initiative with neighboring states", 0],
  ["Earthquake strikes coastal region, thousands displaced from their homes", 0],
  ["European leaders gather for emergency summit on energy policy", 0],
  ["Refugee crisis worsens as more cross border seeking safety from violence", 0],
  ["Trade negotiations between major powers stall amid tariff disputes", 0],
  ["Climate summit produces non-binding agreement after weeks of talks", 0],
  ["Protests continue in capital city following disputed election results", 0],
  ["Peacekeeping forces deployed to monitor ceasefire in contested region", 0],
  ["Foreign minister condemns recent missile strike on civilian infrastructure", 0],
  ["Floodwaters recede revealing extensive damage to rural villages", 0],
  ["Sanctions imposed on financial institutions following human rights report", 0],
  ["Pope addresses crowd of pilgrims gathered for annual religious celebration", 0],
  ["Cargo ship runs aground blocking critical international shipping lane", 0],
  ["Embassy evacuated after unrest spreads through diplomatic quarter", 0],
  ["Africa Union holds emergency session on humanitarian crisis in Sahel", 0],
  ["Border crossing reopens after months of diplomatic standoff", 0],
  ["Volcanic eruption forces evacuation of nearby town and surrounding area", 0],
  ["Election observers report procedural irregularities in several districts", 0],
  ["Hostage situation ends peacefully after lengthy negotiations with militants", 0],
  ["Aid convoy reaches besieged city carrying medical supplies and food", 0],
  ["Treaty ratified establishing new framework for cross-border environmental protection", 0],
  ["Royal family announces engagement to wide popular celebration in the country", 0],
  ["Famine declared in northern province as drought enters fourth consecutive year", 0],
  ["Joint military exercises begin involving forces from a dozen allied nations", 0],

  // === 1: Sports ===
  ["Star quarterback throws four touchdowns leading team to playoff victory", 1],
  ["Tennis world number one upset in straight sets at grand slam tournament", 1],
  ["Cricket team breaks decades old record with massive total in test match", 1],
  ["Basketball coach fired after disappointing start to the regular season", 1],
  ["Marathon world record falls as runner crosses finish line under historic time", 1],
  ["Soccer club signs young midfielder for record transfer fee from rival club", 1],
  ["Boxing champion announces retirement after long career and many title defenses", 1],
  ["Olympic swimmer wins fifth gold medal in personal best time at games", 1],
  ["Formula one driver crashes during qualifying lap but escapes serious injury", 1],
  ["Hockey team clinches division title with overtime goal in final game", 1],
  ["Golf legend makes surprise return to competition after lengthy injury layoff", 1],
  ["Cycling stage winner sprints to victory after long breakaway in the mountains", 1],
  ["Baseball pitcher throws no hitter in tense playoff game watched by millions", 1],
  ["Rugby coach selects rookies for upcoming international tour against tough opponents", 1],
  ["Basketball star scores career high points in losing effort against rivals", 1],
  ["Football manager signs new contract extension following successful season finish", 1],
  ["Tennis doubles team defends title in five set final at premier event", 1],
  ["Skier wins downhill race in dramatic finish edging out previous champion", 1],
  ["Volleyball team upsets defending champion in straight sets at world cup", 1],
  ["Athletics federation suspends sprinter following failed doping control test", 1],
  ["Wrestling champion defeats challenger to retain belt at major weekend event", 1],
  ["Football league announces expansion adding two new franchises in southern cities", 1],
  ["Skateboarder lands first ever quadruple flip in competition stunning crowd", 1],
  ["Gymnast scores perfect ten in floor routine at international championship", 1],
  ["Surfer wins championship title after riding massive wave during final heat", 1],

  // === 2: Business ===
  ["Tech giant reports record quarterly earnings beating wall street expectations again", 2],
  ["Federal reserve raises interest rates by quarter point in latest policy meeting", 2],
  ["Oil prices surge after producers agree to production cuts in surprise deal", 2],
  ["Major bank announces layoffs of thousands of employees in restructuring move", 2],
  ["Electric vehicle maker delays production launch citing supply chain bottlenecks", 2],
  ["Stock market closes at record high amid optimism about economic recovery prospects", 2],
  ["Hedge fund acquires controlling stake in struggling retail chain", 2],
  ["Cryptocurrency exchange files for bankruptcy after liquidity crisis hits hard", 2],
  ["Merger talks between two airlines collapse over regulatory antitrust concerns", 2],
  ["Inflation data shows continued slowdown giving central bank flexibility on policy", 2],
  ["Retail giant reports falling holiday sales as consumers cut back on spending", 2],
  ["Construction firm wins contract to build new high speed rail corridor between cities", 2],
  ["Pharmaceutical company settles class action lawsuit over drug pricing practices", 2],
  ["Investment bank advises caution as bond yields reach multi year highs", 2],
  ["Startup raises large series b funding round valuing company in billions", 2],
  ["Automaker recalls millions of vehicles over faulty airbag manufacturing defect", 2],
  ["Trade deal expected to boost agricultural exports to overseas markets significantly", 2],
  ["Real estate prices in major cities cool as mortgage rates remain elevated", 2],
  ["Energy company announces dividend increase after strong cash flow generation quarter", 2],
  ["Manufacturing index falls signaling potential slowdown in industrial production", 2],
  ["Insurance firm exits state market citing increased natural disaster claim costs", 2],
  ["Private equity firm takes restaurant chain public in highly anticipated offering", 2],
  ["Currency exchange rate volatility hits exporters reporting margin pressure", 2],
  ["Logistics company orders fleet of electric trucks for last mile delivery operations", 2],
  ["Quarterly gdp growth exceeds estimates driven by strong consumer spending and exports", 2],

  // === 3: Sci-Tech ===
  ["Researchers announce breakthrough in quantum computing achieving stable error correction", 3],
  ["New space telescope captures unprecedented images of distant galaxies forming", 3],
  ["Artificial intelligence model achieves state of the art on language reasoning benchmark", 3],
  ["Biotech firm reports promising results from gene therapy trial for rare disorder", 3],
  ["Astronomers detect unusual radio signal from nearby star system using new array", 3],
  ["Battery startup unveils solid state cell with longer life and faster charging speed", 3],
  ["Climate scientists publish study linking ocean warming to extreme weather patterns", 3],
  ["Robotics company demonstrates humanoid robot performing complex household tasks", 3],
  ["Vaccine for tropical disease shows high efficacy in large scale clinical trial", 3],
  ["Quantum encryption network successfully tested between major financial institutions", 3],
  ["Satellite constellation completes deployment providing global high speed internet coverage", 3],
  ["Cancer researchers identify new mechanism allowing tumors to evade immune detection", 3],
  ["Programming language reaches major milestone version with concurrency improvements", 3],
  ["Solar panel efficiency record broken by lab demonstrating new perovskite material", 3],
  ["Cyber security firm warns of widespread vulnerability in commonly used software library", 3],
  ["Neural interface allows paralyzed patient to control prosthetic arm with thought alone", 3],
  ["Particle accelerator detects rare decay pattern challenging standard model predictions", 3],
  ["Open source machine learning framework releases version supporting distributed training", 3],
  ["Marine biologists discover new deep sea species near hydrothermal vents in pacific", 3],
  ["Fusion experiment achieves net energy gain marking historic milestone for the field", 3],
  ["Genome sequencing reveals new insights into evolution of mammals over millions of years", 3],
  ["Augmented reality glasses prototype receives positive reviews from early developer testers", 3],
  ["Mars rover finds evidence suggesting ancient water flow in newly explored crater", 3],
  ["Algorithm developed predicts protein structure with near experimental accuracy now", 3],
  ["Self driving car software passes regulatory milestone allowing wider deployment trials", 3],
];

// CRITICAL: shuffle so train/test split is class-balanced.
// Without this, EXAMPLES is in class order → the last 25 (test split)
// are all class 3 → trivial 100%-accuracy degenerate solution.
// Deterministic seed=137 so the split is reproducible.
let _rngState = 137 >>> 0;
function _rng() {
  _rngState = (_rngState + 0x6D2B79F5) >>> 0;
  let r = _rngState;
  r = Math.imul(r ^ (r >>> 15), r | 1);
  r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
  return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
}
for (let i = EXAMPLES.length - 1; i > 0; i--) {
  const j = Math.floor(_rng() * (i + 1));
  [EXAMPLES[i], EXAMPLES[j]] = [EXAMPLES[j], EXAMPLES[i]];
}

const N = EXAMPLES.length;
console.log(`Loading MiniLM-L6-v2 from @huggingface/transformers...`);
const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
  quantized: true,
});

console.log(`Computing embeddings for ${N} examples...`);
const D = 384;
const features = new Float32Array(N * D);
const labels = new Uint8Array(N);

for (let i = 0; i < N; i++) {
  const [text, label] = EXAMPLES[i];
  // pooling: "mean" gives a single D-dim vector per input.
  // normalize: false — we want raw activations, not unit-norm.
  const out = await extractor(text, { pooling: "mean", normalize: false });
  const vec = out.data;
  if (vec.length !== D) throw new Error(`expected D=${D}, got ${vec.length}`);
  features.set(vec, i * D);
  labels[i] = label;
  if ((i + 1) % 20 === 0) console.log(`  ${i + 1}/${N}`);
}

// Layout for the .bin files:
//   header (16 bytes): [uint32 N][uint32 D][uint32 K][uint32 _reserved]
//   then N*D float32 features, then N uint8 labels.
const K = 4;
const headerBytes = 16;
const buf = new ArrayBuffer(headerBytes + N * D * 4 + N);
const view = new DataView(buf);
view.setUint32(0, N, true);
view.setUint32(4, D, true);
view.setUint32(8, K, true);
view.setUint32(12, 0, true);
new Float32Array(buf, headerBytes, N * D).set(features);
new Uint8Array(buf, headerBytes + N * D * 4, N).set(labels);

const out = "public/data/agnews-mini.bin";
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, Buffer.from(buf));
console.log(`\nwrote ${out}  (${buf.byteLength} bytes, N=${N} D=${D} K=${K})`);

// Sanity: print class balance + a sample vector summary
const counts = new Array(K).fill(0);
for (let i = 0; i < N; i++) counts[labels[i]]++;
console.log(`class counts: ${counts.join(", ")} (target: 25 each)`);
const norms = [];
for (let i = 0; i < Math.min(N, 4); i++) {
  let s = 0;
  for (let j = 0; j < D; j++) s += features[i * D + j] ** 2;
  norms.push(Math.sqrt(s).toFixed(3));
}
console.log(`feature L2 norms (first 4): ${norms.join(", ")}`);
