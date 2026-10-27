// Climate Exposure Explorer: Three.js front end.
// Every H3 land cell is one instance of a hexagonal prism. Height and colour both encode the selected metric;
// switching scenario/decade/metric eases the prisms to their new heights, so change over time is felt, not just read.
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";

// ── Metric definitions: fixed domains so morphing between scenarios is honest ─────────────────────────
const SEQ = ["#2b1308", "#4a1f0e", "#7a2f12", "#a83f16", "#d95926", "#eb6834", "#f28c5f", "#f8b38f", "#fdd8c4"]; // one hue, dark→light on dark
const DIV = { neg: "#d95926", mid: "#383835", pos: "#3987e5" };                                                   // drier ← → wetter
const METRICS = {
  tasmax_djf:   { label: "Summer (Dec-Feb) mean daily max, °C", min: 20, max: 42, fmt: v => `${v.toFixed(1)} °C` },
  d_tasmax_djf: { label: "Summer warming vs 2001-10, Δ °C",     min: 0,  max: 8,  fmt: v => `+${v.toFixed(2)} °C`, delta: true },
  hot_months:   { label: "Months a year with mean daily max ≥ 35 °C", min: 0, max: 9, fmt: v => `${v.toFixed(1)} months` },
  pct_pr:       { label: "Annual rainfall change vs 2001-10, %",  min: -40, max: 40, fmt: v => `${v > 0 ? "+" : ""}${v.toFixed(1)} %`, diverging: true, delta: true },
  exposure:     { label: "Height = people (log) · colour = summer warming Δ °C", min: 0, max: 8, fmt: v => `+${v.toFixed(2)} °C`, delta: true },
};
const SCEN_LABEL = { historical: "Baseline", ssp126: "SSP1-2.6", ssp370: "SSP3-7.0", ssp585: "SSP5-8.5" };
const MAX_H = 4.2;                         // scene units for the tallest prism
const LON0 = 134, LAT0 = -27, KX = Math.cos(LAT0 * Math.PI / 180);
const proj = (lat, lon) => [(lon - LON0) * KX, -(lat - LAT0)];   // equirectangular, metres-ish aspect at mid-latitude

const state = { scenario: "ssp370", period: "2091-2100", metric: "tasmax_djf", minWarm: 3.0 };
let data, mesh, cur, tgt, curCol, tgtCol, hovered = -1, summary = [];
const $ = id => document.getElementById(id);

// ── Scene ─────────────────────────────────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas: $("scene"), antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
const scene = new THREE.Scene();
scene.background = new THREE.Color("#0d0d0d");
scene.fog = new THREE.Fog("#0d0d0d", 70, 140);
const camera = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, 0.1, 500);
camera.position.set(0, 40, 44);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 2); controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.47;
controls.minDistance = 12; controls.maxDistance = 110; controls.autoRotate = true; controls.autoRotateSpeed = 0.35;
controls.addEventListener("start", () => (controls.autoRotate = false));

scene.add(new THREE.HemisphereLight("#ffffff", "#202020", 1.4));
const sun = new THREE.DirectionalLight("#ffffff", 1.6); sun.position.set(-20, 40, 25); scene.add(sun);
const ground = new THREE.Mesh(new THREE.CircleGeometry(60, 64), new THREE.MeshStandardMaterial({ color: "#121211", roughness: 1 }));
ground.rotation.x = -Math.PI / 2; ground.position.y = -0.02; scene.add(ground);

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight);
});

// ── Colour helpers ────────────────────────────────────────────────────────────────────────────────────
const seqColors = SEQ.map(h => new THREE.Color(h));
function seq(t) {                                              // t in [0,1] → interpolated sequential colour
  t = Math.min(1, Math.max(0, t)) * (seqColors.length - 1);
  const i = Math.floor(t), f = t - i;
  return seqColors[i].clone().lerp(seqColors[Math.min(i + 1, seqColors.length - 1)], f);
}
const divC = { neg: new THREE.Color(DIV.neg), mid: new THREE.Color(DIV.mid), pos: new THREE.Color(DIV.pos) };
const div = t => t < 0 ? divC.mid.clone().lerp(divC.neg, Math.min(1, -t)) : divC.mid.clone().lerp(divC.pos, Math.min(1, t));

// ── Build geometry once data arrives ──────────────────────────────────────────────────────────────────
async function init() {
  const [cells, sum, outlines] = await Promise.all(["/api/cells", "/api/summary", "/api/outlines"].map(u => fetch(u).then(r => r.json())));
  data = cells; summary = sum;
  const n = data.cells.length;

  // Unit hex prism, base at y=0. CylinderGeometry's first vertex sits on +z, so we rotate each instance to its H3 vertex.
  const geo = new THREE.CylinderGeometry(1, 1, 1, 6, 1); geo.translate(0, 0.5, 0);
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 });
  mesh = new THREE.InstancedMesh(geo, mat, n);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  data.cells.forEach(c => {
    const [x, z] = proj(c.lat, c.lon);
    const [vx, vz] = proj(c.b[0][0], c.b[0][1]);
    c.x = x; c.z = z; c.r = Math.hypot(vx - x, vz - z) * 0.93;     // small gap between hexes reads as a grid
    c.rot = Math.atan2(vx - x, vz - z);
  });
  cur = new Float32Array(n).fill(0.02); tgt = new Float32Array(n);
  curCol = data.cells.map(() => new THREE.Color("#2b1308")); tgtCol = data.cells.map(() => new THREE.Color());
  scene.add(mesh);

  // State outlines (ABS STE 2021, simplified in notebook 03), drawn just above the ground
  const lineMat = new THREE.LineBasicMaterial({ color: "#8d8c84", transparent: true, opacity: 0.55 });
  for (const o of outlines) {
    const g = JSON.parse(o.geojson);
    const polys = g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : [];
    for (const poly of polys) {
      const ring = poly[0]; if (ring.length < 8) continue;
      const pts = ring.map(([lon, lat]) => { const [x, z] = proj(lat, lon); return new THREE.Vector3(x, 0.01, z); });
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), lineMat));
    }
  }
  $("loading").style.display = "none";
  wireUI(); update();
}

// ── Map the current selection onto per-cell target heights/colours ───────────────────────────────────
function layerKey() { return state.scenario === "historical" ? "historical|2001-2010" : `${state.scenario}|${state.period}`; }

function update() {
  const L = data.layers[layerKey()], m = METRICS[state.metric];
  const maxLogPop = Math.log10(1 + Math.max(...data.cells.map(c => c.pop)));
  data.cells.forEach((c, i) => {
    let v, t;
    if (state.metric === "exposure") {
      v = L.d_tasmax_djf[i] ?? 0;
      tgt[i] = 0.05 + MAX_H * 1.1 * Math.log10(1 + c.pop) / maxLogPop;
      tgtCol[i].copy(seq((v - m.min) / (m.max - m.min)));
    } else {
      v = L[state.metric][i] ?? 0;
      if (m.diverging) {
        t = v / m.max;
        tgt[i] = 0.05 + MAX_H * 0.6 * Math.min(1, Math.abs(t));
        tgtCol[i].copy(div(t));
      } else {
        t = (v - m.min) / (m.max - m.min);
        tgt[i] = 0.05 + MAX_H * Math.min(1.1, Math.max(0, t));
        tgtCol[i].copy(seq(t));
      }
    }
  });
  drawLegend(m); updateKPIs(); loadSidePanels();
  $("hint").textContent = state.scenario === "historical" && m.delta
    ? "Baseline decade: changes are measured against this, so the map is flat by design." : "";
  $("periodBox").style.opacity = state.scenario === "historical" ? 0.35 : 1;
}

// ── Legend: gradient + 5 ticks, text in ink tokens (never the series colour) ──────────────────────────
function drawLegend(m) {
  $("lTitle").textContent = m.label;
  const stops = m.diverging ? [DIV.neg, DIV.mid, DIV.pos] : SEQ;
  $("lBar").style.background = `linear-gradient(90deg, ${stops.join(",")})`;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(f => m.min + f * (m.max - m.min));
  $("lTicks").innerHTML = ticks.map(t => `<span>${Number.isInteger(t) ? t : t.toFixed(1)}</span>`).join("");
}

// ── KPI tiles from gold_exposure_summary ──────────────────────────────────────────────────────────────
const fmtPeople = n => n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : `${n}`;
function updateKPIs() {
  const period = state.scenario === "historical" ? "2001-2010" : state.period;
  const row = summary.find(r => r.state === "Australia" && r.scenario === state.scenario && r.period === period);
  const base = summary.find(r => r.state === "Australia" && r.scenario === "historical");
  if (!row) return;
  $("kWarm").textContent = `+${Number(row.pop_weighted_d_tasmax_djf).toFixed(2)} °C`;
  $("kHot").textContent = fmtPeople(Number(row.pop_summer_max_35c));
  $("kHotSub").textContent = state.scenario === "historical" ? "2001-2010 baseline"
    : `${(Number(row.pop_summer_max_35c) / Number(base.pop_summer_max_35c)).toFixed(1)}× the 2001-10 baseline of ${fmtPeople(Number(base.pop_summer_max_35c))}`;
}

// ── Side panels: UC function (state bars) + top SA2s ──────────────────────────────────────────────────
let sideReq = 0;
async function loadSidePanels() {
  const req = ++sideReq;
  const sc = state.scenario === "historical" ? "ssp370" : state.scenario;
  const q = `scenario=${sc}&period=${state.period}`;
  const [he, top] = await Promise.all([
    fetch(`/api/heat_exposure?${q}&min_warming=${state.minWarm}`).then(r => r.json()),
    fetch(`/api/top_sa2?${q}`).then(r => r.json())]);
  if (req !== sideReq) return;                      // a newer selection won the race
  $("fnName").textContent = `${he.function}('${sc}', '${state.period}', ${state.minWarm})`;
  const rows = he.rows.filter(r => r.state !== "Other Territories");
  const max = Math.max(1, ...rows.map(r => Number(r.population)));
  $("bars").innerHTML = rows.length ? rows.map(r => `
    <div class="bar" title="${r.state}: ${Number(r.population).toLocaleString()} people in ${r.sa2_count} SA2s, avg +${r.avg_d_tasmax_djf} °C">
      <span class="name">${r.state.replace("Australian Capital Territory", "ACT").replace("Northern Territory", "NT")}</span>
      <span class="track"><span class="fill" style="display:block;width:${(100 * Number(r.population) / max).toFixed(1)}%"></span></span>
      <span class="v">${fmtPeople(Number(r.population))}</span>
    </div>`).join("") : `<p class="fn">No SA2s reach that much warming in this scenario.</p>`;
  $("top").innerHTML = top.map(r => `<li><b>${r.sa2_name}</b> (${r.state.replace("Australian Capital Territory", "ACT")}) · ${fmtPeople(Number(r.population))} people · +${Number(r.d_tasmax_djf).toFixed(1)} °C</li>`).join("");
}

// ── UI wiring ─────────────────────────────────────────────────────────────────────────────────────────
function setSeg(id, v) { document.querySelectorAll(`#${id} button`).forEach(b => b.classList.toggle("on", b.dataset.v === v)); }
function wireUI() {
  document.querySelectorAll("#scenario button").forEach(b => b.onclick = () => { state.scenario = b.dataset.v; setSeg("scenario", state.scenario); update(); });
  document.querySelectorAll("#period button").forEach(b => b.onclick = () => { state.period = b.dataset.v; setSeg("period", state.period); update(); });
  document.querySelectorAll("#metric input").forEach(r => r.onchange = () => { state.metric = r.value; update(); });
  let t; $("minWarm").oninput = e => { state.minWarm = Number(e.target.value); $("minW").textContent = state.minWarm.toFixed(1); clearTimeout(t); t = setTimeout(loadSidePanels, 200); };
  $("play").onclick = playCentury;
}

// "Play the century": baseline → 2041-50 → 2091-2100 under the chosen scenario (SSP3-7.0 if baseline selected)
async function playCentury() {
  const sc = state.scenario === "historical" ? "ssp370" : state.scenario;
  $("play").disabled = true;
  const steps = [["historical", "2091-2100"], [sc, "2041-2050"], [sc, "2091-2100"]];
  for (const [s, p] of steps) {
    state.scenario = s; state.period = p; setSeg("scenario", s); setSeg("period", p); update();
    $("play").textContent = s === "historical" ? "2001-2010 baseline…" : `${SCEN_LABEL[s]} · ${p}…`;
    await new Promise(r => setTimeout(r, 2600));
  }
  $("play").textContent = "▶ Play the century"; $("play").disabled = false;
}

// ── Hover tooltip (raycast against the instanced mesh) ────────────────────────────────────────────────
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
addEventListener("pointermove", e => {
  if (!mesh) return;
  mouse.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObject(mesh)[0];
  hovered = hit ? hit.instanceId : -1;
  const tip = $("tip");
  if (hovered < 0 || e.target.id !== "scene") { tip.style.display = "none"; return; }
  const c = data.cells[hovered], L = data.layers[layerKey()], B = data.layers["historical|2001-2010"], i = hovered;
  const row = (k, v) => `<div class="row"><span>${k}</span><span>${v}</span></div>`;
  tip.innerHTML = `<b>${c.state ?? "–"}</b> <span style="color:#52514e">H3 ${c.id}</span>` +
    row("Summer max", `${L.tasmax_djf[i]?.toFixed(1)} °C <span style="color:#52514e">(was ${B.tasmax_djf[i]?.toFixed(1)})</span>`) +
    row("Hot months / yr", `${L.hot_months[i]?.toFixed(1)}`) +
    row("Rainfall change", `${(L.pct_pr[i] ?? 0) > 0 ? "+" : ""}${L.pct_pr[i]?.toFixed(1)} %`) +
    row("People (ABS 2025)", c.pop.toLocaleString()) +
    row("Lat, lon", `${c.lat.toFixed(2)}, ${c.lon.toFixed(2)}`);
  tip.style.display = "block";
  tip.style.left = `${Math.min(e.clientX + 16, innerWidth - 230)}px`; tip.style.top = `${e.clientY + 16}px`;
});

// ── Render loop: ease heights + colours toward targets ────────────────────────────────────────────────
const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
const white = new THREE.Color("#ffffff"), tmp = new THREE.Color();
function frame() {
  requestAnimationFrame(frame);
  if (mesh) {
    data.cells.forEach((c, i) => {
      cur[i] += (tgt[i] - cur[i]) * 0.07;
      curCol[i].lerp(tgtCol[i], 0.07);
      q.setFromAxisAngle(up, c.rot); p.set(c.x, 0, c.z); s.set(c.r, cur[i], c.r);
      mesh.setMatrixAt(i, m4.compose(p, q, s));
      mesh.setColorAt(i, i === hovered ? tmp.copy(curCol[i]).lerp(white, 0.6) : curCol[i]);
    });
    mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true;
  }
  controls.update();
  renderer.render(scene, camera);
}
frame();
init().catch(err => { $("loading").textContent = `Failed to load data: ${err}`; console.error(err); });
