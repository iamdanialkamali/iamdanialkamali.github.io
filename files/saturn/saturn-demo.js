/*
 * SATURN project page: animated walkthrough of one 3D FORCE question.
 * Images, detections, the 3D scene, the program, the soft scores and the baseline
 * answers are read from example.json and cloud.bin.
 *
 * Timing model: one clock. Every step has a fixed length, and every sub-animation in a
 * step has an explicit start and length (seconds from the step start). The picture is a
 * pure function of (step, time in step), so pausing, jumping and replaying never leave a
 * timer behind. Each step ends with a hold in which nothing moves.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const HOLD = 0.9;
const COL = { cam: '#1c7ed6', obj: '#7048e8', target: '#2f9e44' };

const clamp01 = (x) => Math.max(0, Math.min(1, x));
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const fmt = (x) => (x >= 0.095 || x === 0 ? x.toFixed(2) : x.toFixed(3));
const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function h(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}
const svgNS = 'http://www.w3.org/2000/svg';
function s(tag, attrs) {
  const e = document.createElementNS(svgNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

/* only touch the DOM when a value changes */
const _cache = new WeakMap();
function setStyle(el, prop, val) {
  let c = _cache.get(el);
  if (!c) { c = {}; _cache.set(el, c); }
  if (c[prop] !== val) { c[prop] = val; el.style[prop] = val; }
}
const setOp = (el, v) => setStyle(el, 'opacity', String(Math.round(clamp01(v) * 1000) / 1000));
function setCls(el, cls, on) { if (el.classList.contains(cls) !== on) el.classList.toggle(cls, on); }
function setText(el, t) { if (el.textContent !== t) el.textContent = t; }
function setHTML(el, t) { if (el.__html !== t) { el.__html = t; el.innerHTML = t; } }

/* names used in labels and captions (the dataset's objects, matched to SATURN's detections) */
const SHORT = {
  'red sedan (back left)': 'back-left sedan', 'red sedan (back right)': 'back-right sedan',
  'red sedan (front)': 'front sedan', 'yellow minivan': 'minivan', 'green pickup truck': 'pickup',
  'gray airliner': 'airliner', 'gray school bus': 'school bus'
};

async function main(root) {
  const base = root.dataset.src;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const [data, buf] = await Promise.all([
    fetch(base + 'example.json').then((r) => { if (!r.ok) throw new Error('example.json ' + r.status); return r.json(); }),
    fetch(base + 'cloud.bin').then((r) => { if (!r.ok) throw new Error('cloud.bin ' + r.status); return r.arrayBuffer(); })
  ]);
  root.innerHTML = '';
  root.classList.add('sd-ready');

  const S = data.scores;
  const objs = data.objects;
  const N = objs.length;
  const mvId = S.minivan, pkId = S.pickup, ansId = S.answer;
  const cands = objs.filter((o) => S.is_sedan[o.id] > 0.5).map((o) => o.id).sort((a, b) => S.iota[b] - S.iota[a]);
  const shortOf = (i) => SHORT[objs[i].name] || objs[i].name;
  const byName = (n) => objs.find((o) => o.name === n).id;
  const BL = byName('red sedan (back left)'), BR = byName('red sedan (back right)'), FR = byName('red sedan (front)');
  const order = ['Qwen3.5-9B', 'Gemini-3.1-Pro', 'GPT-5.1'];
  const featured = order.map((m) => data.baselines.find((b) => b.model === m)).filter(Boolean);
  const v2 = (x) => fmt(x);

  /* ------------------------------------------------------------ question */
  const q = data.question;
  const pCam = 'behind the minivan car (camera 0 view)';
  const pObj = "to the left of the pickup truck (the pickup truck's view)";
  let qh = esc(q);
  qh = qh.replace('sedan car', '<mark class="sd-m-t">sedan car</mark>');
  qh = qh.replace(esc(pCam), '<mark class="sd-m-cam">' + esc(pCam) + '</mark>');
  qh = qh.replace(esc(pObj), '<mark class="sd-m-obj">' + esc(pObj) + '</mark>');
  root.appendChild(h('div', 'sd-q', '<span class="sd-q-tag">Question</span><span class="sd-q-text">' + qh + '</span>' +
    '<span class="sd-q-meta">3D FORCE REF &middot; ' + data.views.length + ' views &middot; ' + data.hops +
    ' relations &middot; a camera frame and an object frame</span>'));

  const mainEl = h('div', 'sd-main');
  const stage = h('div', 'sd-stage');
  stage.setAttribute('role', 'img');
  stage.setAttribute('aria-label', 'Animated walkthrough of SATURN answering the question above');
  const side = h('div', 'sd-side');
  mainEl.appendChild(stage); mainEl.appendChild(side);
  root.appendChild(mainEl);

  /* ------------------------------------------------------------ layer 1: views + detections */
  const viewsEl = h('div', 'sd-layer sd-views');
  const viewFigs = data.views.map((v, k) => {
    const f = h('figure', 'sd-view');
    const img = h('img'); img.src = base + v.file; img.alt = 'Camera ' + k + ' view'; img.decoding = 'async';
    const svg = s('svg', { viewBox: '0 0 1024 768', preserveAspectRatio: 'none', class: 'sd-ov' });
    const boxes = [];
    objs.forEach((o) => {
      const b = o.boxes[String(k)];
      if (!b) return;
      const x1 = Math.max(0, b[0]), y1 = Math.max(0, b[1]), x2 = Math.min(1024, b[2]), y2 = Math.min(768, b[3]);
      const g = s('g', { class: 'sd-det' });
      g.appendChild(s('rect', { x: x1, y: y1, width: x2 - x1, height: y2 - y1, class: 'sd-det-r' }));
      const tx = Math.min(x1, 1024 - 52), ty = Math.max(y1, 0);
      g.appendChild(s('rect', { x: tx, y: ty, width: 50, height: 44, class: 'sd-det-tagbg' }));
      const t = s('text', { x: tx + 25, y: ty + 33, class: 'sd-det-tag' }); t.textContent = String(o.id);
      g.appendChild(t);
      g.style.opacity = '0';
      svg.appendChild(g);
      boxes.push(g);
    });
    f.appendChild(img); f.appendChild(svg);
    f.appendChild(h('figcaption', null, 'Camera ' + k));
    viewsEl.appendChild(f);
    return { f, boxes };
  });
  const legend = h('div', 'sd-view sd-legend');
  legend.innerHTML = '<div class="sd-legend-in"><div class="sd-legend-big">' + data.views.length + ' views</div>' +
    '<div class="sd-legend-txt">of one scene. The question names two frames of reference: camera&nbsp;0&rsquo;s view and the pickup truck&rsquo;s own view.</div>' +
    '<div class="sd-legend-det"><b>' + N + '</b> objects, matched across the views</div></div>';
  viewsEl.appendChild(legend);
  const legendDet = legend.querySelector('.sd-legend-det');
  stage.appendChild(viewsEl);

  /* ------------------------------------------------------------ layer 2: 3D */
  const l3d = h('div', 'sd-layer sd-3d');
  const photo = h('img', 'sd-photo'); photo.alt = ''; photo.setAttribute('aria-hidden', 'true');
  const hud = h('div', 'sd-hud');
  const strip = h('div', 'sd-strip');
  const stripImgs = data.views.map((v, k) => {
    const im = h('img'); im.src = base + v.file; im.alt = 'Camera ' + k;
    const w = h('div', 'sd-strip-i'); w.appendChild(im); w.appendChild(h('span', null, String(k)));
    strip.appendChild(w); return w;
  });
  l3d.appendChild(photo); l3d.appendChild(hud); l3d.appendChild(strip);
  stage.appendChild(l3d);

  /* ------------------------------------------------------------ layer 3: answer on camera 0 */
  const lans = h('div', 'sd-layer sd-ans');
  const ansImg = h('img'); ansImg.src = base + data.views[0].file; ansImg.alt = 'Camera 0 with the answers drawn';
  const ansSvg = s('svg', { viewBox: '0 0 1024 768', preserveAspectRatio: 'none', class: 'sd-ov' });
  const gtb = data.gt.bbox_cam0;
  const gGT = s('g', { class: 'sd-a-gt' });
  gGT.appendChild(s('rect', { x: gtb[0] - 6, y: gtb[1] - 6, width: gtb[2] - gtb[0] + 12, height: gtb[3] - gtb[1] + 12 }));
  const sb = data.saturn.predicted_bbox_cam0;
  const gSat = s('g', { class: 'sd-a-sat' });
  gSat.appendChild(s('rect', { x: sb[0], y: sb[1], width: sb[2] - sb[0], height: sb[3] - sb[1] }));
  gSat.appendChild(s('rect', { x: sb[0] - 40, y: sb[1] - 54, width: 196, height: 46, class: 'sd-a-tagbg' }));
  const satTag = s('text', { x: sb[0] - 28, y: sb[1] - 20, class: 'sd-a-tag' }); satTag.textContent = 'SATURN ✓';
  gSat.appendChild(satTag);
  const gBase = featured.map((b, k) => {
    const bx = b.box1000_drawn;
    const x1 = bx[0] * 1.024, y1 = bx[1] * 0.768, x2 = bx[2] * 1.024, y2 = bx[3] * 0.768;
    const g = s('g', { class: 'sd-a-base' });
    g.appendChild(s('rect', { x: x1, y: y1, width: x2 - x1, height: y2 - y1 }));
    const cx = x2 - 22 - k * 44, cy = Math.min(y2 + 24, 742);
    g.appendChild(s('circle', { cx, cy, r: 20 }));
    const t = s('text', { x: cx, y: cy + 9 }); t.textContent = String(k + 1);
    g.appendChild(t);
    return g;
  });
  ansSvg.appendChild(gGT); gBase.forEach((g) => ansSvg.appendChild(g)); ansSvg.appendChild(gSat);
  lans.appendChild(ansImg); lans.appendChild(ansSvg);
  lans.appendChild(h('div', 'sd-ans-cap', 'Camera 0 &middot; <span class="sd-k-sat">SATURN</span> &middot; ' +
    '<span class="sd-k-gt">ground truth</span> &middot; <span class="sd-k-base">VLMs 1&ndash;' + featured.length + '</span>'));
  stage.appendChild(lans);
  const stageTitle = h('div', 'sd-stage-title');
  stage.appendChild(stageTitle);

  /* ------------------------------------------------------------ program panel */
  const codeP = h('div', 'sd-panel sd-code');
  codeP.appendChild(h('div', 'sd-panel-h', 'Program'));
  const pre = h('pre', 'sd-pre');
  const lineEls = data.program.map(() => { const d = h('div', 'sd-ln'); pre.appendChild(d); return d; });
  codeP.appendChild(pre);
  side.appendChild(codeP);

  const ln6 = data.program[6] || '';
  const marks6 = [
    ["is_sedan('x1')", 'sd-mk-t'],
    ["cam0_behind('x1', 'x2') & minivan('x2')", 'sd-mk-cam'],
    ["scene.obj_left('x1', 'x3') & pickup('x3')", 'sd-mk-obj'],
    [".iota('x1')", 'sd-mk-io']
  ];
  function hiPy(t) {
    let x = esc(t);
    x = x.replace(/('[^']*')/g, '<span class="sd-s">$1</span>');
    x = x.replace(/\b(return|int)\b/g, '<span class="sd-k">$1</span>');
    x = x.replace(/\b(score|iota|argmax)\b/g, '<span class="sd-f">$1</span>');
    return x;
  }
  function line6Html() {
    let out = '', pos = 0;
    const ms = marks6.map(([sub, cls]) => ({ at: ln6.indexOf(sub), sub, cls })).filter((m) => m.at >= 0).sort((a, b) => a.at - b.at);
    for (const m of ms) {
      if (m.at < pos) continue;
      out += hiPy(ln6.slice(pos, m.at)) + '<span class="' + m.cls + '">' + hiPy(m.sub) + '</span>';
      pos = m.at + m.sub.length;
    }
    return out + hiPy(ln6.slice(pos));
  }
  const fullHtml = data.program.map((t, i) => (i === 6 ? line6Html() : hiPy(t)));
  const totalChars = data.program.reduce((a, t) => a + t.length, 0);
  function reserveProgramHeight() {
    data.program.forEach((t, i) => { lineEls[i].innerHTML = fullHtml[i]; lineEls[i].hidden = false; });
    pre.style.minHeight = '';
    const hgt = pre.getBoundingClientRect().height;
    if (hgt > 0) pre.style.minHeight = Math.ceil(hgt) + 'px';
    typedShown = -2;
  }
  let typedShown = -2;
  function showTyped(nChars) {
    if (nChars === typedShown) return;
    typedShown = nChars;
    setCls(pre, 'sd-pre-on', nChars > 0);
    let left = nChars;
    data.program.forEach((t, i) => {
      if (nChars <= 0) { lineEls[i].innerHTML = ''; lineEls[i].hidden = true; return; }
      if (left >= t.length) { lineEls[i].innerHTML = fullHtml[i]; lineEls[i].hidden = false; lineEls[i].__typed = true; left -= t.length; }
      else if (left > 0) { lineEls[i].innerHTML = esc(t.slice(0, left)) + '<span class="sd-caret"></span>'; lineEls[i].hidden = false; lineEls[i].__typed = false; left = 0; }
      else { lineEls[i].innerHTML = ''; lineEls[i].hidden = true; lineEls[i].__typed = false; }
    });
  }

  /* ------------------------------------------------------------ soft truth values panel */
  const barsP = h('div', 'sd-panel sd-bars');
  barsP.appendChild(h('div', 'sd-panel-h', 'Soft truth values'));
  const tbl = h('div', 'sd-tbl');
  tbl.appendChild(h('div', 'sd-tr sd-th',
    '<div>candidate</div><div>sedan?</div><div class="sd-c-cam">behind minivan<small>camera 0</small></div>' +
    '<div class="sd-c-obj">left of pickup<small>pickup&rsquo;s view</small></div><div>AND<small>= min</small></div>'));
  const rows = cands.map((i) => {
    const r = h('div', 'sd-tr');
    r.appendChild(h('div', 'sd-tname', '<b>' + i + '</b> ' + esc(shortOf(i))));
    const keys = ['is_sedan', 'behind_minivan_cam0', 'left_of_pickup_pickup_view', 'joint_min'];
    const cells = keys.map((key, ci) => {
      const c = h('div', 'sd-cell' + (ci === 1 ? ' sd-c-cam' : ci === 2 ? ' sd-c-obj' : ci === 3 ? ' sd-c-and' : ''));
      const bar = h('i'); const num = h('span');
      c.appendChild(bar); c.appendChild(num);
      r.appendChild(c);
      return { el: c, bar, num, v: S[key][i] };
    });
    tbl.appendChild(r);
    return { r, cells, i };
  });
  barsP.appendChild(tbl);
  barsP.appendChild(h('div', 'sd-sel-h', 'Selection: AND divided by its maximum'));
  const sel = h('div', 'sd-sel');
  const selRows = cands.map((i) => {
    const d = h('div', 'sd-sel-r', '<span class="sd-sel-n"><b>' + i + '</b> ' + esc(shortOf(i)) + '</span><span class="sd-sel-bar"><i></i></span><span class="sd-sel-v"></span>');
    sel.appendChild(d);
    return { d, bar: d.querySelector('i'), v: d.querySelector('.sd-sel-v'), track: d.querySelector('.sd-sel-bar'), val: S.iota[i], i };
  });
  barsP.appendChild(sel);
  const dotLayer = h('div', 'sd-dots');
  barsP.appendChild(dotLayer);
  side.appendChild(barsP);

  /* dots that carry scores into the AND cell, then into the selection bar */
  const flows = [];
  rows.forEach((r, k) => {
    [0, 1, 2].forEach((src) => {
      const d = h('i', 'sd-dot sd-dot-' + src); dotLayer.appendChild(d);
      flows.push({ el: d, kind: 'and', row: k, from: () => r.cells[src].el, to: () => r.cells[3].el });
    });
    const d2 = h('i', 'sd-dot sd-dot-3'); dotLayer.appendChild(d2);
    flows.push({ el: d2, kind: 'sel', row: k, from: () => r.cells[3].el, to: () => selRows[k].track });
  });
  function centerIn(el, host) {
    const a = el.getBoundingClientRect(), b = host.getBoundingClientRect();
    return [a.left - b.left + a.width / 2, a.top - b.top + a.height / 2];
  }

  /* ------------------------------------------------------------ controls + verdict */
  const ctrl = h('div', 'sd-ctrl');
  const pills = h('div', 'sd-pills');
  const capEl = h('p', 'sd-cap'); capEl.setAttribute('aria-live', 'polite');
  const btns = h('div', 'sd-btns');
  const bPrev = h('button', 'sd-b', '<span aria-hidden="true">&#9664;&#9664;</span>'); bPrev.setAttribute('aria-label', 'Previous step');
  const bPlay = h('button', 'sd-b sd-b-main', ''); bPlay.setAttribute('aria-label', 'Play');
  const bNext = h('button', 'sd-b', '<span aria-hidden="true">&#9654;&#9654;</span>'); bNext.setAttribute('aria-label', 'Next step');
  const bRe = h('button', 'sd-b', '<span aria-hidden="true">&#8635;</span>'); bRe.setAttribute('aria-label', 'Replay from the start');
  [bPrev, bPlay, bNext, bRe].forEach((b) => { b.type = 'button'; btns.appendChild(b); });
  const prog = h('div', 'sd-prog');
  const progBar = h('i', 'sd-prog-fill'); prog.appendChild(progBar);
  ctrl.appendChild(pills);
  const crow = h('div', 'sd-ctrl-row'); crow.appendChild(capEl); crow.appendChild(btns);
  ctrl.appendChild(crow); ctrl.appendChild(prog);
  root.appendChild(ctrl);

  const verdict = h('div', 'sd-verdict');
  verdict.appendChild(h('div', 'sd-card sd-card-sat',
    '<div class="sd-card-h"><span class="sd-ok">&#10003;</span> SATURN</div>' +
    '<div>Object ' + ansId + ', the <b>' + esc(shortOf(ansId)) + '</b>, matching the ground truth (IoU ' + data.saturn.iou_gt.toFixed(2) + ').</div>' +
    '<div class="sd-card-m">3D FORCE REF accuracy 81.2%</div>'));
  featured.forEach((b, k) => {
    verdict.appendChild(h('div', 'sd-card sd-card-bad',
      '<div class="sd-card-h"><span class="sd-num">' + (k + 1) + '</span> ' + esc(b.model) + ' <span class="sd-x">&#10007;</span></div>' +
      '<blockquote>&ldquo;' + esc(b.quote) + '&rdquo;</blockquote>' +
      '<div class="sd-card-m">3D FORCE REF accuracy ' + b.ref_accuracy_table1.toFixed(1) + '%</div>'));
  });
  const runs = data.baseline_runs || { runs: 0, wrong: 0 };
  if (runs.runs && runs.runs === runs.wrong) {
    verdict.appendChild(h('div', 'sd-vnote', 'SATURN finds this sedan. All ' + runs.runs + ' baseline runs in our 3D FORCE evaluation miss it.'));
  }
  root.appendChild(verdict);

  /* ------------------------------------------------------------ 3D scene */
  let gl = null;
  try { gl = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' }); } catch (e) { gl = null; }
  const threeOK = !!gl;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 4 / 3, 0.05, 300);
  const labels = new CSS2DRenderer();
  let controls = null;
  if (threeOK) {
    gl.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    gl.setClearColor(0x0f1318, 1);
    gl.domElement.className = 'sd-canvas';
    l3d.insertBefore(gl.domElement, photo);
    labels.domElement.className = 'sd-labels';
    l3d.insertBefore(labels.domElement, hud);
    controls = new OrbitControls(camera, gl.domElement);
    controls.enabled = false;
    controls.enableDamping = false;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.minDistance = 3; controls.maxDistance = 40;
    controls.enableZoom = false;
    controls.enablePan = false;
    gl.domElement.style.touchAction = 'pan-y';
  } else {
    l3d.appendChild(h('div', 'sd-nogl', 'The 3D view needs WebGL, which this browser has turned off. The other panels still play.'));
  }
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const dl = new THREE.DirectionalLight(0xffffff, 1.2); dl.position.set(3, 10, 5); scene.add(dl);

  const n = data.cloud.n, mn = data.cloud.min, mx = data.cloud.max;
  const qv = new Int16Array(buf, 0, n * 3), cv = new Uint8Array(buf, n * 6, n * 3);
  const pos = new Float32Array(n * 3), colA = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) {
    const a = i % 3;
    pos[i] = mn[a] + ((qv[i] + 32768) / 65535) * (mx[a] - mn[a]);
    colA[i] = Math.pow(cv[i] / 255, 2.2);
  }
  const cg = new THREE.BufferGeometry();
  cg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  cg.setAttribute('color', new THREE.BufferAttribute(colA, 3));
  const cmat = new THREE.PointsMaterial({ size: 0.095, vertexColors: true, transparent: true, opacity: 0 });
  scene.add(new THREE.Points(cg, cmat));

  const camObjs = data.cameras.map((c) => {
    const P = V(c.position), F = V(c.forward), U = V(c.up), R = V(c.right);
    const d = 1.5, hx = d * Math.tan((c.fov_x_deg * Math.PI) / 360), hy = d * Math.tan((c.fov_y_deg * Math.PI) / 360);
    const cs = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([a, b]) =>
      P.clone().addScaledVector(F, d).addScaledVector(R, a * hx).addScaledVector(U, b * hy));
    const pts = [];
    cs.forEach((c2) => pts.push(P, c2));
    for (let k = 0; k < 4; k++) pts.push(cs[k], cs[(k + 1) % 4]);
    pts.push(cs[0].clone().lerp(cs[1], 0.35), P.clone().addScaledVector(F, d).addScaledVector(U, hy * 1.45));
    pts.push(P.clone().addScaledVector(F, d).addScaledVector(U, hy * 1.45), cs[0].clone().lerp(cs[1], 0.65));
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const m = new THREE.LineBasicMaterial({ color: 0xdee2e6, transparent: true, opacity: 0 });
    const L = new THREE.LineSegments(g, m);
    scene.add(L);
    const lab = h('div', 'sd-l sd-l-cam', 'camera ' + c.id);
    const lo = new CSS2DObject(lab); lo.position.copy(P).addScaledVector(U, 0.45); scene.add(lo);
    return { c, P, F, U, R, line: L, mat: m, lab };
  });

  const ringGeo = new THREE.RingGeometry(0.78, 0.96, 48);
  const objViz = objs.map((o) => {
    const C = V(o.center);
    const ringM = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
    const ring = new THREE.Mesh(ringGeo, ringM);
    ring.rotation.x = -Math.PI / 2; ring.position.set(C.x, 0.46, C.z); ring.renderOrder = 2;
    scene.add(ring);
    const fg = V(o.front_ground).normalize();
    const arrow = new THREE.ArrowHelper(fg, new THREE.Vector3(C.x, C.y + 0.65, C.z), 1.45, 0xffa94d, 0.42, 0.3);
    arrow.line.material.transparent = true; arrow.cone.material.transparent = true;
    arrow.line.material.opacity = 0; arrow.cone.material.opacity = 0; arrow.visible = false;
    scene.add(arrow);
    const lab = h('div', 'sd-l', '<b>' + o.id + '</b> ' + esc(shortOf(o.id)));
    const lo = new CSS2DObject(lab); lo.position.set(C.x, C.y + 0.75, C.z); scene.add(lo);
    const sc = h('div', 'sd-sc');
    const so = new CSS2DObject(sc); so.position.set(C.x, C.y + 1.55, C.z); scene.add(so);
    return { o, C, fg, ring, ringM, arrow, lab, sc };
  });

  function groundShade(fn, colorHex) {
    const g = new THREE.PlaneGeometry(20, 20, 100, 100);
    g.rotateX(-Math.PI / 2);
    const p = g.attributes.position;
    const col = new Float32Array(p.count * 4);
    const c = new THREE.Color(colorHex);
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), z = p.getZ(i);
      col[i * 4] = c.r; col[i * 4 + 1] = c.g; col[i * 4 + 2] = c.b; col[i * 4 + 3] = fn(x, z);
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 4));
    const m = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, depthWrite: false, depthTest: false, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(g, m);
    mesh.position.y = 0.42; mesh.renderOrder = 1; mesh.visible = false;
    scene.add(mesh);
    return { mesh, m };
  }
  function dashedLine(a, b, colorHex) {
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    const m = new THREE.LineDashedMaterial({ color: colorHex, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0, depthTest: false });
    const L = new THREE.Line(g, m); L.computeLineDistances(); L.visible = false; L.renderOrder = 3; scene.add(L); return { L, m };
  }
  const mv = objViz[mvId], pk = objViz[pkId];
  const cam0 = camObjs[0];
  const camFg = new THREE.Vector3(cam0.F.x, 0, cam0.F.z).normalize();
  /* frame 1: farther from camera 0 than the minivan */
  const shadeCam = groundShade((x, z) => {
    const depth = (x - mv.C.x) * camFg.x + (z - mv.C.z) * camFg.z;
    const lat = Math.abs((x - mv.C.x) * camFg.z - (z - mv.C.z) * camFg.x);
    return 0.42 * clamp01(0.5 + depth / 1.4) * clamp01(1 - (lat - 5) / 3) * clamp01(1 - (depth - 5) / 2);
  }, COL.cam);
  const perpCam = new THREE.Vector3(-camFg.z, 0, camFg.x);
  const lineCam = dashedLine(mv.C.clone().setY(0.3).addScaledVector(perpCam, -6), mv.C.clone().setY(0.3).addScaledVector(perpCam, 6), 0x74c0fc);
  /* frame 2: the pickup truck's own left (three.js is right-handed, y up: left = up x front) */
  const pkF = pk.fg.clone();
  const pkL = new THREE.Vector3(pkF.z, 0, -pkF.x);
  const shadeObj = groundShade((x, z) => {
    const dx = x - pk.C.x, dz = z - pk.C.z;
    const side = dx * pkL.x + dz * pkL.z;
    const r = Math.hypot(dx, dz);
    return 0.42 * clamp01(0.5 + side / 1.2) * clamp01(1 - (r - 6.5) / 2.5);
  }, COL.obj);
  const lineObj = dashedLine(pk.C.clone().setY(0.3).addScaledVector(pkF, -6), pk.C.clone().setY(0.3).addScaledVector(pkF, 7), 0xb197fc);

  /* axes of the pickup's frame: local +Z = front, local +X = left */
  const axG = new THREE.Group();
  const axMats = [];
  [[0, 0, 1, 'front', false], [0, 0, -1, 'behind', false], [1, 0, 0, 'left', true], [-1, 0, 0, 'right', false]].forEach(([x, y, z, name, big]) => {
    const a = new THREE.ArrowHelper(new THREE.Vector3(x, y, z), new THREE.Vector3(0, 0, 0), big ? 2.6 : 1.8,
      big ? 0x9775fa : 0xced4da, big ? 0.5 : 0.36, big ? 0.34 : 0.24);
    a.line.material.transparent = true; a.cone.material.transparent = true;
    axMats.push(a.line.material, a.cone.material);
    axG.add(a);
    const lab = h('div', 'sd-ax' + (big ? ' sd-ax-on' : ''), name);
    const lo = new CSS2DObject(lab); lo.position.set(x * (big ? 3.05 : 2.2), 0.05, z * (big ? 3.05 : 2.2));
    axG.add(lo);
    axMats.push(lab);
  });
  axG.position.set(pk.C.x, pk.C.y + 0.9, pk.C.z);
  axG.rotation.y = Math.atan2(pkF.x, pkF.z);
  axG.visible = false;
  scene.add(axG);
  function setAxes(v) {
    axG.visible = v > 0.01;
    axMats.forEach((m) => { if (m.style) setOp(m, v); else m.opacity = v; });
  }

  const avatar = new THREE.Group();
  const avM = new THREE.MeshLambertMaterial({ color: 0xb197fc, transparent: true, opacity: 0, emissive: 0x222222 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.24, 0.75, 20), avM); body.position.y = 0.38;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.21, 20, 14), avM); head.position.y = 0.98;
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.42, 16), avM); nose.rotation.x = Math.PI / 2; nose.position.set(0, 0.98, 0.33);
  avatar.add(body, head, nose);
  avatar.scale.setScalar(1.25);
  avatar.visible = false;
  scene.add(avatar);
  const avLab = h('div', 'sd-l sd-l-av', 'you, at the pickup');
  const avLo = new CSS2DObject(avLab); avLo.position.set(0, 1.75, 0); avatar.add(avLo);

  /* camera poses */
  const center = new THREE.Vector3(0, 0, -0.4);
  const poseCam = (k, dfov) => {
    const c = camObjs[k];
    return { pos: c.P.clone(), tgt: c.P.clone().addScaledVector(c.F, 9), up: c.U.clone(), fov: c.c.fov_y_deg + (dfov || 0) };
  };
  const poseBird = { pos: new THREE.Vector3(0.3, 13.5, 8.5), tgt: new THREE.Vector3(0, 0, -0.4), up: new THREE.Vector3(0, 1, 0), fov: 42 };
  const sedMid = new THREE.Vector3();
  [pkId, BL, BR, FR].forEach((i) => sedMid.add(objViz[i].C));
  sedMid.multiplyScalar(0.25).setY(0);
  const posePk = { pos: sedMid.clone().addScaledVector(pkF, -7.5).setY(15.5), tgt: sedMid.clone().addScaledVector(pkF, 0.6),
    up: new THREE.Vector3(0, 1, 0), fov: 46 };
  const poseBird2 = { pos: new THREE.Vector3(-3.5, 13, 9), tgt: new THREE.Vector3(0, 0, -0.5), up: new THREE.Vector3(0, 1, 0), fov: 42 };
  /* frame 1: just behind and above camera 0, looking the way camera 0 looks */
  const poseCam0 = { pos: cam0.P.clone().addScaledVector(camFg, -2.5).add(new THREE.Vector3(0, 2.2, 0)),
    tgt: mv.C.clone().setY(0).addScaledVector(camFg, 1.0), up: new THREE.Vector3(0, 1, 0), fov: 50 };

  function orbitLerp(a, b, t) {
    const da = a.clone().sub(center), db = b.clone().sub(center);
    const ra = Math.hypot(da.x, da.z), rb = Math.hypot(db.x, db.z);
    const aa = Math.atan2(da.z, da.x), ab = Math.atan2(db.z, db.x);
    let d = ab - aa; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
    const ang = aa + d * t, r = ra + (rb - ra) * t, y = a.y + (b.y - a.y) * t;
    return new THREE.Vector3(center.x + r * Math.cos(ang), y, center.z + r * Math.sin(ang));
  }
  function lerpPose(A, B, t, orbit) {
    const e = ease(clamp01(t));
    return {
      pos: orbit ? orbitLerp(A.pos, B.pos, e) : A.pos.clone().lerp(B.pos, e),
      tgt: A.tgt.clone().lerp(B.tgt, e),
      up: A.up.clone().lerp(B.up, e).normalize(),
      fov: A.fov + (B.fov - A.fov) * e
    };
  }

  /* ------------------------------------------------------------ steps
   * Each step: title, minimum length (reading time), sub-animations {name: [start, length]},
   * and caption parts [{at, html}] that appear when their part of the picture starts. */
  const andStart = (k) => 1.2 + 0.55 * k;     // AND dots for row k
  const andLen = 0.8;
  const selStart = (k) => 1.2 + 0.55 * cands.length + 0.3 + 0.4 * k;
  const selLen = 0.7;
  const steps = [
    { t: 'Views', min: 4.2, a: { v0: [0, 0.5], v1: [0.25, 0.5], v2: [0.5, 0.5], legend: [0.8, 0.5] },
      cap: [{ at: 0, html: 'SATURN starts from ' + data.views.length + ' views of one scene and a question that mixes two frames of reference: camera&nbsp;0&rsquo;s view and the pickup truck&rsquo;s own view.' }] },
    { t: 'Detect', min: 3.8, a: { boxes: [0.1, 1.5], count: [1.6, 0.4] },
      cap: [{ at: 0, html: 'SAM3 detects the objects in every view, and SATURN matches the detections across views into ' + N + ' objects.' }] },
    { t: '3D scene', min: 0, a: { cross: [0, 0.4], cloud: [0.2, 0.6], photo0: [0.6, 0.6], show: [0.9, 0.5],
        orbit01: [1.6, 1.5], photo1: [3.1, 0.8], orbit12: [3.9, 1.5], photo2: [5.4, 0.8], rise: [6.2, 1.5], arrows: [7.7, 0.5] },
      cap: [{ at: 0, html: 'VGGT builds one 3D scene and recovers the three cameras. Each photo lines up with the scene when seen from its own camera.' },
        { at: 7.7, html: ' Orient Anything&nbsp;V2 gives every object a facing direction (orange arrows).' }] },
    { t: 'Program', min: 3.4, a: { type: [0, 1.8] },
      cap: [{ at: 0, html: 'A code LLM writes the program. The program names the concepts and the two relations and says how to combine them.' }] },
    { t: 'Concepts', min: 3.8, a: { lines: [0, 0.3], rings: [0.3, 0.6], fill: [0.5, 0.6] },
      cap: [{ at: 0, html: 'The VLM scores every object for each concept. It finds three sedans, one minivan and one pickup truck.' }] },
    { t: 'Camera 0', min: 6.2, a: { move: [0, 1.5], code: [0, 0.3], shade: [1.5, 0.6], scores: [2.2, 0.4], fill: [2.2, 0.7] },
      cap: [{ at: 0, html: 'Frame 1 is camera&nbsp;0&rsquo;s view. <span class="sd-c-cam">Behind the minivan</span> means farther from camera&nbsp;0 than the minivan.' },
        { at: 2.2, html: ' Back-left sedan ' + v2(S.behind_minivan_cam0[BL]) + ', back-right ' + v2(S.behind_minivan_cam0[BR]) + ', front ' + v2(S.behind_minivan_cam0[FR]) + '.' }] },
    { t: 'Pickup frame', min: 7.0, a: { move: [0, 1.6], code: [0, 0.3], walk: [0.3, 1.3], axes: [1.7, 0.5], shade: [2.1, 0.6], scores: [2.8, 0.4], fill: [2.8, 0.7] },
      cap: [{ at: 0, html: 'Frame 2 is the pickup truck&rsquo;s own view: stand at the pickup and face where it faces. <span class="sd-c-obj">Left of the pickup</span> is that observer&rsquo;s left.' },
        { at: 2.8, html: ' Back-left sedan ' + v2(S.left_of_pickup_pickup_view[BL]) + ', back-right ' + v2(S.left_of_pickup_pickup_view[BR]) + ', front ' + v2(S.left_of_pickup_pickup_view[FR]) + '.' }] },
    { t: 'Soft AND', min: 6.6, a: { move: [0, 1.2], code: [0, 0.3], avatarOut: [0, 0.4], win: [selStart(cands.length - 1) + selLen, 0.4] },
      cap: [{ at: 0, html: 'Soft AND keeps each sedan&rsquo;s weakest score, so a sedan must pass both frames.' },
        { at: andStart(cands.length - 1) + andLen, html: ' Back-left ' + v2(S.joint_min[BL]) + ', back-right ' + v2(S.joint_min[BR]) + ', front ' + v2(S.joint_min[FR]) + '.' },
        { at: selStart(cands.length - 1) + selLen, html: ' The back-left sedan scores highest.' }] },
    { t: 'Answer', min: 6.4, a: { cross: [0, 0.4], gt: [0.5, 0.3], sat: [0.9, 0.3], base: [1.4, 0.3 * featured.length], cards: [2.2, 0.5] },
      cap: [{ at: 0, html: 'SATURN returns object ' + ansId + ', the back-left sedan, which is the ground truth (IoU ' + data.saturn.iou_gt.toFixed(2) + ').' },
        { at: 1.4, html: ' Qwen3.5-9B and Gemini-3.1-Pro pick the back-right sedan, which is not on the pickup&rsquo;s left; GPT-5.1&rsquo;s box lands beside the target.' }] }
  ];
  /* step 8 dots are part of the step's own clock */
  rows.forEach((r, k) => { steps[7].a['and' + k] = [andStart(k), andLen]; steps[7].a['sel' + k] = [selStart(k), selLen]; });
  steps.forEach((st) => {
    let end = 0;
    for (const k in st.a) end = Math.max(end, st.a[k][0] + st.a[k][1]);
    st.end = end;                                    // everything has stopped moving here
    st.len = Math.max(end + HOLD, st.min);
  });
  const starts = [];
  let total = 0;
  steps.forEach((st) => { starts.push(total); total += st.len; });

  steps.forEach((st, i) => {
    const b = h('button', 'sd-pill', '<span>' + (i + 1) + '</span><em>' + st.t + '</em>');
    b.type = 'button'; b.setAttribute('aria-label', 'Step ' + (i + 1) + ': ' + st.t);
    b.addEventListener('click', () => jumpTo(i));
    pills.appendChild(b);
    st.pill = b;
    if (i > 0) { const tick = h('b', 'sd-tick'); tick.style.left = (100 * starts[i] / total).toFixed(3) + '%'; prog.appendChild(tick); }
  });

  /* ------------------------------------------------------------ render one moment */
  let ctlActive = false;
  const curPose = { pos: new THREE.Vector3(), tgt: new THREE.Vector3() };

  function render(i, t) {
    const st = steps[i];
    const A = (name, step) => {             // progress of a sub-animation; earlier steps are complete
      const k = step === undefined ? i : step;
      if (k < i) return 1;
      if (k > i) return 0;
      const a = steps[k].a[name];
      return a ? clamp01((t - a[0]) / a[1]) : 0;
    };
    const past = (k) => i > k;              // step k is complete

    /* layers */
    const cross3d = i === 2 ? A('cross') : i >= 3 && i <= 7 ? 1 : i === 8 ? 1 - A('cross') : 0;
    setOp(viewsEl, i <= 1 ? 1 : i === 2 ? 1 - A('cross') : 0);
    setOp(l3d, cross3d);
    setOp(lans, i === 8 ? A('cross') : 0);
    setCls(viewsEl, 'sd-live', i <= 1); setCls(l3d, 'sd-live', i >= 2 && i <= 7); setCls(lans, 'sd-live', i === 8);
    setHTML(stageTitle, '<span>' + (i + 1) + '</span>' + st.t);

    /* views and detections */
    viewFigs.forEach((vf, k) => {
      setOp(vf.f, A('v' + k, 0));
      const nb = vf.boxes.length;
      vf.boxes.forEach((g, j) => {
        const p = i > 1 ? 1 : i < 1 ? 0 : clamp01((t - (0.1 + (1.2 * (k * nb + j)) / (data.views.length * nb))) / 0.3);
        setOp(g, p);
      });
    });
    setOp(legend, A('legend', 0));
    setOp(legendDet, A('count', 1));

    /* 3D */
    cmat.opacity = A('cloud', 2);
    let ph = null;
    if (i === 2) {
      if (t < 1.2) ph = { k: 0, a: 1 - A('photo0') };
      else if (t >= 3.1 && t < 3.9) ph = { k: 1, a: Math.sin(Math.PI * A('photo1')) };
      else if (t >= 5.4 && t < 6.2) ph = { k: 2, a: Math.sin(Math.PI * A('photo2')) };
    }
    if (ph) { if (!photo.src.endsWith(data.views[ph.k].file)) photo.src = base + data.views[ph.k].file; setOp(photo, ph.a); }
    else setOp(photo, 0);
    const ac = i <= 1 ? 0 : i === 2 ? (t < 3.1 ? 0 : t < 5.4 ? 1 : t < 6.2 ? 2 : -1) : i === 5 ? 0 : -1;
    const showCams = A('show', 2);
    const eye = poseAt(i, t).pos;
    camObjs.forEach((c, k) => {
      const inside = ctlActive ? 1 : clamp01((eye.distanceTo(c.P) - 0.3) / 1.2);
      c.mat.opacity = showCams * (k === ac ? 1 : 0.55) * inside;
      c.line.visible = c.mat.opacity > 0.01;
      c.mat.color.set(k === ac ? 0xffd43b : 0xdee2e6);
      setOp(c.lab, showCams * inside);
      setCls(c.lab, 'sd-on', k === ac);
    });
    stripImgs.forEach((w, k) => setCls(w, 'sd-on', k === ac));
    setOp(strip, i >= 2 && i <= 7 ? 1 : 0);

    const showObj = A('show', 2), arrows = A('arrows', 2), sem = A('rings', 4);
    objViz.forEach((v) => {
      const id = v.o.id;
      const role = id === mvId ? 'cam' : id === pkId ? 'obj' : S.is_sedan[id] > 0.5 ? 'g' : '';
      const c = role === 'cam' ? 0x4dabf7 : role === 'obj' ? 0x9775fa : role === 'g' ? 0x51cf66 : 0xffffff;
      let col = new THREE.Color(0xffffff).lerp(new THREE.Color(c), sem).getHex();
      let op = (role ? 0.55 + 0.4 * sem : 0.55 - 0.35 * sem) * showObj;
      const win = (i === 7 && A('win') > 0.5) || i === 8;
      if (win && id === ansId) { col = 0x51cf66; op = showObj; }
      v.ringM.color.setHex(col); v.ringM.opacity = op;
      const ar = arrows * (i === 6 ? (id === pkId ? 1 : 0.2) : 1);
      v.arrow.line.material.opacity = ar; v.arrow.cone.material.opacity = ar; v.arrow.visible = ar > 0.01;
      setOp(v.lab, showObj * (role ? 1 : 1 - 0.55 * sem));
      setCls(v.lab, 'sd-l-g', role === 'g' && sem > 0.5);
      setCls(v.lab, 'sd-l-b', role === 'obj' && sem > 0.5);
      setCls(v.lab, 'sd-l-s', role === 'cam' && sem > 0.5);
      setCls(v.lab, 'sd-l-win', win && id === ansId);
    });

    /* frame 1: camera 0 */
    const f1 = i === 5 ? A('shade') : 0;
    shadeCam.m.opacity = f1; shadeCam.mesh.visible = f1 > 0.01;
    lineCam.m.opacity = 0.9 * f1; lineCam.L.visible = f1 > 0.01;
    /* frame 2: the pickup's own view */
    const f2 = i === 6 ? A('shade') : 0;
    shadeObj.m.opacity = f2; shadeObj.mesh.visible = f2 > 0.01;
    lineObj.m.opacity = 0.9 * f2; lineObj.L.visible = f2 > 0.01;
    setAxes(i === 6 ? A('axes') : 0);
    let avOp = 0;
    if (i === 6) {
      avOp = clamp01(A('walk') / 0.25);
      const w = ease(A('walk'));
      const start = pk.C.clone().addScaledVector(pkF, -4).setY(0.3);
      const end = pk.C.clone().setY(pk.C.y + 0.75);
      avatar.position.copy(start.lerp(end, w));
      avatar.rotation.y = Math.atan2(pkF.x, pkF.z);
    } else if (i === 7) {
      avOp = 1 - A('avatarOut');
      avatar.position.copy(pk.C.clone().setY(pk.C.y + 0.75));
    }
    avM.opacity = avOp; avatar.visible = avOp > 0.01; setOp(avLab, avOp);

    /* scores floating over the candidates */
    objViz.forEach((v) => {
      const id = v.o.id, k = cands.indexOf(id);
      let txt = '', op = 0, cls = '';
      if (k >= 0) {
        if (i === 5) { op = A('scores'); txt = v2(S.behind_minivan_cam0[id]); cls = 'sd-sc-cam'; }
        else if (i === 6) { op = A('scores'); txt = v2(S.left_of_pickup_pickup_view[id]); cls = 'sd-sc-obj'; }
        else if (i === 7) { op = t >= andStart(k) + andLen ? 1 : 0; txt = 'AND ' + v2(S.joint_min[id]); cls = id === ansId && A('win') > 0.5 ? 'sd-sc-win' : 'sd-sc-and'; }
      }
      setText(v.sc, txt); setOp(v.sc, op);
      if (v.sc.dataset.c !== cls) { v.sc.className = 'sd-sc ' + cls; v.sc.dataset.c = cls; }
    });

    /* HUD */
    let hudT = '';
    if (i === 2) hudT = ac >= 0 ? 'seen from camera ' + ac : '';
    else if (i === 5) hudT = 'frame 1: camera 0&rsquo;s view';
    else if (i === 6) hudT = 'frame 2: the pickup truck&rsquo;s own view';
    if (hud.dataset.t !== hudT) { hud.innerHTML = hudT; hud.dataset.t = hudT; }
    setCls(hud, 'sd-hud-cam', i === 5); setCls(hud, 'sd-hud-obj', i === 6);

    /* program: nothing before typing starts; highlights only on typed lines */
    const typed = i < 3 ? 0 : i === 3 ? Math.round(totalChars * A('type')) : totalChars;
    showTyped(typed);
    const active = new Set();
    if (i === 4) [0, 1, 2].forEach((k) => active.add(k));
    if (i === 5) { active.add(3); active.add(6); }
    if (i === 6) active.add(6);
    if (i === 7) active.add(6);
    if (i === 8) active.add(7);
    lineEls.forEach((el, k) => {
      const on = active.has(k) && el.__typed;
      setCls(el, 'sd-on', on);
      setCls(el, 'sd-dim', active.size > 0 && typed === totalChars && !active.has(k));
    });
    setCls(pre, 'sd-mk0', i === 4); setCls(pre, 'sd-mk1', i === 5); setCls(pre, 'sd-mk2', i === 6); setCls(pre, 'sd-mk3', i === 7);

    /* table */
    const fills = [A('fill', 4), A('fill', 5), A('fill', 6)];
    rows.forEach((r, k) => {
      r.cells.forEach((c, ci) => {
        let f;
        if (ci < 3) f = fills[ci];
        else f = i > 7 ? 1 : i < 7 ? 0 : t >= andStart(k) + andLen ? clamp01((t - andStart(k) - andLen) / 0.2) : 0;
        setStyle(c.bar, 'width', (100 * c.v * ease(f)).toFixed(1) + '%');
        setText(c.num, f > 0 ? v2(c.v) : '');
        setOp(c.num, f > 0 ? 1 : 0);
      });
      setCls(r.r, 'sd-win', ((i === 7 && A('win') > 0.5) || i === 8) && r.i === ansId);
    });
    selRows.forEach((r, k) => {
      const f = i > 7 ? 1 : i < 7 ? 0 : t >= selStart(k) + selLen ? clamp01((t - selStart(k) - selLen) / 0.25) : 0;
      setStyle(r.bar, 'width', (100 * r.val * ease(f)).toFixed(1) + '%');
      setText(r.v, f > 0 ? v2(r.val) : '');
      setCls(r.d, 'sd-win', ((i === 7 && A('win') > 0.5) || i === 8) && r.i === ansId);
    });
    setCls(barsP, 'sd-live1', i === 5); setCls(barsP, 'sd-live2', i === 6); setCls(barsP, 'sd-live3', i === 7);

    /* dots: travel while the composition happens; arrival = the value appears */
    flows.forEach((fl) => {
      let p = -1;
      if (i === 7) {
        const st0 = fl.kind === 'and' ? andStart(fl.row) : selStart(fl.row);
        const ln = fl.kind === 'and' ? andLen : selLen;
        if (t >= st0 && t <= st0 + ln + 0.15) p = (t - st0) / ln;
      }
      if (p < 0) { setStyle(fl.el, 'opacity', '0'); return; }
      const a = centerIn(fl.from(), barsP), b = centerIn(fl.to(), barsP);
      const e = ease(Math.min(1, p));
      const x = a[0] + (b[0] - a[0]) * e, y = a[1] + (b[1] - a[1]) * e - Math.sin(Math.PI * Math.min(1, p)) * 10;
      const merge = p > 1 ? 1 - (p - 1) / 0.15 : 1;      // the dot dissolves into the value it produced
      setStyle(fl.el, 'transform', 'translate(' + x.toFixed(1) + 'px,' + y.toFixed(1) + 'px) scale(' + (0.6 + 0.4 * merge).toFixed(2) + ')');
      setStyle(fl.el, 'opacity', merge.toFixed(2));
    });

    /* answer layer */
    setOp(gGT, i === 8 ? A('gt') : 0);
    setOp(gSat, i === 8 ? A('sat') : 0);
    gBase.forEach((g, k) => setOp(g, i === 8 ? clamp01((t - 1.4 - 0.3 * k) / 0.3) : 0));
    setOp(verdict, i === 8 ? 0.2 + 0.8 * A('cards') : 0.2);
    setCls(verdict, 'sd-v-on', i === 8 && A('cards') > 0.5);

    /* caption parts, in step with the picture */
    let cap = '';
    st.cap.forEach((c) => { if (t >= c.at) cap += c.html; });
    setHTML(capEl, cap);

    /* steps */
    steps.forEach((s2, k) => { setCls(s2.pill, 'sd-on', k === i); setCls(s2.pill, 'sd-done', k < i); });

    /* camera */
    if (threeOK && !ctlActive) {
      const ps = poseAt(i, t);
      camera.position.copy(ps.pos); camera.up.copy(ps.up); camera.lookAt(ps.tgt);
      if (Math.abs(camera.fov - ps.fov) > 1e-3) { camera.fov = ps.fov; camera.updateProjectionMatrix(); }
      curPose.pos.copy(ps.pos); curPose.tgt.copy(ps.tgt);
    }
  }

  function poseAt(i, t) {
    const A = (name) => { const a = steps[i].a[name]; return a ? clamp01((t - a[0]) / a[1]) : 0; };
    if (i <= 1) return poseCam(0);
    if (i === 2) {
      if (t < steps[2].a.orbit01[0]) return poseCam(0);
      if (t < steps[2].a.orbit12[0]) return lerpPose(poseCam(0), poseCam(1), A('orbit01'), true);
      if (t < steps[2].a.rise[0]) return lerpPose(poseCam(1), poseCam(2), A('orbit12'), true);
      return lerpPose(poseCam(2), poseBird, A('rise'), false);
    }
    if (i === 3 || i === 4) return poseBird;
    if (i === 5) return lerpPose(poseBird, poseCam0, A('move'), false);
    if (i === 6) return lerpPose(poseCam0, posePk, A('move'), false);
    if (i === 7) return lerpPose(posePk, poseBird2, A('move'), false);
    return poseBird2;
  }

  function resize() {
    if (!threeOK) return;
    const w = l3d.clientWidth || 600, hh = l3d.clientHeight || 450;
    gl.setSize(w, hh, false);
    labels.setSize(w, hh);
    camera.aspect = w / hh; camera.updateProjectionMatrix();
    dirty = true; kick();
  }

  /* ------------------------------------------------------------ the clock */
  let T = 0, playing = false, last = 0, visible = false, started = false, raf = 0, dirty = true;
  const stepAt = (time) => { let i = 0; while (i < steps.length - 1 && time >= starts[i + 1]) i++; return i; };
  const endOf = (i) => starts[i] + steps[i].end;     // the moment step i has finished moving
  function setPlayBtn() {
    bPlay.innerHTML = playing ? '<span aria-hidden="true">&#10074;&#10074;</span>' : '<span aria-hidden="true">&#9654;</span>';
    bPlay.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    root.classList.toggle('sd-playing', playing);
  }
  function releaseControls() { if (controls) { controls.enabled = false; } ctlActive = false; }
  function play() {
    if (T >= total) T = 0;
    releaseControls();
    playing = true; started = true; last = performance.now(); setPlayBtn(); dirty = true; kick();
  }
  function pause() {
    playing = false; setPlayBtn();
    if (controls) { controls.target.copy(curPose.tgt); controls.enabled = true; controls.update(); }
    dirty = true; kick();
  }
  function jumpTo(i) {
    i = Math.max(0, Math.min(steps.length - 1, i));
    releaseControls();
    T = endOf(i); last = performance.now(); dirty = true;
    if (!playing && controls) { draw(); controls.target.copy(curPose.tgt); controls.enabled = true; controls.update(); }
    kick();
  }
  bPlay.addEventListener('click', () => (playing ? pause() : play()));
  bNext.addEventListener('click', () => jumpTo(stepAt(T) + 1));
  bPrev.addEventListener('click', () => jumpTo(stepAt(T) - 1));
  bRe.addEventListener('click', () => { releaseControls(); T = 0; play(); });
  if (threeOK) {
    controls.addEventListener('start', () => { ctlActive = true; dirty = true; kick(); });
    controls.addEventListener('change', () => { dirty = true; kick(); });
  }

  function draw() {
    const i = stepAt(Math.min(T, total - 1e-6));
    const local = Math.min(T, total) - starts[i];
    const tt = reduce ? Math.max(local, steps[i].end) : local;   // reduced motion: end states only
    render(i, Math.min(tt, steps[i].len));
    setStyle(progBar, 'width', (100 * Math.min(T, total) / total).toFixed(3) + '%');
    if (threeOK) { gl.render(scene, camera); labels.render(scene, camera); }
  }
  function frame(now) {
    raf = 0;
    if (playing) {
      T += Math.min(100, now - last) / 1000;
      last = now;
      if (T >= total) { T = total; playing = false; setPlayBtn(); if (controls) { controls.target.copy(curPose.tgt); controls.enabled = true; } }
      dirty = true;
    }
    if (dirty) { draw(); dirty = false; }
    if (visible && playing) raf = requestAnimationFrame(frame);
  }
  function kick() { if (!raf && visible) raf = requestAnimationFrame(frame); }

  new ResizeObserver(resize).observe(l3d);
  new ResizeObserver(() => { reserveProgramHeight(); dirty = true; kick(); }).observe(codeP);
  new IntersectionObserver((ents) => {
    ents.forEach((e) => {
      visible = e.isIntersecting;
      if (visible) {
        last = performance.now();
        if (!started && !reduce) play(); else { dirty = true; kick(); }
      } else if (raf) { cancelAnimationFrame(raf); raf = 0; }
    });
  }, { threshold: 0.3 }).observe(root);

  if (reduce) { T = endOf(0); root.classList.add('sd-reduced'); }
  setPlayBtn();
  resize();
  draw();

  /* hooks for automated checks */
  root.__sd = {
    go: (i, p) => { started = true; playing = false; setPlayBtn(); releaseControls(); T = starts[i] + (p === undefined ? steps[i].end : p * steps[i].len); dirty = true; draw(); },
    time: () => T, total, starts, steps: steps.length, playing: () => playing
  };
}

const rootEl = document.getElementById('sat-demo');
if (rootEl) {
  main(rootEl).catch((err) => {
    rootEl.classList.add('sd-failed');
    const m = document.createElement('p');
    m.className = 'sd-err';
    m.textContent = 'The animation could not load (' + (err && err.message ? err.message : err) + ').';
    rootEl.appendChild(m);
    if (window.console) console.error(err);
  });
}
