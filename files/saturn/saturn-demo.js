/*
 * SATURN project page: animated walkthrough of one real example.
 * Every value shown (images, detections, 3D scene, program, soft scores,
 * baseline answers) is read from example.json and cloud.bin, which were
 * exported from an actual SATURN run and the paper's baseline result files.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const COL = { target: '#2f9e44', f1: '#7048e8', f2: '#1c7ed6', bad: '#e03131', det: '#15aabf', answer: '#2f9e44' };
const TSHORT = { 'green tank (left)': 'tank, left', 'green tank (right)': 'tank, right', 'green sedan': 'sedan' };
const SHORT = {
  'yellow SUV': 'SUV', 'yellow pickup truck': 'yellow pickup', 'green tank (left)': 'green tank L',
  'brown pickup truck': 'brown pickup', 'green sedan': 'green sedan', 'green tank (right)': 'green tank R',
  'purple double-decker bus': 'bus'
};

const clamp01 = (x) => Math.max(0, Math.min(1, x));
const seg = (p, a, b) => clamp01((p - a) / (b - a));
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const fmt = (x) => (x >= 0.095 || x === 0 ? x.toFixed(2) : x.toFixed(3));
const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);

function h(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const svgNS = 'http://www.w3.org/2000/svg';
function s(tag, attrs) {
  const e = document.createElementNS(svgNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

/* small cache so per-frame updates only touch the DOM when a value changes */
const _cache = new WeakMap();
function setStyle(el, prop, val) {
  let c = _cache.get(el);
  if (!c) { c = {}; _cache.set(el, c); }
  if (c[prop] !== val) { c[prop] = val; el.style[prop] = val; }
}
const setOp = (el, v) => setStyle(el, 'opacity', String(Math.round(v * 1000) / 1000));
function setCls(el, cls, on) { if (el.classList.contains(cls) !== on) el.classList.toggle(cls, on); }
function setText(el, t) { if (el.textContent !== t) el.textContent = t; }

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
  const busId = S.bus, suvId = S.suv, ansId = S.answer;
  const cands = objs.filter((o) => S.is_green[o.id] > 0.5).map((o) => o.id)
    .sort((a, b) => S.iota[b] - S.iota[a]);
  const nameOf = (i) => objs[i].name;
  const shortOf = (i) => SHORT[objs[i].name] || objs[i].name;
  const leftTank = ansId;
  const rightTank = cands.find((i) => i !== ansId && objs[i].name.includes('tank'));
  const sedan = cands.find((i) => objs[i].name.includes('sedan'));
  const featured = data.baselines.filter((b) => b.quote);
  const nWrong = data.baselines.filter((b) => !b.correct).length;

  /* ------------------------------------------------------------ DOM */
  const q = data.question;
  const p1 = "behind the double-decker bus (the double-decker bus's view)";
  const p2 = 'behind the suv (camera 0 view)';
  let qh = esc(q);
  qh = qh.replace('green object', '<mark class="sd-m-t">green object</mark>');
  qh = qh.replace(esc(p1), '<mark class="sd-m-f1">' + esc(p1) + '</mark>');
  qh = qh.replace(esc(p2), '<mark class="sd-m-f2">' + esc(p2) + '</mark>');
  const qbar = h('div', 'sd-q', '<span class="sd-q-tag">Question</span><span class="sd-q-text">' + qh + '</span>' +
    '<span class="sd-q-meta">3D FORCE REF &middot; ' + data.views.length + ' views &middot; ' + data.hops +
    ' relations &middot; object frame + camera frame</span>');
  root.appendChild(qbar);

  const main = h('div', 'sd-main');
  const stage = h('div', 'sd-stage');
  stage.setAttribute('role', 'img');
  stage.setAttribute('aria-label', 'Animated walkthrough of SATURN answering the question above');
  const side = h('div', 'sd-side');
  main.appendChild(stage); main.appendChild(side);
  root.appendChild(main);

  /* views grid (2x2 of 4:3 cells inside a 4:3 stage) */
  const viewsEl = h('div', 'sd-layer sd-views');
  const viewFigs = [];
  data.views.forEach((v, k) => {
    const f = h('figure', 'sd-view');
    const img = h('img'); img.src = base + v.file; img.alt = 'Camera ' + k + ' view'; img.decoding = 'async';
    const svg = s('svg', { viewBox: '0 0 1024 768', preserveAspectRatio: 'none', class: 'sd-ov' });
    const boxes = [];
    objs.forEach((o) => {
      const b = o.boxes[String(k)];
      if (!b) return;
      const x1 = Math.max(0, b[0]), y1 = Math.max(0, b[1]), x2 = Math.min(1024, b[2]), y2 = Math.min(768, b[3]);
      const g = s('g', { class: 'sd-det', 'data-obj': o.id });
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
    viewFigs.push({ f, boxes });
  });
  const legend = h('div', 'sd-view sd-legend');
  legend.innerHTML = '<div class="sd-legend-in"><div class="sd-legend-big">' + data.views.length + ' views</div>' +
    '<div>of one rendered scene. The question explicitly names both frames of reference: the bus&rsquo;s own view and camera&nbsp;0&rsquo;s view.</div><div class="sd-legend-det"><b>' + N + '</b> objects detected and matched across views</div></div>';
  viewsEl.appendChild(legend);
  stage.appendChild(viewsEl);

  /* 3D layer */
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

  /* answer layer */
  const lans = h('div', 'sd-layer sd-ans');
  const ansImg = h('img'); ansImg.src = base + data.views[0].file; ansImg.alt = 'Camera 0 with the answers drawn';
  const ansSvg = s('svg', { viewBox: '0 0 1024 768', preserveAspectRatio: 'none', class: 'sd-ov' });
  const gtb = data.gt.bbox_cam0;
  const gGT = s('g', { class: 'sd-a-gt' });
  gGT.appendChild(s('rect', { x: gtb[0] - 6, y: gtb[1] - 6, width: gtb[2] - gtb[0] + 12, height: gtb[3] - gtb[1] + 12 }));
  const sb = data.saturn.predicted_bbox_cam0;
  const gSat = s('g', { class: 'sd-a-sat' });
  gSat.appendChild(s('rect', { x: sb[0], y: sb[1], width: sb[2] - sb[0], height: sb[3] - sb[1] }));
  const satTagBg = s('rect', { x: sb[0], y: sb[3] + 6, width: 196, height: 46, class: 'sd-a-tagbg' });
  const satTag = s('text', { x: sb[0] + 12, y: sb[3] + 40, class: 'sd-a-tag' }); satTag.textContent = 'SATURN ✓';
  gSat.appendChild(satTagBg); gSat.appendChild(satTag);
  const gBase = featured.map((b, k) => {
    const x1 = b.box1000[0] * 1.024, y1 = b.box1000[1] * 0.768, x2 = b.box1000[2] * 1.024, y2 = b.box1000[3] * 0.768;
    const g = s('g', { class: 'sd-a-base' });
    g.appendChild(s('rect', { x: x1, y: y1, width: x2 - x1, height: y2 - y1 }));
    const tx = x2 - 40 - k * 46, ty = y2 + 4;
    g.appendChild(s('circle', { cx: tx + 20, cy: Math.min(ty + 20, 745), r: 19 }));
    const t = s('text', { x: tx + 20, y: Math.min(ty + 20, 745) + 9 }); t.textContent = String(k + 1);
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

  /* program panel */
  const codeP = h('div', 'sd-panel sd-code');
  codeP.appendChild(h('div', 'sd-panel-h', 'Program <span>written by the code LLM, run by SATURN</span>'));
  const pre = h('pre', 'sd-pre');
  const lineEls = data.program.map((ln) => { const d = h('div', 'sd-ln'); pre.appendChild(d); return d; });
  codeP.appendChild(pre);
  side.appendChild(codeP);

  /* soft truth values panel */
  const barsP = h('div', 'sd-panel sd-bars');
  barsP.appendChild(h('div', 'sd-panel-h', 'Soft truth values <span>per candidate, from the execution trace</span>'));
  const tbl = h('div', 'sd-tbl');
  tbl.appendChild(h('div', 'sd-tr sd-th',
    '<div>candidate</div><div>green?</div><div class="sd-c-f1">behind bus<small>bus&rsquo;s view</small></div>' +
    '<div class="sd-c-f2">behind SUV<small>camera 0</small></div><div>AND<small>= min</small></div>'));
  const rows = cands.map((i) => {
    const r = h('div', 'sd-tr');
    r.appendChild(h('div', 'sd-tname', '<b>' + i + '</b> ' + esc(TSHORT[nameOf(i)] || nameOf(i))));
    const cells = ['is_green', 'behind_bus_bus_view', 'behind_suv_cam0', 'joint_min'].map((key, ci) => {
      const c = h('div', 'sd-cell' + (ci === 1 ? ' sd-c-f1' : ci === 2 ? ' sd-c-f2' : ci === 3 ? ' sd-c-and' : ''));
      const bar = h('i'); const num = h('span');
      c.appendChild(bar); c.appendChild(num);
      r.appendChild(c);
      return { bar, num, v: S[key][i] };
    });
    tbl.appendChild(r);
    return { r, cells, i };
  });
  barsP.appendChild(tbl);
  const sel = h('div', 'sd-sel');
  const selRows = cands.map((i) => {
    const d = h('div', 'sd-sel-r', '<span class="sd-sel-n"><b>' + i + '</b> ' + esc(TSHORT[nameOf(i)] || shortOf(i)) + '</span><span class="sd-sel-bar"><i></i></span><span class="sd-sel-v"></span>');
    sel.appendChild(d);
    return { d, bar: d.querySelector('i'), v: d.querySelector('.sd-sel-v'), val: S.iota[i] };
  });
  barsP.appendChild(h('div', 'sd-sel-h', 'Selection over objects <span>(AND divided by its maximum, then argmax)</span>'));
  barsP.appendChild(sel);
  barsP.appendChild(h('div', 'sd-foot', (N - cands.length) + ' other objects are not green, so every AND is 0 for them.'));
  side.appendChild(barsP);

  /* controls */
  const ctrl = h('div', 'sd-ctrl');
  const pills = h('div', 'sd-pills');
  const capEl = h('p', 'sd-cap'); capEl.setAttribute('aria-live', 'polite');
  const btns = h('div', 'sd-btns');
  const bPrev = h('button', 'sd-b', '<span aria-hidden="true">&#9664;&#9664;</span>'); bPrev.setAttribute('aria-label', 'Previous step');
  const bPlay = h('button', 'sd-b sd-b-main', ''); bPlay.setAttribute('aria-label', 'Play');
  const bNext = h('button', 'sd-b', '<span aria-hidden="true">&#9654;&#9654;</span>'); bNext.setAttribute('aria-label', 'Next step');
  const bRe = h('button', 'sd-b', '<span aria-hidden="true">&#8635;</span>'); bRe.setAttribute('aria-label', 'Replay from the start');
  [bPrev, bPlay, bNext, bRe].forEach((b) => { b.type = 'button'; btns.appendChild(b); });
  const prog = h('div', 'sd-prog', '<i></i>');
  const progBar = prog.querySelector('i');
  ctrl.appendChild(pills);
  const row = h('div', 'sd-ctrl-row'); row.appendChild(capEl); row.appendChild(btns);
  ctrl.appendChild(row); ctrl.appendChild(prog);
  root.appendChild(ctrl);

  /* verdict */
  const verdict = h('div', 'sd-verdict');
  const satIoU = data.saturn.iou_gt.toFixed(2);
  verdict.appendChild(h('div', 'sd-card sd-card-sat',
    '<div class="sd-card-h"><span class="sd-ok">&#10003;</span> SATURN</div>' +
    '<div>Object ' + ansId + ', the <b>' + esc(nameOf(ansId)) + '</b>.</div>' +
    '<div class="sd-card-m">IoU ' + satIoU + ' with the ground-truth box. Backbone: Qwen3-VL-8B.</div>'));
  featured.forEach((b, k) => {
    verdict.appendChild(h('div', 'sd-card sd-card-bad',
      '<div class="sd-card-h"><span class="sd-num">' + (k + 1) + '</span> ' + esc(b.model) + ' <span class="sd-x">&#10007;</span></div>' +
      '<blockquote>&ldquo;' + esc(b.quote) + '&rdquo;</blockquote>' +
      '<div class="sd-card-m">IoU ' + (b.iou_gt === null ? 'n/a' : b.iou_gt.toFixed(2)) + ' with the ground truth</div>'));
  });
  verdict.appendChild(h('div', 'sd-vnote', '' + nWrong + ' of the ' + data.baselines.length + ' VLMs in the paper&rsquo;s Table&nbsp;1 answer this question incorrectly (IoU &le; 0.5). Quotes reproduce the models&rsquo; answers verbatim.'));
  root.appendChild(verdict);

  /* ------------------------------------------------------------ program rendering */
  const ln6 = data.program[6] || '';
  const spans6 = [
    ["is_green('x3')", 'sd-mk-t'],
    ["scene.obj_behind('x3', 'x1') & bus('x1')", 'sd-mk-f1'],
    ["cam0_behind('x3', 'x2') & suv('x2')", 'sd-mk-f2'],
    [".iota('x3')", 'sd-mk-io']
  ];
  function hiPy(t) {
    let x = esc(t);
    x = x.replace(/('[^']*')/g, '<span class="sd-s">$1</span>');
    x = x.replace(/\b(return|int)\b/g, '<span class="sd-k">$1</span>');
    x = x.replace(/\b(score|iota|argmax)\b/g, '<span class="sd-f">$1</span>');
    return x;
  }
  function line6Html() {
    let out = ''; let pos = 0;
    const marks = spans6.map(([sub, cls]) => ({ at: ln6.indexOf(sub), sub, cls })).filter((m) => m.at >= 0).sort((a, b) => a.at - b.at);
    for (const m of marks) {
      if (m.at < pos) continue;
      out += hiPy(ln6.slice(pos, m.at)) + '<span class="' + m.cls + '">' + hiPy(m.sub) + '</span>';
      pos = m.at + m.sub.length;
    }
    return out + hiPy(ln6.slice(pos));
  }
  const fullHtml = data.program.map((t, i) => (i === 6 ? line6Html() : hiPy(t)));
  const totalChars = data.program.reduce((a, t) => a + t.length, 0);
  let typedShown = -1;
  function showTyped(nChars) {
    if (nChars === typedShown) return;
    typedShown = nChars;
    let left = nChars;
    data.program.forEach((t, i) => {
      if (left >= t.length && left > 0) { lineEls[i].innerHTML = fullHtml[i] || '&nbsp;'; left -= t.length; }
      else if (left > 0) { lineEls[i].innerHTML = esc(t.slice(0, left)) + '<span class="sd-caret"></span><span class="sd-ghost">' + esc(t.slice(left)) + '</span>'; left = 0; }
      else { lineEls[i].innerHTML = '<span class="sd-ghost">' + esc(t) + '</span>'; }
    });
  }

  /* ------------------------------------------------------------ 3D */
  let gl = null;
  try {
    gl = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' });
  } catch (e) { gl = null; }
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
    controls.enableDamping = true;
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

  /* point cloud */
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
  const cloud = new THREE.Points(cg, cmat);
  scene.add(cloud);

  /* cameras (frusta) */
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

  /* objects: ground ring, facing arrow, labels */
  const ringGeo = new THREE.RingGeometry(0.78, 0.96, 48);
  const objViz = objs.map((o) => {
    const C = V(o.center);
    const ringM = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
    const ring = new THREE.Mesh(ringGeo, ringM);
    ring.rotation.x = -Math.PI / 2; ring.position.set(C.x, 0.16, C.z); ring.renderOrder = 2;
    scene.add(ring);
    const fg = V(o.front_ground).normalize();
    const arrow = new THREE.ArrowHelper(fg, new THREE.Vector3(C.x, 1.25, C.z), 1.45, 0xffa94d, 0.42, 0.3);
    arrow.line.material.transparent = true; arrow.cone.material.transparent = true;
    arrow.line.material.opacity = 0; arrow.cone.material.opacity = 0;
    scene.add(arrow);
    const lab = h('div', 'sd-l', '<b>' + o.id + '</b> ' + esc(SHORT[o.name] || o.name));
    const lo = new CSS2DObject(lab); lo.position.set(C.x, C.y + 0.75, C.z); scene.add(lo);
    const sc = h('div', 'sd-sc');
    const so = new CSS2DObject(sc); so.position.set(C.x, C.y + 1.55, C.z); scene.add(so);
    return { o, C, fg, ring, ringM, arrow, lab, sc };
  });

  /* soft-region shading on the ground (illustrative), observer avatar, frame axes */
  function groundShade(fn, colorHex) {
    const g = new THREE.PlaneGeometry(18, 18, 90, 90);
    g.rotateX(-Math.PI / 2);
    const p = g.attributes.position;
    const col = new Float32Array(p.count * 4);
    const c = new THREE.Color(colorHex);
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), z = p.getZ(i);
      col[i * 4] = c.r; col[i * 4 + 1] = c.g; col[i * 4 + 2] = c.b; col[i * 4 + 3] = fn(x, z);
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 4));
    const m = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(g, m);
    mesh.position.y = 0.12; mesh.renderOrder = 1;
    scene.add(mesh);
    return { mesh, m };
  }
  const fade = (x, z, c) => clamp01(1 - (Math.hypot(x - c.x, z - c.z) - 4.5) / 3.5);
  const bus = objViz[busId], suv = objViz[suvId];
  const busF = bus.fg.clone();
  const shadeBus = groundShade((x, z) => {
    const dx = x - bus.C.x, dz = z - bus.C.z, r = Math.hypot(dx, dz) || 1;
    const cosT = (dx * busF.x + dz * busF.z) / r;
    return 0.62 * Math.pow((1 - cosT) / 2, 1.4) * fade(x, z, bus.C) * clamp01(r / 0.9);
  }, COL.f1);
  const cam0 = camObjs[0];
  const camFg = new THREE.Vector3(cam0.F.x, 0, cam0.F.z).normalize();
  const shadeCam = groundShade((x, z) => {
    const depth = (x - suv.C.x) * camFg.x + (z - suv.C.z) * camFg.z;
    const lat = Math.abs((x - suv.C.x) * camFg.z - (z - suv.C.z) * camFg.x);
    return 0.62 * clamp01(0.5 + depth / 1.6) * clamp01(1 - (lat - 5) / 3) * clamp01(1 - (depth - 5.5) / 2);
  }, COL.f2);

  function dashedLine(a, b, colorHex) {
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    const m = new THREE.LineDashedMaterial({ color: colorHex, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0 });
    const L = new THREE.Line(g, m); L.computeLineDistances(); scene.add(L); return { L, m };
  }
  const perpBus = new THREE.Vector3(-busF.z, 0, busF.x);
  const lineBus = dashedLine(bus.C.clone().setY(0.2).addScaledVector(perpBus, -5), bus.C.clone().setY(0.2).addScaledVector(perpBus, 5), 0xb197fc);
  const perpCam = new THREE.Vector3(-camFg.z, 0, camFg.x);
  const lineCam = dashedLine(suv.C.clone().setY(0.2).addScaledVector(perpCam, -6), suv.C.clone().setY(0.2).addScaledVector(perpCam, 6), 0x74c0fc);

  function makeAxes(colorHex, names) {
    const g = new THREE.Group();
    const mats = [];
    const dirs = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0]];
    dirs.forEach((d, k) => {
      const big = k === 1;
      const a = new THREE.ArrowHelper(new THREE.Vector3(...d), new THREE.Vector3(0, 0, 0), big ? 2.6 : 1.8,
        big ? colorHex : 0xced4da, big ? 0.5 : 0.36, big ? 0.34 : 0.24);
      a.line.material.transparent = true; a.cone.material.transparent = true;
      mats.push(a.line.material, a.cone.material);
      g.add(a);
      const lab = h('div', 'sd-ax' + (big ? ' sd-ax-on' : ''), names[k]);
      const lo = new CSS2DObject(lab); lo.position.set(d[0] * (big ? 3.05 : 2.2), 0.05, d[2] * (big ? 3.05 : 2.2));
      g.add(lo);
      mats.push(lab);
    });
    scene.add(g);
    return { g, mats };
  }
  /* local +Z = facing direction, local +X = left (three.js is right-handed, y up) */
  const axBus = makeAxes(0x9775fa, ['front', 'behind', 'left', 'right']);
  axBus.g.position.set(bus.C.x, 1.35, bus.C.z);
  axBus.g.rotation.y = Math.atan2(busF.x, busF.z);

  function setAxes(ax, v) {
    ax.g.visible = v > 0.01;
    ax.mats.forEach((m) => { if (m.style) setOp(m, v); else m.opacity = v; });
  }

  const avatar = new THREE.Group();
  const avM = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0, emissive: 0x222222 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.24, 0.75, 20), avM); body.position.y = 0.38;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.21, 20, 14), avM); head.position.y = 0.98;
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.42, 16), avM); nose.rotation.x = Math.PI / 2; nose.position.set(0, 0.98, 0.33);
  avatar.add(body, head, nose);
  avatar.scale.setScalar(1.25);
  scene.add(avatar);
  const avLab = h('div', 'sd-l sd-l-av', '');
  const avLo = new CSS2DObject(avLab); avLo.position.set(0, 1.75, 0); avatar.add(avLo);

  /* ------------------------------------------------------------ camera poses */
  const center = new THREE.Vector3(0, 0, -0.2);
  const poseCam = (k) => {
    const c = camObjs[k];
    return { pos: c.P.clone(), tgt: c.P.clone().addScaledVector(c.F, 9), up: c.U.clone(), fov: c.c.fov_y_deg };
  };
  const poseBird = { pos: new THREE.Vector3(0.6, 17.5, 10.5), tgt: new THREE.Vector3(0.2, 0, -0.3), up: new THREE.Vector3(0, 1, 0), fov: 40 };
  const poseBird2 = { pos: new THREE.Vector3(-5.5, 15.5, 11.5), tgt: new THREE.Vector3(0.2, 0, -0.4), up: new THREE.Vector3(0, 1, 0), fov: 40 };
  const busMid = new THREE.Vector3();
  [busId, leftTank, rightTank, sedan].forEach((i) => busMid.add(objViz[i].C));
  busMid.multiplyScalar(0.25).setY(0);
  const poseBus = { pos: busMid.clone().addScaledVector(busF, -6.5).setY(16.5), tgt: busMid.clone().addScaledVector(busF, 0.4),
    up: new THREE.Vector3(0, 1, 0), fov: 44 };
  const poseCam0Wide = (() => { const P0 = poseCam(0); P0.fov += 6; return P0; })();

  const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3();
  function orbitLerp(a, b, t) {
    const da = tmpA.copy(a).sub(center), db = tmpB.copy(b).sub(center);
    const ra = Math.hypot(da.x, da.z), rb = Math.hypot(db.x, db.z);
    let aa = Math.atan2(da.z, da.x), ab = Math.atan2(db.z, db.x);
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
  function drift(P, t, deg) {
    const p = P.pos.clone().sub(P.tgt); const a = (deg * Math.PI / 180) * t;
    const x = p.x * Math.cos(a) - p.z * Math.sin(a), z = p.x * Math.sin(a) + p.z * Math.cos(a);
    return { pos: new THREE.Vector3(P.tgt.x + x, P.pos.y, P.tgt.z + z), tgt: P.tgt.clone(), up: P.up.clone(), fov: P.fov };
  }
  const birdEnd3 = drift(poseBird, 1, 12);

  /* ------------------------------------------------------------ steps */
  const v2 = (x) => fmt(x);
  const steps = [
    { t: 'Views', dur: 4200, cap: 'SATURN starts with ' + data.views.length + ' views of one scene and a question that combines two frames of reference: the bus&rsquo;s own view and camera&nbsp;0&rsquo;s view.' },
    { t: 'Detect', dur: 3800, cap: 'SAM3 detects the objects in every view, and SATURN matches the detections across views to identify ' + N + ' objects.' },
    { t: '3D scene', dur: 11000, cap: 'VGGT reconstructs one 3D scene and recovers the three cameras, so each photo aligns with the reconstruction when viewed from the corresponding camera. Orient Anything&nbsp;V2 estimates each object&rsquo;s facing direction, shown by an orange arrow.' },
    { t: 'Program', dur: 5600, cap: 'A code LLM then writes a short program that names the predicates and specifies how to combine the predicate scores. The program performs no geometry calculations directly.' },
    { t: 'Concepts', dur: 3800, cap: 'To evaluate the program, the VLM scores each object for the requested concepts, identifying three green objects, one bus, and one SUV.' },
    { t: 'Bus frame', dur: 8000, cap: 'For the first frame, stand at the bus and face toward the bus&rsquo;s front. <span class="sd-c1">Behind</span> refers to the bus&rsquo;s back side, with scores of ' + v2(S.behind_bus_bus_view[leftTank]) + ' for the left tank, ' + v2(S.behind_bus_bus_view[rightTank]) + ' for the right tank, and ' + v2(S.behind_bus_bus_view[sedan]) + ' for the sedan.' },
    { t: 'Camera 0 frame', dur: 7000, cap: 'For the second frame, look from camera&nbsp;0. <span class="sd-c2">Behind the SUV</span> means farther from camera&nbsp;0 than the SUV, with scores of ' + v2(S.behind_suv_cam0[leftTank]) + ' for the left tank, ' + v2(S.behind_suv_cam0[rightTank]) + ' for the right tank, and ' + v2(S.behind_suv_cam0[sedan]) + ' for the sedan.' },
    { t: 'Soft AND', dur: 5600, cap: 'Soft AND takes each candidate&rsquo;s minimum score, giving ' + v2(S.joint_min[leftTank]) + ' for the left tank, ' + v2(S.joint_min[rightTank]) + ' for the right tank, and ' + v2(S.joint_min[sedan]) + ' for the sedan. Object ' + ansId + ' has the highest combined score.' },
    { t: 'Answer', dur: 7000, cap: 'SATURN returns object ' + ansId + ', the left green tank, matching the ground truth (IoU ' + satIoU + '). The four VLMs below each reasoned about the bus&rsquo;s facing direction but selected the right-hand tank.' }
  ];
  const pillEls = steps.map((st, i) => {
    const b = h('button', 'sd-pill', '<span>' + (i + 1) + '</span><em>' + st.t + '</em>');
    b.type = 'button'; b.setAttribute('aria-label', 'Step ' + (i + 1) + ': ' + st.t);
    b.addEventListener('click', () => { userJump(i); });
    pills.appendChild(b);
    return b;
  });

  function poseAt(i, p) {
    if (i <= 1) return poseCam(0);
    if (i === 2) {
      if (p < 0.2) return poseCam(0);
      if (p < 0.42) return lerpPose(poseCam(0), poseCam(1), seg(p, 0.2, 0.4), true);
      if (p < 0.5) return poseCam(1);
      if (p < 0.72) return lerpPose(poseCam(1), poseCam(2), seg(p, 0.5, 0.7), true);
      if (p < 0.78) return poseCam(2);
      return lerpPose(poseCam(2), poseBird, seg(p, 0.78, 0.98), false);
    }
    if (i === 3) return drift(poseBird, p, 12);
    if (i === 4) return birdEnd3;
    if (i === 5) return lerpPose(birdEnd3, poseBus, seg(p, 0, 0.42), false);
    if (i === 6) return lerpPose(poseBus, poseCam0Wide, seg(p, 0, 0.42), false);
    if (i === 7) return lerpPose(poseCam0Wide, poseBird2, seg(p, 0, 0.4), false);
    return poseBird2;
  }
  /* which photo overlays the 3D view, and how strongly (only at the exact camera poses) */
  function photoAt(i, p) {
    if (i !== 2) return null;
    if (p < 0.2) return { k: 0, a: 1 - seg(p, 0.03, 0.16) };
    if (p >= 0.38 && p < 0.52) return { k: 1, a: seg(p, 0.4, 0.43) * (1 - seg(p, 0.46, 0.5)) };
    if (p >= 0.68 && p < 0.8) return { k: 2, a: seg(p, 0.7, 0.72) * (1 - seg(p, 0.75, 0.78)) };
    return null;
  }
  function activeCam(i, p) {
    if (i <= 1) return 0;
    if (i === 2) { if (p < 0.3) return 0; if (p < 0.6) return 1; if (p < 0.85) return 2; }
    if (i === 6) return 0;
    return -1;
  }

  let ctlActive = false;   // the user is steering the 3D camera
  const curPose = { pos: new THREE.Vector3(), tgt: new THREE.Vector3(), up: new THREE.Vector3(0, 1, 0), fov: 40 };

  function applyState(i, p) {
    const P = (k) => (k < i ? 1 : k === i ? p : 0);
    const mode = i <= 1 ? 'views' : i === 8 ? 'answer' : '3d';
    const fadeAns = i === 8 ? seg(p, 0, 0.12) : 0;
    const in3d = i === 2 ? seg(p, 0, 0.05) : 1;
    setOp(viewsEl, mode === 'views' ? 1 : i === 2 ? 1 - in3d : 0);
    setOp(l3d, mode === '3d' ? in3d : i === 8 ? 1 - fadeAns : 0);
    setOp(lans, i === 8 ? fadeAns : 0);
    setCls(viewsEl, 'sd-live', mode === 'views'); setCls(l3d, 'sd-live', mode === '3d'); setCls(lans, 'sd-live', i === 8);
    if (stageTitle.dataset.step !== String(i)) { stageTitle.innerHTML = '<span>' + (i + 1) + '</span>' + steps[i].t; stageTitle.dataset.step = String(i); }

    /* views + detections */
    setOp(viewsEl.querySelector('.sd-legend-det'), seg(P(1), 0.6, 0.9));
    viewFigs.forEach((vf, k) => {
      setOp(vf.f, seg(P(0), 0.05 + k * 0.12, 0.3 + k * 0.12));
      vf.boxes.forEach((g, j) => setOp(g, seg(P(1), 0.05 + (j / N) * 0.6 + k * 0.04, 0.2 + (j / N) * 0.6 + k * 0.04)));
    });
    setOp(legend, seg(P(0), 0.45, 0.7));

    /* 3D */
    const p2 = P(2);
    cmat.opacity = seg(p2, 0.04, 0.16);
    const ph = photoAt(i, p);
    if (ph) { if (!photo.src.endsWith(data.views[ph.k].file)) photo.src = base + data.views[ph.k].file; setOp(photo, ph.a); }
    else setOp(photo, 0);
    const ac = activeCam(i, p);
    const showCams = seg(p2, 0.12, 0.22);
    const eye = poseAt(i, p).pos;
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

    const showObj = seg(p2, 0.14, 0.24);
    const showArrow = seg(p2, 0.82, 0.95);
    const sem = P(4);
    objViz.forEach((v) => {
      const id = v.o.id;
      let col = 0xffffff, op = 0.55 * showObj, lop = showObj;
      if (sem > 0) {
        const role = id === busId ? 'bus' : id === suvId ? 'suv' : S.is_green[id] > 0.5 ? 'g' : '';
        const c = role === 'bus' ? 0x9775fa : role === 'suv' ? 0x4dabf7 : role === 'g' ? 0x51cf66 : 0xffffff;
        col = new THREE.Color(0xffffff).lerp(new THREE.Color(c), sem).getHex();
        op = (role ? 0.95 : 0.55 - 0.35 * sem) * showObj;
        lop = role ? 1 : 1 - 0.55 * sem;
        setCls(v.lab, 'sd-l-g', role === 'g' && sem > 0.5);
        setCls(v.lab, 'sd-l-b', role === 'bus' && sem > 0.5);
        setCls(v.lab, 'sd-l-s', role === 'suv' && sem > 0.5);
      } else { setCls(v.lab, 'sd-l-g', false); setCls(v.lab, 'sd-l-b', false); setCls(v.lab, 'sd-l-s', false); }
      if (i >= 7 && id === ansId) { col = 0x51cf66; op = showObj; }
      v.ringM.color.setHex(col); v.ringM.opacity = op;
      const ar = showArrow * (i >= 5 && i <= 6 ? (id === busId ? 1 : 0.25) : 1);
      v.arrow.line.material.opacity = ar; v.arrow.cone.material.opacity = ar;
      v.arrow.visible = ar > 0.01;
      setOp(v.lab, lop);
      setCls(v.lab, 'sd-l-win', i >= 7 && id === ansId && (i === 8 || seg(p, 0.75, 1) > 0.5));
    });

    /* frames */
    const f1in = i === 5 ? seg(p, 0.42, 0.62) : 0;
    const f1out = i === 5 ? 1 : 0;
    shadeBus.m.opacity = f1in * f1out;
    lineBus.m.opacity = f1in * f1out * 0.9;
    setAxes(axBus, i === 5 ? seg(p, 0.36, 0.52) : 0);
    const f2in = i === 6 ? seg(p, 0.42, 0.62) : 0;
    shadeCam.m.opacity = f2in;
    lineCam.m.opacity = f2in * 0.9;

    /* observer avatar */
    let avOp = 0, avPos = null, avYaw = 0, avCol = COL.f1, avText = '';
    if (i === 5) {
      avOp = seg(p, 0.08, 0.2);
      const start = bus.C.clone().addScaledVector(busF, -4).setY(0);
      const end = bus.C.clone().setY(1.05);
      const t = ease(seg(p, 0.1, 0.36));
      avPos = start.lerp(end, t);
      const walk = Math.atan2(busF.x, busF.z);
      avYaw = walk;
      avText = 'you, at the bus';
    } else if (i === 6) {
      avOp = 1 - seg(p, 0, 0.2);
      avPos = bus.C.clone().setY(1.05);
      avYaw = Math.atan2(busF.x, busF.z);
      avText = 'you, at the bus';
    }
    avM.opacity = avOp;
    avatar.visible = avOp > 0.01;
    if (avPos) { avatar.position.copy(avPos); avatar.rotation.y = avYaw; }
    avM.color.set(avCol);
    setText(avLab, avText);
    setOp(avLab, avOp);
    setStyle(avLab, 'borderColor', avCol);

    /* per-object score labels */
    objViz.forEach((v) => {
      const id = v.o.id; let txt = '', op = 0, cls = '';
      if (cands.includes(id)) {
        if (i === 5) { op = seg(p, 0.66, 0.8); txt = v2(S.behind_bus_bus_view[id]); cls = 'sd-sc-f1'; }
        else if (i === 6) { op = seg(p, 0.66, 0.8); txt = v2(S.behind_suv_cam0[id]); cls = 'sd-sc-f2'; }
        else if (i === 7) { op = seg(p, 0.3, 0.5); txt = 'AND ' + v2(S.joint_min[id]); cls = id === ansId ? 'sd-sc-win' : 'sd-sc-and'; }
      }
      setText(v.sc, txt); setOp(v.sc, op);
      if (v.sc.dataset.c !== cls) { v.sc.className = 'sd-sc ' + cls; v.sc.dataset.c = cls; }
    });

    /* HUD */
    let hudT = '';
    if (i === 2) hudT = ac >= 0 ? 'seen from camera ' + ac : '';
    else if (i === 5) hudT = 'frame: the bus&rsquo;s own view &middot; shaded: behind the bus';
    else if (i === 6) hudT = 'frame: camera 0&rsquo;s view &middot; shaded: behind the SUV';
    if (hud.dataset.t !== hudT) { hud.innerHTML = hudT; hud.dataset.t = hudT; }
    setCls(hud, 'sd-hud-f1', i === 5); setCls(hud, 'sd-hud-f2', i === 6);

    /* program */
    showTyped(Math.round(totalChars * (i < 3 ? 0 : i === 3 ? clamp01(p / 0.9) : 1)));
    const active = new Set();
    if (i === 4) [0, 1, 2, 3, 4].forEach((k) => active.add(k));
    if (i === 5) active.add(6);
    if (i === 6) { active.add(5); active.add(6); }
    if (i === 7) active.add(6);
    if (i === 8) active.add(7);
    lineEls.forEach((el, k) => { setCls(el, 'sd-on', active.has(k)); setCls(el, 'sd-dim', active.size > 0 && !active.has(k)); });
    setCls(pre, 'sd-mk1', i === 5); setCls(pre, 'sd-mk2', i === 6); setCls(pre, 'sd-mk3', i === 7); setCls(pre, 'sd-mk0', i === 4);

    /* bars */
    const fills = [P(4), i > 5 ? 1 : i === 5 ? seg(p, 0.66, 0.9) : 0, i > 6 ? 1 : i === 6 ? seg(p, 0.66, 0.9) : 0,
      i > 7 ? 1 : i === 7 ? seg(p, 0.15, 0.5) : 0];
    rows.forEach((r) => {
      r.cells.forEach((c, ci) => {
        const f = fills[ci];
        setStyle(c.bar, 'width', (100 * c.v * ease(f)).toFixed(1) + '%');
        setText(c.num, f > 0.02 ? v2(c.v) : '');
        setOp(c.num, f);
      });
      setCls(r.r, 'sd-win', (i === 7 && seg(p, 0.75, 1) > 0.5 || i === 8) && r.i === ansId);
    });
    const selF = i > 7 ? 1 : i === 7 ? seg(p, 0.45, 0.75) : 0;
    selRows.forEach((r) => {
      setStyle(r.bar, 'width', (100 * r.val * ease(selF)).toFixed(1) + '%');
      setText(r.v, selF > 0.02 ? v2(r.val) : '');
      setCls(r.d, 'sd-win', (i === 7 && seg(p, 0.75, 1) > 0.5 || i === 8) && cands[selRows.indexOf(r)] === ansId);
    });
    setCls(barsP, 'sd-live1', i === 5); setCls(barsP, 'sd-live2', i === 6); setCls(barsP, 'sd-live3', i === 7);

    /* answer layer */
    setOp(gGT, seg(p, 0.1, 0.22) * (i === 8 ? 1 : 0));
    setOp(gSat, seg(p, 0.2, 0.34) * (i === 8 ? 1 : 0));
    gBase.forEach((g, k) => setOp(g, (i === 8 ? 1 : 0) * seg(p, 0.4 + k * 0.08, 0.5 + k * 0.08)));
    setOp(verdict, i === 8 ? seg(p, 0.35, 0.6) * 0.8 + 0.2 : 0.2);
    setCls(verdict, 'sd-v-on', i === 8 && p > 0.4);

    /* controls */
    pillEls.forEach((b, k) => { setCls(b, 'sd-on', k === i); setCls(b, 'sd-done', k < i); });
    if (capEl.dataset.step !== String(i)) { capEl.innerHTML = steps[i].cap; capEl.dataset.step = String(i); }
    setStyle(progBar, 'width', (100 * p).toFixed(2) + '%');

    /* camera */
    if (threeOK && !ctlActive) {
      const ps = poseAt(i, p);
      camera.position.copy(ps.pos); camera.up.copy(ps.up); camera.lookAt(ps.tgt);
      if (Math.abs(camera.fov - ps.fov) > 1e-3) { camera.fov = ps.fov; camera.updateProjectionMatrix(); }
      curPose.pos.copy(ps.pos); curPose.tgt.copy(ps.tgt);
    }
  }

  /* ------------------------------------------------------------ sizing */
  function resize() {
    if (!threeOK) return;
    const w = l3d.clientWidth || 600, hh = l3d.clientHeight || 450;
    gl.setSize(w, hh, false);
    labels.setSize(w, hh);
    camera.aspect = w / hh; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(l3d);
  resize();

  /* ------------------------------------------------------------ player */
  let step = 0, p = 0, playing = false, last = 0, visible = false, started = false, raf = 0, dirty = true;
  function setPlayBtn() {
    bPlay.innerHTML = playing ? '<span aria-hidden="true">&#10074;&#10074;</span>' : '<span aria-hidden="true">&#9654;</span>';
    bPlay.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    root.classList.toggle('sd-playing', playing);
  }
  function stopControls() {
    if (!controls) return;
    controls.enabled = false; ctlActive = false; dirty = true;
  }
  function play() {
    if (step === steps.length - 1 && p >= 1) { step = 0; p = 0; }
    stopControls();
    playing = true; started = true; last = performance.now(); setPlayBtn(); kick();
  }
  function pause() {
    playing = false; setPlayBtn();
    if (controls) { controls.target.copy(curPose.tgt); controls.enabled = true; controls.update(); }
    kick();
  }
  function go(i, prog) {
    step = Math.max(0, Math.min(steps.length - 1, i)); p = prog; last = performance.now(); dirty = true; kick();
  }
  function userJump(i) {
    stopControls();
    go(i, playing && !reduce ? 0 : 1);
    if (!playing && controls) { applyState(step, p); controls.target.copy(curPose.tgt); controls.enabled = true; controls.update(); }
  }
  bPlay.addEventListener('click', () => (playing ? pause() : play()));
  bNext.addEventListener('click', () => userJump(step + 1));
  bPrev.addEventListener('click', () => userJump(p > 0.25 && playing ? step : step - 1));
  bRe.addEventListener('click', () => { stopControls(); go(0, 0); play(); });
  if (threeOK) {
    controls.addEventListener('start', () => { ctlActive = true; dirty = true; kick(); });
    controls.addEventListener('change', () => { dirty = true; kick(); });
  }

  function frame(now) {
    raf = 0;
    if (playing) {
      const dt = Math.min(100, now - last);
      last = now;
      p += reduce ? 0 : dt / steps[step].dur;
      if (reduce) { p = 1; holdT += dt; if (holdT >= steps[step].dur) { holdT = 0; advance(); } }
      else if (p >= 1) advance();
      dirty = true;
    }
    if (dirty) {
      applyState(step, Math.min(1, p));
      if (threeOK) {
        if (ctlActive && controls) controls.update();
        gl.render(scene, camera);
        labels.render(scene, camera);
      }
      dirty = playing || (controls && controls.enabled && ctlActive);
    }
    if (visible && (playing || dirty)) raf = requestAnimationFrame(frame);
  }
  let holdT = 0;
  function advance() {
    if (step < steps.length - 1) { step += 1; p = reduce ? 1 : 0; }
    else { p = 1; playing = false; setPlayBtn(); if (controls) { controls.target.copy(curPose.tgt); controls.enabled = true; } }
  }
  function kick() { if (!raf && visible) raf = requestAnimationFrame(frame); }

  const io = new IntersectionObserver((ents) => {
    ents.forEach((e) => {
      visible = e.isIntersecting;
      if (visible) {
        if (!started && !reduce) play();
        else { dirty = true; kick(); }
      } else if (raf) { cancelAnimationFrame(raf); raf = 0; }
    });
  }, { threshold: 0.3 });
  io.observe(root);

  if (reduce) {
    go(0, 1);
    root.classList.add('sd-reduced');
  }
  setPlayBtn();
  applyState(0, reduce ? 1 : 0);
  if (threeOK) { gl.render(scene, camera); labels.render(scene, camera); }

  /* hooks for automated checks (screenshots at given steps) */
  root.__sd = { go: (i, prog) => { playing = false; setPlayBtn(); stopControls(); go(i, prog); }, steps: steps.length };
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
