/*
 * NePTune project page: two animated walkthroughs built from examples in the paper.
 *   #npt-fwd : Figure 2 (CLEVR). The Python program turns into a tree and is evaluated:
 *              declarative soft logic over all boxes, then imperative control flow.
 *   #npt-ref : Figure 3, Expression 1 (Ref-GTA). A purely declarative program, run forward,
 *              then a loss on its answer is propagated end to end into the VLM.
 * Every value comes from example.json; the notes on the page say how each one was produced.
 */
(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var clamp01 = function (x) { return Math.max(0, Math.min(1, x)); };
  var seg = function (p, a, b) { return clamp01((p - a) / (b - a)); };
  var ease = function (t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  var lerp = function (a, b, t) { return a + (b - a) * t; };
  var f2 = function (x) { return x.toFixed(2); };
  var esc = function (t) { return String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); };

  function h(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  function s(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  /* tiny cache so per-frame updates only touch the DOM when something changes */
  var cache = new WeakMap();
  function memo(el, key, val) {
    var c = cache.get(el);
    if (!c) { c = {}; cache.set(el, c); }
    if (c[key] === val) return false;
    c[key] = val; return true;
  }
  function setAttr(el, k, v) { if (memo(el, 'a:' + k, v)) el.setAttribute(k, v); }
  function setStyle(el, k, v) { if (memo(el, 's:' + k, v)) el.style[k] = v; }
  function setOp(el, v) { setStyle(el, 'opacity', String(Math.round(v * 1000) / 1000)); }
  function setCls(el, c, on) { if (el.classList.contains(c) !== on) el.classList.toggle(c, on); }
  function setText(el, t) { if (el.textContent !== t) el.textContent = t; }
  function setHTML(el, t) { if (memo(el, 'html', t)) el.innerHTML = t; }

  /* ------------------------------------------------------------------ code tokens */
  function tokens(line) {
    var out = [], re = /("[^"]*"|'[^']*')|\b(def|for|in|if|return)\b|\b(score|query|set|range|len|iota)(?=\()|(&)/g, last = 0, m;
    while ((m = re.exec(line))) {
      if (m.index > last) out.push([line.slice(last, m.index), '']);
      out.push([m[0], m[1] ? 'nd-s' : m[2] ? 'nd-kw' : m[3] ? 'nd-fn' : 'nd-op']);
      last = re.lastIndex;
    }
    if (last < line.length) out.push([line.slice(last), '']);
    return out;
  }
  function partial(toks, n, caret) {
    var html = '', used = 0;
    for (var k = 0; k < toks.length; k++) {
      var t = toks[k][0], c = toks[k][1];
      var vis = Math.max(0, Math.min(t.length, n - used));
      if (vis > 0) html += c ? '<span class="' + c + '">' + esc(t.slice(0, vis)) + '</span>' : esc(t.slice(0, vis));
      if (caret && used + vis === n && vis < t.length) { html += '<span class="nd-caret"></span>'; caret = false; }
      if (vis < t.length) html += '<span class="nd-ghost">' + esc(t.slice(vis)) + '</span>';
      used += t.length;
    }
    if (caret && used === n) html += '<span class="nd-caret"></span>';
    return html;
  }

  /* ------------------------------------------------------------------ the two trees */
  var CFG = {
    fwd: {
      role: ['', 'con', 'con', 'sl', 'imp', 'imp', 'imp', 'imp', 'imp', 'imp'],
      lineNode: { 1: 'sphere', 2: 'small', 3: 'and', 4: 'set', 5: 'for', 6: 'if', 7: 'query', 8: 'set', 9: 'len' },
      chips: [1, 2, 3, 4, 5, 6, 7, 8, 9],
      nodes: {
        len:    { k: 'imp', lines: [9], lab: 'len(colors)' },
        set:    { k: 'imp', lines: [4, 8], lab: 'colors', sub: ' ' },
        'for':  { k: 'imp', lines: [5], lab: 'for i in range(n)' },
        'if':   { k: 'imp', lines: [6], lab: 'if small_sphere[i]', sub: ' ' },
        and:    { k: 'sl', lines: [3], lab: 'sphere("x") & small("x")', labN: 'sphere & small', sub: 'soft AND: element-wise min', vec: true },
        query:  { k: 'imp', lines: [7], lab: 'query("what color…", i)', labN: 'query("…color…", i)', sub: 'VLM answer: a string' },
        sphere: { k: 'con', lines: [1], lab: 'sphere("x")', sub: 'score("…a sphere?", 1)', vec: true },
        small:  { k: 'con', lines: [2], lab: 'small("x")', sub: 'score("…small?", 1)', vec: true },
        vlm:    { k: 'vlm', lines: [], lab: 'VLM', sub: ' ' }
      },
      edges: [
        ['sphere', 'and', 'e-con'], ['small', 'and', 'e-con'], ['and', 'if', 'e-decl'], ['query', 'if', 'e-imp'],
        ['if', 'for', 'e-imp'], ['for', 'set', 'e-imp'], ['set', 'len', 'e-imp'],
        ['vlm', 'sphere', 'e-vlm'], ['vlm', 'small', 'e-vlm'], ['vlm', 'query', 'e-vlm']
      ],
      region: { top: 'and', bottom: ['sphere', 'small'], label: 'DECLARATIVE · first-order logic over all boxes', labelN: 'DECLARATIVE · first-order logic' },
      impLabel: true,
      layouts: {
        wide: { W: 640, H: 532, fs: 13, fsub: 10.5, nodes: {
          len: [320, 30, 160, 36], set: [320, 98, 330, 44], 'for': [320, 164, 196, 34], 'if': [320, 228, 270, 44],
          and: [168, 318, 262, 80], query: [492, 312, 232, 50], sphere: [92, 428, 176, 80], small: [282, 428, 176, 80],
          vlm: [320, 508, 616, 34] } },
        narrow: { W: 420, H: 612, fs: 14, fsub: 11.5, nodes: {
          len: [210, 30, 170, 38], set: [210, 98, 406, 46], 'for': [210, 164, 210, 36], 'if': [256, 232, 296, 48],
          and: [112, 332, 212, 86], query: [322, 324, 188, 54], sphere: [106, 456, 202, 86], small: [316, 456, 202, 86],
          vlm: [210, 584, 408, 36] } }
      }
    },
    ref: {
      role: ['con', 'con', 'sl', 'sl'],
      lineNode: { 0: 'man', 1: 'black', 2: 'and', 3: 'iota' },
      chips: [0, 1, 2, 3],
      nodes: {
        iota:  { k: 'sl', lines: [3], lab: "man_wearing_black.iota('x1')", sub: 'best match: softmax over the boxes', vec: true },
        and:   { k: 'sl', lines: [2], lab: "is_man('x1') & is_wearing_black('x1')", labN: 'is_man & is_wearing_black', sub: 'soft AND: element-wise min', vec: true },
        man:   { k: 'con', lines: [0], lab: "is_man('x1')", sub: 'score("…a man?", 1)', vec: true },
        black: { k: 'con', lines: [1], lab: "is_wearing_black('x1')", sub: 'score("…wearing black?", 1)', vec: true },
        vlm:   { k: 'vlm', lines: [], lab: 'VLM', sub: ' ' },
        loss:  { k: 'loss', lines: [], lab: 'BCE loss', sub: ' ' }
      },
      edges: [['man', 'and', 'e-con'], ['black', 'and', 'e-con'], ['and', 'iota', 'e-decl'],
              ['vlm', 'man', 'e-vlm'], ['vlm', 'black', 'e-vlm'], ['iota', 'loss', 'e-grad']],
      region: { top: 'iota', bottom: ['man', 'black'], mid: 'and', label: 'FIRST-ORDER LOGIC · differentiable end to end', labelN: 'FIRST-ORDER LOGIC · differentiable' },
      impLabel: false,
      layouts: {
        wide: { W: 640, H: 470, fs: 13, fsub: 10.5, nodes: {
          iota: [280, 74, 300, 88], loss: [540, 74, 150, 52], and: [300, 208, 330, 88],
          man: [165, 342, 250, 88], black: [435, 342, 250, 88], vlm: [320, 444, 616, 34] } },
        narrow: { W: 420, H: 508, fs: 14, fsub: 11.5, nodes: {
          iota: [158, 76, 300, 92], loss: [366, 76, 100, 54], and: [210, 216, 400, 92],
          man: [106, 354, 202, 92], black: [316, 354, 202, 92], vlm: [210, 476, 408, 36] } }
      }
    }
  };

  /* ================================================================== widget */
  function Widget(root, D, mode) {
    var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var C = CFG[mode];
    var R = mode === 'ref' ? D.ref : D.fwd;
    var N = R.boxes.length;
    var prog = R.program;
    var base = root.dataset.src;
    var NODES = C.nodes, EDGES = C.edges, LAYOUTS = C.layouts, ROLE = C.role, LINE_NODE = C.lineNode;
    root.innerHTML = '';
    root.classList.add('nd-ready');
    if (mode === 'ref') root.classList.add('npt-tune');

    /* ---------------- question */
    var q = h('div', 'nd-q');
    q.appendChild(h('span', 'nd-q-tag', mode === 'ref' ? 'Expression' : 'Question'));
    q.appendChild(h('span', 'nd-q-text', esc(R.question)));
    q.appendChild(h('span', 'nd-q-meta', mode === 'ref'
      ? 'Expression 1 of Figure 3 in the paper &middot; Ref-GTA (video game images) &middot; ' + N + ' detected boxes'
      : 'Figure 2 of the paper &middot; CLEVR scene &middot; ' + N + ' detected boxes'));
    root.appendChild(q);

    var main = h('div', 'nd-main');
    var left = h('div', 'nd-left');
    var right = h('div', 'nd-right');
    main.appendChild(left); main.appendChild(right);
    root.appendChild(main);

    /* ---------------- code panel */
    var codeP = h('div', 'nd-panel nd-codep');
    codeP.appendChild(h('div', 'nd-panel-h', 'Program <span>' + (mode === 'ref' ? 'from Figure 3 of the paper' : 'from Figure 2, written by the LLM') + '</span>'));
    var pre = h('pre', 'nd-pre');
    var lineEls = prog.map(function (line, i) {
      var ind = line.length - line.replace(/^ +/, '').length;
      var el = h('div', 'nd-ln' + (ROLE[i] ? ' nd-ln-' + ROLE[i] : ''));
      el.style.setProperty('--ind', String(ind));
      var txt = h('span', 'nd-lt');
      el.appendChild(txt);
      pre.appendChild(el);
      return { el: el, txt: txt, toks: tokens(line.slice(ind)), n: line.length - ind };
    });
    codeP.appendChild(pre);
    left.appendChild(codeP);
    var totalChars = lineEls.reduce(function (a, l) { return a + l.n; }, 0);

    /* ---------------- image panel */
    var imgP = h('div', 'nd-panel nd-imgp');
    imgP.appendChild(h('div', 'nd-panel-h', 'Image <span>boxes from Grounding DINO; red box = the object the VLM is asked about</span>'));
    var wrap = h('div', 'nd-imgwrap');
    var img = h('img');
    img.src = base + R.image;
    img.alt = mode === 'ref' ? 'Ref-GTA image from Figure 3 of the paper with the detected boxes' : 'CLEVR scene from Figure 2 of the paper with the detected boxes';
    img.width = R.img_w; img.height = R.img_h;
    wrap.appendChild(img);
    var ov = s('svg', { 'class': 'nd-ov', viewBox: '0 0 ' + R.img_w + ' ' + R.img_h, preserveAspectRatio: 'none' });
    var TAG = Math.round(R.img_w / 32);
    var boxEls = R.boxes.map(function (b, i) {
      var g = s('g', {}, ov);
      var r = s('rect', { x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1], rx: 4, 'class': 'nd-box' }, g);
      var tb = s('rect', { x: b[0], y: b[1] - 1, width: TAG, height: TAG, rx: 5, 'class': 'nd-tagbg' }, g);
      var tt = s('text', { x: b[0] + TAG / 2, y: b[1] + TAG * 0.77, 'class': 'nd-tag' }, g);
      tt.style.fontSize = Math.round(TAG * 0.87) + 'px';
      tt.textContent = String(i);
      return { g: g, r: r, tb: tb };
    });
    var ansEls = {}, gtEl = null, selEl = null;
    if (mode === 'fwd') {
      Object.keys(R.answers || {}).forEach(function (k) {
        var b = R.boxes[+k];
        var g = s('g', {}, ov);
        var txt = '“' + R.answers[k] + '”';
        var w = 18 * txt.length + 16;
        var x = Math.min(R.img_w - w - 4, Math.max(4, (b[0] + b[2]) / 2 - w / 2));
        var y = b[3] + 6 + 40 > R.img_h ? b[1] - 46 : b[3] + 6;
        s('rect', { x: x, y: y, width: w, height: 40, rx: 8, 'class': 'nd-ansbg' }, g);
        var t = s('text', { x: x + 8, y: y + 30, 'class': 'nd-ans' }, g);
        t.textContent = txt;
        ansEls[k] = g;
      });
    } else {
      var gb = R.gt_box;
      gtEl = s('g', { opacity: 0 }, ov);
      s('rect', { x: gb[0], y: gb[1], width: gb[2] - gb[0], height: gb[3] - gb[1], rx: 4, 'class': 'nd-gt' }, gtEl);
      var gtl = s('text', { x: gb[0], y: gb[3] + 24, 'class': 'nd-gtlab' }, gtEl);
      gtl.textContent = 'ground truth';
      selEl = s('g', { opacity: 0 }, ov);
      selEl.rect = s('rect', { x: 0, y: 0, width: 10, height: 10, rx: 4, 'class': 'nd-sel' }, selEl);
      selEl.bg = s('rect', { x: 0, y: 0, width: 150, height: 30, rx: 6, 'class': 'nd-selbg' }, selEl);
      selEl.t = s('text', { x: 0, y: 0, 'class': 'nd-sellab' }, selEl);
    }
    wrap.appendChild(ov);
    var imgCap = h('div', 'nd-imgcap');
    wrap.appendChild(imgCap);
    imgP.appendChild(wrap);
    left.insertBefore(imgP, codeP);

    /* ---------------- tuning panel (ref) */
    var tuneP = null, tuneRows = [], spark = null, sparkDot = null, lossTxt = null;
    if (mode === 'ref') {
      tuneP = h('div', 'nd-panel nd-tunep');
      tuneP.appendChild(h('div', 'nd-panel-h', 'Tuning <span>soft truth values per box</span>'));
      var grid = h('div', 'nd-tune-grid');
      grid.appendChild(h('div', 'nd-tr nd-th', '<span>box</span><span class="c-con">is_man</span><span class="c-con">wearing black</span><span class="c-sl">AND</span><span class="c-sl">iota</span>'));
      for (var i = 0; i < N; i++) {
        var b = R.boxes[i];
        var row = h('div', 'nd-tr');
        var bw = b[2] - b[0], bh = b[3] - b[1], th = 22 / Math.max(bw, bh);
        var thumb = '<span class="nd-thumb" style="width:' + Math.round(bw * th) + 'px;height:' + Math.round(bh * th) + 'px;background:url(' + base + R.image + ') ' +
          (-b[0] * th).toFixed(1) + 'px ' + (-b[1] * th).toFixed(1) + 'px / ' + (R.img_w * th).toFixed(1) + 'px ' + (R.img_h * th).toFixed(1) + 'px no-repeat"></span>';
        row.innerHTML = '<span class="nd-tname"><b>' + i + '</b>' + thumb + (i === R.gt_index ? '<em class="nd-gtmark">target</em>' : '') + '</span>' +
          '<span class="nd-cell c-con"><i></i><span></span></span><span class="nd-cell c-con"><i></i><span></span></span>' +
          '<span class="nd-cell c-sl"><i></i><span></span></span><span class="nd-cell c-sl"><i></i><span></span></span>';
        var cells = row.querySelectorAll('.nd-cell');
        tuneRows.push({ row: row, c: [cells[0], cells[1], cells[2], cells[3]] });
        grid.appendChild(row);
      }
      tuneP.appendChild(grid);
      var lr = h('div', 'nd-loss-row');
      lossTxt = h('span', '', 'loss');
      var sv = s('svg', { viewBox: '0 0 200 38' });
      spark = s('path', { 'class': 'nd-spark', d: '' }, sv);
      sparkDot = s('circle', { r: 3, 'class': 'nd-spark-dot', cx: -10, cy: -10 }, sv);
      lr.appendChild(lossTxt); lr.appendChild(sv);
      tuneP.appendChild(lr);
      left.appendChild(tuneP);
    }

    /* ---------------- tree panel */
    var treeP = h('div', 'nd-panel nd-treep');
    treeP.appendChild(h('div', 'nd-panel-h', 'Program tree <span>' + (mode === 'ref' ? 'forward pass, then the loss flows back' : 'built from the code, then evaluated') + '</span>'));
    treeP.appendChild(h('div', 'nd-legend', mode === 'ref'
      ? '<span><b>Declarative:</b><i class="lg-con"></i>concept score</span><span><i class="lg-sl"></i>soft logic</span><span><i class="lg-grad"></i>gradient</span>'
      : '<span><b>Declarative:</b><i class="lg-con"></i>concept score</span><span><i class="lg-sl"></i>soft logic</span><span><b>Imperative:</b><i class="lg-imp"></i>Python control flow</span>'));
    var svg = s('svg', { 'class': 'nd-tree', role: 'img' });
    svg.setAttribute('aria-label', mode === 'ref'
      ? 'Tree of the program for "Man wearing a black suit": iota at the root, the soft AND below it, and the is_man and is_wearing_black concept scores at the leaves.'
      : 'Tree of the program for "How many colors of small spheres are there?": len(colors) at the root, the for loop and if branch below it, and the soft AND of the sphere and small concept scores at the leaves.');
    treeP.appendChild(svg);
    right.appendChild(treeP);

    /* ---------------- controls */
    var ctrl = h('div', 'nd-ctrl');
    var pills = h('div', 'nd-pills');
    var capEl = h('p', 'nd-cap'); capEl.setAttribute('aria-live', 'polite');
    var btns = h('div', 'nd-btns');
    var bPrev = h('button', 'nd-b', '<span aria-hidden="true">&#9664;&#9664;</span>'); bPrev.setAttribute('aria-label', 'Previous step');
    var bPlay = h('button', 'nd-b nd-b-main', ''); bPlay.setAttribute('aria-label', 'Play');
    var bNext = h('button', 'nd-b', '<span aria-hidden="true">&#9654;&#9654;</span>'); bNext.setAttribute('aria-label', 'Next step');
    var bRe = h('button', 'nd-b', '<span aria-hidden="true">&#8635;</span>'); bRe.setAttribute('aria-label', 'Replay from the start');
    [bPrev, bPlay, bNext, bRe].forEach(function (b) { b.type = 'button'; btns.appendChild(b); });
    var progEl = h('div', 'nd-prog', '<i></i>');
    var progBar = progEl.querySelector('i');
    ctrl.appendChild(pills);
    var crow = h('div', 'nd-ctrl-row'); crow.appendChild(capEl); crow.appendChild(btns);
    ctrl.appendChild(crow); ctrl.appendChild(progEl);
    root.appendChild(ctrl);

    var fly = h('div', 'nd-fly');
    root.appendChild(fly);
    var chips = C.chips.map(function (li) {
      var id = LINE_NODE[li];
      var c = h('div', 'nd-chipfly k-' + NODES[id].k);
      c.style.opacity = '0';
      fly.appendChild(c);
      return { li: li, id: id, el: c };
    });

    /* ================================================================ tree build */
    var T = null;
    var layoutName = '';
    function nodeLab(id, L) { return (L === 'narrow' && NODES[id].labN) || NODES[id].lab; }

    function regionPath(tl, tr, y0, bl, br, ym, y1) {
      var c = 9, d = 'M' + tl + ',' + (y0 + c) + ' Q' + tl + ',' + y0 + ' ' + (tl + c) + ',' + y0 + ' H' + (tr - c) + ' Q' + tr + ',' + y0 + ' ' + tr + ',' + (y0 + c);
      if (br - tr > 2 * c) d += ' V' + (ym - c) + ' Q' + tr + ',' + ym + ' ' + (tr + c) + ',' + ym + ' H' + (br - c) + ' Q' + br + ',' + ym + ' ' + br + ',' + (ym + c);
      else d += ' V' + ym + ' H' + br;
      d += ' V' + (y1 - c) + ' Q' + br + ',' + y1 + ' ' + (br - c) + ',' + y1 + ' H' + (bl + c) + ' Q' + bl + ',' + y1 + ' ' + bl + ',' + (y1 - c);
      if (tl - bl > 2 * c) d += ' V' + (ym + c) + ' Q' + bl + ',' + ym + ' ' + (bl + c) + ',' + ym + ' H' + (tl - c) + ' Q' + tl + ',' + ym + ' ' + tl + ',' + (ym - c);
      else d += ' V' + ym + ' H' + tl;
      return d + ' Z';
    }

    function buildTree(L) {
      layoutName = L;
      var LY = LAYOUTS[L];
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute('viewBox', '0 0 ' + LY.W + ' ' + LY.H);
      T = { L: L, LY: LY, nodes: {}, edges: {}, dots: [], dotI: 0 };
      var gR = s('g', {}, svg), gE = s('g', {}, svg), gN = s('g', {}, svg), gTop = s('g', {}, svg);
      var P = LY.nodes;
      function box(id) { var a = P[id]; return { x: a[0], y: a[1], w: a[2], h: a[3], l: a[0] - a[2] / 2, t: a[1] - a[3] / 2, r: a[0] + a[2] / 2, b: a[1] + a[3] / 2 }; }
      /* region around the first-order-logic part */
      var RG = C.region, TB = box(RG.top), B1 = box(RG.bottom[0]), B2 = box(RG.bottom[1]);
      var ymid = RG.mid ? box(RG.mid).t - 10 : B1.t - 10;
      var rd = regionPath(TB.l - 6, TB.r + 8, TB.t - 10, Math.min(B1.l, B2.l) - 5, Math.max(B1.r, B2.r) + 5, ymid, B1.b + 19);
      if (RG.mid) {
        var MB = box(RG.mid);
        rd = regionPath(Math.min(TB.l, MB.l) - 6, Math.max(TB.r, MB.r) + 8, TB.t - 10, Math.min(B1.l, B2.l) - 5, Math.max(B1.r, B2.r) + 5, B1.t - 10, B1.b + 19);
      }
      T.region = s('path', { d: rd, 'class': 'nd-region' }, gR);
      T.regionT = s('text', { 'class': 'nd-region-t t-sans', x: Math.min(B1.l, B2.l) + 4, y: B1.b + 13 }, gR);
      T.regionT.textContent = L === 'narrow' ? RG.labelN : RG.label;
      if (C.impLabel) {
        T.impT = s('text', { 'class': 'nd-imp-t t-sans', x: 6, y: 16 }, gR);
        T.impT.innerHTML = '<tspan x="6" dy="0">IMPERATIVE</tspan><tspan x="6" dy="12">Python control flow</tspan>';
      }
      EDGES.forEach(function (e) {
        var a = box(e[0]), b = box(e[1]), d;
        if (e[0] === 'vlm') d = 'M' + b.x + ',' + a.t + ' L' + b.x + ',' + b.b;
        else if (e[1] === 'loss') d = 'M' + a.r + ',' + a.y + ' L' + b.l + ',' + b.y;
        else { var my = (a.t + b.b) / 2; d = 'M' + a.x + ',' + a.t + ' C' + a.x + ',' + my + ' ' + b.x + ',' + my + ' ' + b.x + ',' + b.b; }
        var path = s('path', { d: d, 'class': 'nd-edge ' + e[2] }, gE);
        var len = path.getTotalLength();
        path.style.strokeDasharray = len + ' ' + len;
        var gp = s('path', { d: d, 'class': 'nd-gedge' }, gE);
        gp.style.strokeDasharray = len + ' ' + len; gp.style.strokeDashoffset = String(len); gp.style.opacity = '0';
        T.edges[e[0] + '>' + e[1]] = { path: path, gp: gp, len: len };
      });
      Object.keys(NODES).forEach(function (id) {
        var nd = NODES[id], bx = box(id);
        var g = s('g', { 'class': 'nd-node k-' + nd.k, transform: 'translate(' + bx.l + ',' + bx.t + ')' }, gN);
        var rx = nd.k === 'imp' ? Math.min(bx.h / 2, 18) : 9;
        var rect = s('rect', { 'class': 'nb', x: 0, y: 0, width: bx.w, height: bx.h, rx: rx }, g);
        var hasSub = !!nd.sub;
        var labY = nd.vec ? LY.fs + 6 : hasSub ? bx.h / 2 - 2 : bx.h / 2 + LY.fs * 0.36;
        var lab = s('text', { 'class': 'nl', x: bx.w / 2, y: labY, 'text-anchor': 'middle' }, g);
        lab.style.fontSize = LY.fs + 'px';
        lab.textContent = id === 'vlm' ? 'Vision-language model (VLM)' : nodeLab(id, L);
        var sub = null;
        if (id === 'vlm') { lab.setAttribute('class', 'nl t-sans'); lab.setAttribute('y', bx.h / 2 + LY.fs * 0.36); }
        else if (hasSub) {
          sub = s('text', { 'class': 'ns t-sans', x: bx.w / 2, y: nd.vec ? LY.fs + 6 + LY.fsub + 3 : bx.h / 2 + LY.fsub + 1, 'text-anchor': 'middle' }, g);
          sub.style.fontSize = LY.fsub + 'px';
          sub.textContent = nd.sub.trim();
        }
        var ref = { g: g, rect: rect, lab: lab, sub: sub, box: bx, bars: [] };
        if (nd.vec) {
          var x0 = 10, x1 = bx.w - 10, gap = N > 4 ? 2 : 10, bw = (x1 - x0 - gap * (N - 1)) / N;
          var yb = bx.h - 14, mh = bx.h - (LY.fs + 6 + LY.fsub + 3) - 22;
          s('rect', { 'class': 'nd-bar-bg', x: x0 - 3, y: yb - mh - 3, width: x1 - x0 + 6, height: mh + 4, rx: 3 }, g);
          ref.cur = s('rect', { 'class': 'nd-bcur', opacity: 0, x: 0, y: yb - mh - 2, width: bw + 2, height: mh + 3, rx: 2 }, g);
          for (var i = 0; i < N; i++) {
            var bxx = x0 + i * (bw + gap);
            var tgt = id === 'iota' && mode === 'ref' ? s('rect', { 'class': 'nd-bar b-tgt', x: bxx, y: yb, width: bw, height: 0, opacity: 0 }, g) : null;
            var bar = s('rect', { 'class': 'nd-bar b-' + (nd.k === 'con' ? 'con' : 'sl'), x: bxx, y: yb, width: bw, height: 0 }, g);
            s('title', {}, bar).textContent = 'box ' + i;
            s('text', { 'class': 'nd-bidx t-sans', x: bxx + bw / 2, y: yb + 10 }, g).textContent = N > 4 ? String(i) : 'box ' + i;
            var val = N <= 4 ? s('text', { 'class': 'nd-bval t-sans', x: bxx + bw / 2, y: yb - 4, opacity: 0 }, g) : null;
            var arr = mode === 'ref' ? s('path', { d: '', 'class': 'nd-arrow', opacity: 0 }, g) : null;
            ref.bars.push({ el: bar, x: bxx, w: bw, tgt: tgt, arr: arr, val: val });
          }
          ref.yb = yb; ref.mh = mh;
        }
        var bg = s('g', { 'class': 'nd-badge', opacity: 0 }, gTop);
        var br = s('rect', { x: 0, y: 0, width: 40, height: 18, rx: 9 }, bg);
        var bt = s('text', { 'class': 't-sans', x: 20, y: 13 }, bg);
        ref.badge = { g: bg, r: br, t: bt };
        T.nodes[id] = ref;
        g.addEventListener('mouseenter', function () { hov(id, true); });
        g.addEventListener('mouseleave', function () { hov(id, false); });
      });
      if (mode === 'fwd') {
        var ln = box('len');
        var ag = s('g', { 'class': 'nd-answer', opacity: 0, transform: 'translate(' + (ln.r + 12) + ',' + (ln.y - 15) + ')' }, gTop);
        s('rect', { x: 0, y: 0, width: 92, height: 30, rx: 15 }, ag);
        T.ansT = s('text', { x: 46, y: 20 }, ag);
        T.ans = ag;
        var cg = s('g', { 'class': 'nd-chip', opacity: 0 }, gTop);
        T.chipR = s('rect', { x: -30, y: -11, width: 60, height: 22, rx: 11 }, cg);
        T.chipT = s('text', { x: 0, y: 4 }, cg);
        T.chip = cg;
      }
      T.gDots = s('g', {}, gTop);
    }

    function hov(id, on) {
      if (!T) return;
      var n = T.nodes[id]; if (n) setCls(n.g, 'nd-hov', on);
      (NODES[id].lines || []).forEach(function (li) { setCls(lineEls[li].el, 'nd-hov', on); });
    }
    lineEls.forEach(function (l, li) {
      l.el.addEventListener('mouseenter', function () { if (LINE_NODE[li]) hov(LINE_NODE[li], true); });
      l.el.addEventListener('mouseleave', function () { if (LINE_NODE[li]) hov(LINE_NODE[li], false); });
    });

    /* ---------------- per-frame helpers */
    function badge(id, text, cls, op) {
      var n = T.nodes[id], b = n.badge;
      if (!op || !text) { setOp(b.g, 0); return; }
      setText(b.t, text);
      var w = Math.max(30, text.length * 6.4 + 14);
      setAttr(b.r, 'width', w.toFixed(0)); setAttr(b.t, 'x', (w / 2).toFixed(1));
      var bx = n.box;
      var x = Math.min(T.LY.W - w - 2, bx.r - w + 8), y = bx.t - 12;
      setAttr(b.g, 'transform', 'translate(' + x.toFixed(1) + ',' + y.toFixed(1) + ')');
      setAttr(b.g, 'class', 'nd-badge' + (cls ? ' bd-' + cls : ''));
      setOp(b.g, op);
    }
    function setBars(id, vals, fill) {
      var n = T.nodes[id];
      for (var i = 0; i < N; i++) {
        var f = typeof fill === 'number' ? fill : fill[i];
        var v = (vals ? vals[i] : 0) * f;
        var hh = Math.max(0, v * n.mh);
        setAttr(n.bars[i].el, 'y', (n.yb - hh).toFixed(2));
        setAttr(n.bars[i].el, 'height', hh.toFixed(2));
        setText(n.bars[i].el.firstChild, 'box ' + i + ': ' + f2(vals ? vals[i] : 0));
        var vl = n.bars[i].val;
        if (vl) {
          setText(vl, f2(vals[i]));
          setAttr(vl, 'y', (n.yb - Math.max(hh, 0) - 4 < n.yb - n.mh + 9 ? n.yb - hh + 12 : n.yb - hh - 4).toFixed(1));
          setAttr(vl, 'class', 'nd-bval t-sans' + (n.yb - hh - 4 < n.yb - n.mh + 9 ? ' in' : ''));
          setOp(vl, f > 0.98 ? 1 : 0);
        }
      }
    }
    function cur(id, i) {
      var n = T.nodes[id];
      if (i === null || i < 0) { setOp(n.cur, 0); return; }
      setOp(n.cur, 1);
      setAttr(n.cur, 'x', (n.bars[i].x - 1).toFixed(2));
    }
    function nodeOp(id, v) { setOp(T.nodes[id].g, v); }
    function edgeDraw(key, t) {
      var e = T.edges[key]; if (!e) return;
      setStyle(e.path, 'strokeDashoffset', String((e.len * (1 - t)).toFixed(1)));
    }
    function gEdge(key, t) {    // gradient overlay, drawn from the parent end back to the child
      var e = T.edges[key]; if (!e) return;
      setStyle(e.gp, 'opacity', t > 0 ? '1' : '0');
      setStyle(e.gp, 'strokeDashoffset', String((-e.len * (1 - t)).toFixed(1)));
    }
    function dotsReset() { T.dotI = 0; }
    function dot(x, y, color, r, op) {
      var d = T.dots[T.dotI];
      if (!d) { d = s('circle', { 'class': 'nd-dot', r: 4 }, T.gDots); T.dots.push(d); }
      T.dotI++;
      setAttr(d, 'cx', x.toFixed(1)); setAttr(d, 'cy', y.toFixed(1)); setAttr(d, 'fill', color); setAttr(d, 'r', (r || 4).toFixed(2));
      setStyle(d, 'opacity', String(op === undefined ? 1 : op));
    }
    function dotsDone() { for (var i = T.dotI; i < T.dots.length; i++) setStyle(T.dots[i], 'opacity', '0'); }
    /* A value travelling from point a to point b. t in (0, 1) places the head (eased) with a short tail;
       outside (0, 1) nothing is drawn: the value has not left yet, or it has arrived and the caller
       draws what it produced from t >= 1 on. */
    function comet(a, b, t, color, r) {
      if (t <= 0 || t >= 1) return;
      r = r || 4.2;
      var horiz = Math.abs(b.x - a.x) > 2 * Math.abs(b.y - a.y);
      var c1 = horiz ? { x: (a.x + b.x) / 2, y: a.y } : { x: a.x, y: (a.y + b.y) / 2 };
      var c2 = horiz ? { x: (a.x + b.x) / 2, y: b.y } : { x: b.x, y: (a.y + b.y) / 2 };
      var e = ease(t);
      [0.12, 0.06, 0].forEach(function (lag, j) {
        var q = e - lag; if (q <= 0) return;
        var u = 1 - q, A = u * u * u, B = 3 * u * u * q, Cc = 3 * u * q * q, Dd = q * q * q;
        dot(A * a.x + B * c1.x + Cc * c2.x + Dd * b.x, A * a.y + B * c1.y + Cc * c2.y + Dd * b.y, color,
            j === 2 ? r : r * (0.5 + 0.2 * j), j === 2 ? 1 : 0.3 + 0.25 * j);
      });
    }
    /* points in tree coordinates */
    function barTop(id, k, v) { var n = T.nodes[id], b = n.bars[k]; return { x: n.box.l + b.x + b.w / 2, y: n.box.t + n.yb - clamp01(v) * n.mh }; }
    function barBase(id, k) { return barTop(id, k, 0); }
    function nodeSide(id, side) { var b = T.nodes[id].box; return side === 't' ? { x: b.x, y: b.t } : side === 'b' ? { x: b.x, y: b.b } : side === 'l' ? { x: b.l, y: b.y } : { x: b.r, y: b.y }; }
    function fromVlm(p) { return { x: p.x, y: T.nodes.vlm.box.t }; }
    /* staggered schedule for n values flowing over [a, b]: launch l, arrival a (bar starts to grow), filled f */
    function wave(n, a, b) {
      var L = b - a, F = L * (n > 1 ? 0.45 : 0.7), FL = L * 0.12, st = n > 1 ? (L - F - FL) / (n - 1) : 0, out = [];
      for (var k = 0; k < n; k++) { var l = a + k * st; out.push({ l: l, a: l + F, f: l + F + FL }); }
      return out;
    }
    function along(keys, f) {
      var tot = 0; keys.forEach(function (k) { tot += T.edges[k].len; });
      var d = f * tot;
      for (var i = 0; i < keys.length; i++) {
        var e = T.edges[keys[i]];
        if (d <= e.len || i === keys.length - 1) return e.path.getPointAtLength(Math.min(d, e.len));
        d -= e.len;
      }
    }
    function lineHot(set) { lineEls.forEach(function (l, i) { setCls(l.el, 'nd-on', set.indexOf(i) >= 0); }); }
    function nodeHot(ids) { Object.keys(T.nodes).forEach(function (id) { setCls(T.nodes[id].g, 'nd-hot', ids.indexOf(id) >= 0); }); }
    function showTyped(nChars, caretOn) {
      var used = 0;
      lineEls.forEach(function (l) {
        var vis = Math.max(0, Math.min(l.n, nChars - used));
        var caret = caretOn && vis > 0 && vis < l.n || (caretOn && vis === l.n && used + l.n === nChars);
        setHTML(l.txt, partial(l.toks, vis, caret));
        used += l.n;
      });
    }

    /* ---------------- chip flight (code line -> node) */
    function rel(r) { var rr = root.getBoundingClientRect(); return { x: r.left - rr.left, y: r.top - rr.top, w: r.width, h: r.height }; }
    function nodeScreen(id) {
      var m = svg.getScreenCTM(); if (!m) return null;
      var b = T.nodes[id].box, pt = svg.createSVGPoint();
      pt.x = b.x; pt.y = b.t + (NODES[id].vec ? T.LY.fs + 2 : b.h / 2);
      var sp = pt.matrixTransform(m), rr = root.getBoundingClientRect();
      return { x: sp.x - rr.left, y: sp.y - rr.top, k: m.a };
    }
    var nCh = chips.length, CH0 = 0.04, CHD = nCh > 5 ? 0.25 : 0.32, CHS = (0.88 - CH0 - CHD) / Math.max(1, nCh - 1);
    function chipLand(k) { return CH0 + k * CHS + CHD; }
    function nodeLand(id) {
      var best = 2;
      chips.forEach(function (c, k) { if (c.id === id) best = Math.min(best, chipLand(k)); });
      return best;
    }
    function flyChips(p) {
      chips.forEach(function (c, k) {
        var f = seg(p, CH0 + k * CHS, CH0 + k * CHS + CHD);
        if (f <= 0 || f >= 1) { setStyle(c.el, 'opacity', '0'); return; }
        setText(c.el, nodeLab(c.id, layoutName));
        var lr = rel(lineEls[c.li].txt.getClientRects()[0] || lineEls[c.li].el.getBoundingClientRect());
        var ns = nodeScreen(c.id); if (!ns) return;
        var cw = c.el.offsetWidth, chh = c.el.offsetHeight;
        var e = ease(f);
        var sx = lr.x + Math.min(cw, lr.w) / 2, sy = lr.y + lr.h / 2;
        var x = lerp(sx, ns.x, e), y = lerp(sy, ns.y, e) - Math.sin(Math.PI * e) * 40;
        var sc = lerp(1, (T.LY.fs * ns.k) / 12, e);
        setStyle(c.el, 'transform', 'translate(' + (x - cw / 2).toFixed(1) + 'px,' + (y - chh / 2).toFixed(1) + 'px) scale(' + sc.toFixed(3) + ')');
        setStyle(c.el, 'opacity', String((Math.min(1, f / 0.12) * Math.min(1, (1 - f) / 0.14)).toFixed(3)));
      });
    }
    function hideChips() { chips.forEach(function (c) { setStyle(c.el, 'opacity', '0'); }); }
    /* typing: starts right away and ends at TY1 of the opening step; each line's role colour
       fades in as soon as that line is complete, so nothing is shown before it is typed */
    var TY0 = 0.1, TY1 = 0.72, TFADE = 0.1, lineEnd = [], lacc = 0;
    lineEls.forEach(function (l) { lacc += l.n; lineEnd.push(lacc / totalChars); });
    function buildPhase(P1, P2) {   // shared: typing (P1) and code -> tree (P2)
      var u = seg(P1, TY0, TY1);
      showTyped(Math.round(totalChars * u), u > 0 && u < 1);
      setStyle(pre, 'backgroundColor', 'rgba(248, 249, 250, ' + seg(P1, 0, TY0).toFixed(3) + ')');
      lineEls.forEach(function (l, i) {
        var tEnd = TY0 + (TY1 - TY0) * lineEnd[i];
        var a = seg(P1, tEnd, tEnd + TFADE).toFixed(3);
        if (memo(l.el, '--a', a)) l.el.style.setProperty('--a', a);
      });
      if (P2 > 0 && P2 < 1) flyChips(P2); else hideChips();
      Object.keys(NODES).forEach(function (id) {
        if (id === 'vlm' || id === 'loss') return;
        var l = nodeLand(id);
        nodeOp(id, seg(P2, l - 0.03, l + 0.04));
      });
      EDGES.forEach(function (e) {
        if (e[0] === 'vlm' || e[1] === 'loss') return;
        var st = Math.max(nodeLand(e[0]), nodeLand(e[1]));
        edgeDraw(e[0] + '>' + e[1], seg(P2, st, st + 0.08));
      });
      var top = nodeLand(C.region.top);
      setOp(T.region, seg(P2, Math.min(nodeLand(C.region.bottom[0]), top) - 0.03, top + 0.06));
      setOp(T.regionT, seg(P2, top, top + 0.08));
    }

    /* ================================================================ steps */
    var GREEN = '#40c057', BLUE = '#4dabf7', PURPLE = '#9775fa', ORANGE = '#e8590c';
    var FLY = 0.55, GROW = 0.25;     // one scoring slot: the score travels from the VLM, then its bar grows
    var steps, applyMode;
    if (mode === 'fwd') {
      /* ---------------------------------------------- forward (Figure 2, CLEVR) */
      var SPH = R.sphere, SML = R.small, AND = R.and, answers = R.answers || {};
      var pass = AND.map(function (v) { return v >= 0.5; });
      var passIdx = []; pass.forEach(function (ok, i) { if (ok) passIdx.push(i); });
      var colorsOrdered = [];
      passIdx.forEach(function (i) { if (answers[i] !== undefined && colorsOrdered.indexOf(answers[i]) < 0) colorsOrdered.push(answers[i]); });
      var setStr = function (list) { return list.length ? '{ ' + list.map(function (c) { return '"' + c + '"'; }).join(', ') + ' }' : 'set()  =  { }'; };
      var iterW = pass.map(function (ok) { return ok ? 2.3 : 1; });
      var iterTot = iterW.reduce(function (a, b) { return a + b; }, 0);
      var iterAt = function (u) {
        var t = u * iterTot, acc = 0;
        for (var k = 0; k < N; k++) {
          if (t < acc + iterW[k] || k === N - 1) return { k: k, q: clamp01((t - acc) / iterW[k]) };
          acc += iterW[k];
        }
      };
      var listVals = function (arr, idx) { return idx.map(function (i) { return 'box ' + i + ' (' + f2(arr[i]) + ')'; }).join(', '); };
      var WAND = wave(N, 0.02, 0.98);
      steps = [
        { t: 'Program', dur: 3400, cap: 'NePTune gets an image and a question (Figure&nbsp;2 in the paper), and an LLM translates the question into a Python program. <span class="nd-c-con">Green</span> lines ask the VLM for concept scores, the <span class="nd-c-sl">blue</span> line composes them with soft logic, and the <span class="nd-c-imp">purple</span> lines are Python control flow.' },
        { t: 'Code to tree', dur: 7000, cap: 'Each line becomes a node of a tree. The <span class="nd-c-con">green</span> and <span class="nd-c-sl">blue</span> nodes form a declarative first-order-logic formula that reasons over all boxes at once. The <span class="nd-c-imp">purple</span> nodes are the imperative Python around it.' },
        { t: 'Detect', dur: 3600, cap: 'Grounding DINO proposes ' + N + ' boxes. From here on, every declarative node holds a vector with one soft truth value per box (the bars, box&nbsp;0 to ' + (N - 1) + ').' },
        { t: 'Concept scores', dur: 9000, cap: '<code>score()</code> draws a red box around one object at a time and asks the VLM a yes/no question. The score is p(Yes), computed from the logits of the &ldquo;Yes&rdquo; and &ldquo;No&rdquo; tokens. Each score flows into its bar in <span class="nd-c-con">sphere</span> or <span class="nd-c-con">small</span>.' },
        { t: 'Soft AND', dur: 7200, cap: '<code>&amp;</code> composes the two vectors with the soft AND, the element-wise minimum (Table&nbsp;2). The two scores of each box flow into its <span class="nd-c-sl">small_sphere</span> bar, which is high exactly where both are high: ' + listVals(AND, passIdx) + '. ' + (R.and_note || '') },
        { t: 'Loop and branch', dur: 3000 + 1000 * iterTot, cap: 'Python takes over. <span class="nd-c-imp">for</span> visits every box, <span class="nd-c-imp">if</span> turns its soft score into True or False (a score of at least 0.5 is True), and for each True box <span class="nd-c-imp">query()</span> asks the VLM for the color. The answers go into a set.' },
        { t: 'Answer', dur: 4800, cap: 'The VLM answered ' + (function () {
            var cnt = {}, words = ['', 'one', 'two', 'three', 'four', 'five'];
            passIdx.forEach(function (i) { cnt[answers[i]] = (cnt[answers[i]] || 0) + 1; });
            var parts = colorsOrdered.map(function (c) { return '&ldquo;' + esc(c) + '&rdquo; for ' + (words[cnt[c]] || cnt[c]) + (cnt[c] > 1 ? ' small spheres' : ' small sphere'); });
            return parts.length > 1 ? parts.slice(0, -1).join(', ') + ' and ' + parts[parts.length - 1] : parts.join('');
          })() + ', so the set holds ' + colorsOrdered.length + ' colors and <code>len(colors)</code> returns <b>' + R.answer + '</b>.' }
      ];
      applyMode = function (i, p) {
        p = clamp01(p / steps[i].e);
        var P = steps.map(function (_, k) { return k < i ? 1 : k === i ? p : 0; });
        dotsReset();
        setOp(imgP, 1);
        buildPhase(P[0], P[1]);
        setOp(T.impT, seg(P[1], nodeLand('len'), nodeLand('len') + 0.08));
        boxEls.forEach(function (b, k) { var a = 0.06 + (0.6 * k) / N; setOp(b.g, seg(P[2], a, a + 0.12)); });
        nodeOp('vlm', seg(P[3], 0, 0.05));
        ['vlm>sphere', 'vlm>small', 'vlm>query'].forEach(function (k) { edgeDraw(k, seg(P[3], 0.01, 0.06)); });
        var hot = [], lines = [], red = -1, cap = '';
        ['sphere', 'small', 'and'].forEach(function (id) { cur(id, null); badge(id, '', '', 0); });

        /* concept scores: one box at a time; the score travels up from the VLM and the bar grows on arrival */
        var uS = seg(P[3], 0.06, 0.52) * N, uM = seg(P[3], 0.54, 1) * N, fillS = [], fillM = [], fillA = [];
        for (var k = 0; k < N; k++) { fillS.push(ease(seg(uS - k, FLY, FLY + GROW))); fillM.push(ease(seg(uM - k, FLY, FLY + GROW))); }
        setBars('sphere', SPH, fillS); setBars('small', SML, fillM);
        if (i === 3) {
          var lane = function (id, u, vals, line, question) {
            if (u <= 0 || u >= N) return;
            var j = Math.min(N - 1, Math.floor(u)), q = u - j;
            red = j; cur(id, j); lines = [line]; hot = [id]; cap = question;
            comet(fromVlm(barBase(id, j)), barBase(id, j), seg(q, 0, FLY), GREEN);
            if (q >= FLY) badge(id, 'box ' + j + ': ' + f2(vals[j]), 'con', 1);
          };
          lane('sphere', uS, SPH, 1, 'Is the object in the red bounding box a sphere?');
          lane('small', uM, SML, 2, 'Is the object in the red bounding box small?');
        }

        /* soft AND: the two scores of every box flow into its AND bar, which grows on arrival (a wave over the boxes) */
        for (var k2 = 0; k2 < N; k2++) fillA.push(ease(seg(P[4], WAND[k2].a, WAND[k2].f)));
        setBars('and', AND, fillA);
        if (i === 4) {
          lines = [3]; hot = ['and'];
          var lastK = -1;
          for (var k3 = 0; k3 < N; k3++) {
            var tt = seg(P[4], WAND[k3].l, WAND[k3].a);
            comet(barTop('sphere', k3, SPH[k3]), barBase('and', k3), tt, GREEN, 3.6);
            comet(barTop('small', k3, SML[k3]), barBase('and', k3), tt, GREEN, 3.6);
            if (P[4] >= WAND[k3].a) lastK = k3;
          }
          if (lastK >= 0) { cur('and', lastK); cur('sphere', lastK); cur('small', lastK);
            badge('and', 'min(' + f2(SPH[lastK]) + ', ' + f2(SML[lastK]) + ') = ' + f2(AND[lastK]), 'sl', 1); }
        }

        /* loop and branch */
        var setList = [], passShown = [], ansShown = {}, forTxt = '', ifTxt = '', qTxt = '';
        setOp(T.chip, 0);
        var finalLoop = function () { setList = colorsOrdered.slice(); pass.forEach(function (ok, k) { passShown[k] = ok; if (ok) ansShown[k] = true; }); };
        if (i > 5) finalLoop();
        if (i === 5) {
          var u = seg(p, 0.02, 1.0);
          var it = iterAt(u), kk = it.k, qq = it.q;
          for (var m = 0; m < kk; m++) { passShown[m] = pass[m]; if (pass[m]) { ansShown[m] = true; if (setList.indexOf(answers[m]) < 0) setList.push(answers[m]); } }
          if (u > 0 && u < 1) {
            red = kk; cur('and', kk);
            forTxt = 'i = ' + kk; lines = [5]; hot = ['for'];
            if (qq > 0.16) {
              /* the box's soft value travels to the if node; the verdict appears on arrival */
              comet(barTop('and', kk, AND[kk]), nodeSide('if', 'b'), seg(qq, 0.16, 0.38), BLUE);
              lines = [6]; hot = ['if'];
              if (qq >= 0.38) ifTxt = 'small_sphere[' + kk + '] = ' + f2(AND[kk]) + (pass[kk] ? '  ≥ 0.5 → True' : '  < 0.5 → False');
            }
            if (qq >= 0.38) passShown[kk] = pass[kk];
            if (pass[kk] && qq > 0.45) {
              lines = qq < 0.7 ? [7] : [8]; hot = qq < 0.7 ? ['query'] : ['set'];
              comet(fromVlm(nodeSide('query', 'b')), nodeSide('query', 'b'), seg(qq, 0.45, 0.62), GREEN);
              if (qq >= 0.62) { qTxt = '“' + answers[kk] + '”'; ansShown[kk] = true; }
              if (qq > 0.7 && qq < 0.97) {
                var pt = along(['query>if', 'if>for', 'for>set'], ease(seg(qq, 0.7, 0.97)));
                setText(T.chipT, '"' + answers[kk] + '"');
                var cw = Math.max(44, String(answers[kk]).length * 7.5 + 22);
                setAttr(T.chipR, 'x', (-cw / 2).toFixed(1)); setAttr(T.chipR, 'width', cw.toFixed(1));
                setAttr(T.chip, 'transform', 'translate(' + pt.x.toFixed(1) + ',' + pt.y.toFixed(1) + ')');
                setOp(T.chip, 1);
              }
              if (qq >= 0.97 && setList.indexOf(answers[kk]) < 0) setList.push(answers[kk]);
            }
          } else if (u >= 1) finalLoop();
        }
        badge('for', forTxt, 'imp', forTxt ? 1 : 0);
        badge('query', qTxt, 'imp', qTxt ? 1 : 0);
        setText(T.nodes['if'].sub, ifTxt || 'True when the soft score is at least 0.5');
        setText(T.nodes.set.sub, setStr(setList));

        /* answer: the set flows into len(), the result appears on arrival */
        if (i === 6) { lines = [9]; hot = ['len']; comet(nodeSide('set', 't'), nodeSide('len', 'b'), seg(p, 0.05, 0.45), PURPLE, 5); }
        setText(T.ansT, '= ' + R.answer);
        T.ans.querySelector('rect').style.fill = '#7048e8';
        setOp(T.ans, i === 6 ? seg(p, 0.45, 0.55) : 0);
        badge('len', '', '', 0);
        boxEls.forEach(function (b, k) {
          var isRed = k === red, isPass = !isRed && passShown[k] === true, isFail = !isRed && passShown[k] === false;
          setCls(b.r, 'nd-red', isRed); setCls(b.tb, 'nd-red', isRed);
          setCls(b.r, 'nd-pass', isPass); setCls(b.tb, 'nd-pass', isPass);
          if (P[2] >= 1) setOp(b.g, isFail && i >= 5 ? 0.35 : 1);
        });
        Object.keys(ansEls).forEach(function (k) { setOp(ansEls[k], ansShown[k] ? 1 : 0); });
        setText(imgCap, cap || (i === 2 ? N + ' boxes from Grounding DINO' : ''));
        lineHot(lines); nodeHot(hot);
        dotsDone();
      };
    } else {
      /* ---------------------------------------------- Ref-GTA, end to end */
      var ST = R.steps, K = ST.length - 1, G = R.gt_index, S0 = ST[0], SK = ST[K];
      var argmax = function (a) { var b = 0; for (var i = 0; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
      var pred0 = argmax(S0.out), predK = argmax(SK.out);
      var vec = function (a) { return '[' + a.map(f2).join(', ') + ']'; };
      var WA2 = wave(N, 0.02, 0.46), IO_L = 0.5, IO_A = 0.74, IO_F = 0.86;     // forward pass schedule
      var LS_L = 0.3, LS_A = 0.6;                                               // loss: values flow into the loss
      var G1 = [0.02, 0.22], G2 = [0.26, 0.44], G3 = [0.48, 0.66], G4 = [0.7, 0.86];   // gradient, stage by stage
      var ROUNDS = 5, PULSE = 0.6;                                              // tuning: gradient pulse, then the update
      steps = [
        { t: 'Program', dur: 3200, cap: 'NePTune also grounds referring expressions. Expression&nbsp;1 of Figure&nbsp;3 in the paper, <i>' + esc(R.question) + '</i>, comes from Ref-GTA, a video game domain that is new to VLMs trained on natural images. Its program is a single declarative formula: two concept scores, a soft AND and <code>iota</code>, which picks the best-matching box.' },
        { t: 'Program tree', dur: 5000, cap: 'As a tree, the concept scores sit at the leaves, the soft AND above them and <code>iota</code> at the root. Every node is a differentiable operation, so the whole formula is <span class="nd-c-grad">differentiable end to end</span>.' },
        { t: 'Concept scores', dur: 6000, cap: 'Grounding DINO proposes ' + N + ' boxes, and the VLM scores each one: <span class="nd-c-con">is_man</span> = ' + vec(S0.man) + ' and <span class="nd-c-con">is_wearing_black</span> = ' + vec(S0.black) + '.' },
        { t: 'Forward pass', dur: 6800, cap: 'The scores flow up the formula. The soft AND takes the minimum per box, ' + vec(S0.and) + ', and <code>iota</code> turns these into a distribution over the boxes (a softmax, Table&nbsp;2): ' + vec(S0.out) + '. NePTune selects box&nbsp;' + pred0 + (pred0 === G ? ', the man in the suit.' : '.') },
        { t: 'Loss', dur: 6000, cap: 'To fine-tune, the paper compares the program&rsquo;s output with the ground-truth box (box&nbsp;' + G + ') using binary cross-entropy. The answer flows into the loss: L = ' + f2(S0.loss) + '.' },
        { t: 'Gradient', dur: 7600, cap: 'The gradient of the loss flows back through <code>iota</code> and the soft AND into the concept scores, and from each score, through its Yes/No logits, into the VLM. At the AND it goes to the smaller of the two scores of each box. ' + (R.route_note || '') },
        { t: 'Tuning', dur: 10000, cap: 'Each update sends the gradient through the program and then tunes the VLM&rsquo;s concept scores: the score of the target box rises, the other box falls, and the loss drops from ' + f2(S0.loss) + ' to ' + f2(SK.loss) + ' over ' + K + ' steps. ' + (pred0 !== G && predK === G ? 'The selection moves to the right box.' : 'The selection becomes more confident.') },
        { t: 'In the paper', dur: 7600, cap: 'This is how the paper adapts NePTune to Ref-GTA. Fine-tuning a 1B VLM through the program with only 1,000 samples raises NePTune (1B) from 34.92% to <b>69.90%</b>, while standard fine-tuning of the same VLM reaches 32.61% (Table&nbsp;7).' }
      ];
      var L0 = ST.map(function (st) { return st.loss; }), lmax = Math.max.apply(null, L0) * 1.05;
      var recvLeaf = function (vM, vB, k) { return vM[k] <= vB[k] ? 'man' : 'black'; };   // min routes the gradient to the smaller input
      /* one gradient pulse from the loss down to the VLM, stages at [0, .25, .5, .75, 1] of u */
      var pulse = function (u, vM, vB, vA, vO, r) {
        var st = [[0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1]];
        for (var k = 0; k < N; k++) {
          var leaf = recvLeaf(vM, vB, k), lv = leaf === 'man' ? vM[k] : vB[k];
          comet(nodeSide('loss', 'l'), barTop('iota', k, vO[k]), seg(u, st[0][0], st[0][1]), ORANGE, r);
          comet(barBase('iota', k), barTop('and', k, vA[k]), seg(u, st[1][0], st[1][1]), ORANGE, r);
          comet(barBase('and', k), barTop(leaf, k, lv), seg(u, st[2][0], st[2][1]), ORANGE, r);
          comet(barBase(leaf, k), fromVlm(barBase(leaf, k)), seg(u, st[3][0], st[3][1]), ORANGE, r);
        }
      };
      applyMode = function (i, p) {
        p = clamp01(p / steps[i].e);
        var P = steps.map(function (_, k) { return k < i ? 1 : k === i ? p : 0; });
        dotsReset();
        setOp(imgP, 1);
        buildPhase(P[0], P[1]);
        nodeOp('vlm', seg(P[2], 0, 0.06));
        edgeDraw('vlm>man', seg(P[2], 0.01, 0.08)); edgeDraw('vlm>black', seg(P[2], 0.01, 0.08));
        nodeOp('loss', seg(P[4], 0.1, 0.22));
        edgeDraw('iota>loss', seg(P[4], 0.15, 0.28));

        /* values: tuning runs in ROUNDS; in each round the gradient pulse travels first, then the scores move */
        var uT = seg(P[6], 0.02, 1) * ROUNDS, rc = Math.min(ROUNDS - 1, Math.floor(uT)), qT = P[6] >= 1 ? 1 : uT - rc;
        var i0 = Math.round(rc * K / ROUNDS), i1 = Math.round((rc + 1) * K / ROUNDS), w = ease(seg(qT, PULSE + 0.02, 0.98));
        var A = ST[i0], B = ST[i1];
        var mix = function (key) { return A[key].map(function (v, k) { return lerp(v, B[key][k], w); }); };
        var vM = mix('man'), vB = mix('black'), vA = mix('and'), vO = mix('out'), lossNow = lerp(A.loss, B.loss, w);
        var sIdx = i0 + (i1 - i0) * w;
        var lines = [], hot = [], red = -1, cap = '';
        ['man', 'black', 'and', 'iota'].forEach(function (id) { cur(id, null); badge(id, '', '', 0); });

        /* concept scores: one box at a time, the bar grows when the score arrives */
        var uM = seg(P[2], 0.12, 0.56) * N, uB = seg(P[2], 0.58, 1) * N, fM = [], fB = [], fA = [], fO = [];
        for (var k = 0; k < N; k++) { fM.push(ease(seg(uM - k, FLY, FLY + GROW))); fB.push(ease(seg(uB - k, FLY, FLY + GROW))); }
        setBars('man', vM, fM); setBars('black', vB, fB);
        boxEls.forEach(function (b, k) { setOp(b.g, seg(P[2], 0.02 + 0.04 * k, 0.08 + 0.04 * k)); });
        if (i === 2) {
          var lane = function (id, u, vals, line, question) {
            if (u <= 0 || u >= N) return;
            var j = Math.min(N - 1, Math.floor(u)), q = u - j;
            red = j; cur(id, j); lines = [line]; hot = [id]; cap = question;
            comet(fromVlm(barBase(id, j)), barBase(id, j), seg(q, 0, FLY), GREEN);
            if (q >= FLY) badge(id, 'box ' + j + ': ' + f2(vals[j]), 'con', 1);
          };
          lane('man', uM, vM, 0, 'Is the main object inside of the red bounding box a man?');
          lane('black', uB, vB, 1, 'Is the man inside of the red bounding box wearing black?');
          if (!cap) cap = N + ' boxes from Grounding DINO';
        }

        /* forward pass: scores flow into the AND bars (each grows on arrival), then all AND values flow into iota together */
        for (var k2 = 0; k2 < N; k2++) { fA.push(ease(seg(P[3], WA2[k2].a, WA2[k2].f))); fO.push(ease(seg(P[3], IO_A, IO_F))); }
        setBars('and', vA, fA); setBars('iota', vO, fO);
        if (i === 3) {
          var lastK = -1;
          for (var k3 = 0; k3 < N; k3++) {
            var tt = seg(P[3], WA2[k3].l, WA2[k3].a);
            comet(barTop('man', k3, vM[k3]), barBase('and', k3), tt, GREEN);
            comet(barTop('black', k3, vB[k3]), barBase('and', k3), tt, GREEN);
            comet(barTop('and', k3, vA[k3]), barBase('iota', k3), seg(P[3], IO_L, IO_A), BLUE);
            if (P[3] >= WA2[k3].a) lastK = k3;
          }
          if (lastK >= 0 && P[3] < IO_L) { cur('and', lastK); badge('and', 'min(' + f2(vM[lastK]) + ', ' + f2(vB[lastK]) + ') = ' + f2(vA[lastK]), 'sl', 1); }
          lines = P[3] < IO_L - 0.02 ? [2] : [3]; hot = P[3] < IO_L - 0.02 ? ['and'] : ['iota'];
        }
        var sel = argmax(vO), selOp = seg(P[3], IO_F, IO_F + 0.05);
        if (selEl) {
          var sb = R.boxes[sel];
          setAttr(selEl.rect, 'x', sb[0]); setAttr(selEl.rect, 'y', sb[1]); setAttr(selEl.rect, 'width', sb[2] - sb[0]); setAttr(selEl.rect, 'height', sb[3] - sb[1]);
          var lw = 210, lx = Math.min(R.img_w - lw - 6, Math.max(6, sb[0])), ly = Math.max(4, sb[1] - 40);
          setAttr(selEl.bg, 'x', lx); setAttr(selEl.bg, 'y', ly); setAttr(selEl.bg, 'width', lw); setAttr(selEl.bg, 'height', 32);
          setAttr(selEl.t, 'x', lx + 10); setAttr(selEl.t, 'y', ly + 23);
          setText(selEl.t, 'NePTune: box ' + sel + ' (' + f2(vO[sel]) + ')');
          setAttr(selEl, 'class', sel === G ? 'nd-selg ok' : 'nd-selg');
          setOp(selEl, selOp);
          setOp(gtEl, seg(P[4], 0.02, 0.15));
        }

        /* loss: target first, then the answer flows into the loss node and L appears on arrival */
        var tOp = seg(P[4], 0.06, 0.18);
        T.nodes.iota.bars.forEach(function (b, k) {
          var hh = Math.max(2, (k === G ? 1 : 0) * T.nodes.iota.mh);
          setAttr(b.tgt, 'y', (T.nodes.iota.yb - hh).toFixed(1)); setAttr(b.tgt, 'height', hh.toFixed(1)); setOp(b.tgt, tOp);
        });
        var lossShown = P[4] >= LS_A;
        setText(T.nodes.loss.sub, lossShown ? 'L = ' + f2(lossNow) : 'vs. ground truth');
        if (i === 4) {
          lines = [3]; hot = ['loss'];
          for (var k4 = 0; k4 < N; k4++) comet(barTop('iota', k4, vO[k4]), nodeSide('loss', 'l'), seg(P[4], LS_L, LS_A), BLUE);
        }
        badge('iota', lossShown ? 'L = ' + f2(lossNow) : '', 'grad', lossShown ? seg(P[4], LS_A, LS_A + 0.05) : 0);

        /* gradient: loss -> iota -> AND -> the smaller input -> VLM; each node lights up when the pulse arrives */
        var reach = { iota: G1[1], and: G2[1], man: G3[1], black: G3[1] };
        var anyRecv = { man: false, black: false };
        for (var k5 = 0; k5 < N; k5++) anyRecv[recvLeaf(vM, vB, k5)] = true;
        ['iota', 'and', 'man', 'black'].forEach(function (id) { setCls(T.nodes[id].g, 'nd-glow', P[5] >= reach[id] && (id === 'iota' || id === 'and' || anyRecv[id])); });
        setCls(T.nodes.vlm.g, 'nd-glow', P[5] >= G4[1]);
        var glow = P[5] > 0;
        setCls(T.region, 'rg-grad', glow); setCls(T.regionT, 'rg-grad', glow);
        gEdge('iota>loss', seg(P[5], G1[0], G1[1])); gEdge('and>iota', seg(P[5], G2[0], G2[1]));
        gEdge('man>and', anyRecv.man ? seg(P[5], G3[0], G3[1]) : 0); gEdge('black>and', anyRecv.black ? seg(P[5], G3[0], G3[1]) : 0);
        gEdge('vlm>man', anyRecv.man ? seg(P[5], G4[0], G4[1]) : 0); gEdge('vlm>black', anyRecv.black ? seg(P[5], G4[0], G4[1]) : 0);
        if (i === 5) {
          for (var k6 = 0; k6 < N; k6++) {
            var leaf = recvLeaf(vM, vB, k6), lv = leaf === 'man' ? vM[k6] : vB[k6];
            comet(nodeSide('loss', 'l'), barTop('iota', k6, vO[k6]), seg(p, G1[0], G1[1]), ORANGE, 4.6);
            comet(barBase('iota', k6), barTop('and', k6, vA[k6]), seg(p, G2[0], G2[1]), ORANGE, 4.6);
            comet(barBase('and', k6), barTop(leaf, k6, lv), seg(p, G3[0], G3[1]), ORANGE, 4.6);
            comet(barBase(leaf, k6), fromVlm(barBase(leaf, k6)), seg(p, G4[0], G4[1]), ORANGE, 4.6);
          }
          lines = p < G1[1] ? [3] : p < G2[1] ? [2] : [anyRecv.man ? 0 : 1, anyRecv.black ? 1 : 0];
        }
        if (i === 6) pulse(seg(qT, 0, PULSE), vM, vB, vA, vO, 3.6);
        setText(T.nodes.vlm.lab, P[5] >= G4[1] ? (layoutName === 'narrow' ? 'VLM: gradient → logits → LoRA weights' : 'VLM: gradient → Yes/No logits → LoRA weights')
          : 'Vision-language model (VLM)');
        /* arrows: which way gradient descent moves each score, shown where the gradient arrives */
        ['man', 'black', 'and', 'iota'].forEach(function (id) {
          var n = T.nodes[id], aOp = seg(P[5], reach[id], reach[id] + 0.05);
          n.bars.forEach(function (b, k) {
            var up = k === G;
            var recv = id === 'and' || id === 'iota' || recvLeaf(vM, vB, k) === id;
            var v = id === 'man' ? vM[k] : id === 'black' ? vB[k] : id === 'and' ? vA[k] : vO[k];
            var room = up ? 1 - v : v;
            var op = recv && room > 0.02 ? aOp : 0;
            var top = n.yb - v * n.mh, sz = Math.min(6, b.w / 4), cx = N <= 4 ? b.x + b.w - sz - 4 : b.x + b.w / 2;
            var ay = Math.max(n.yb - n.mh + sz * 1.6, top - 3);
            var d = up ? 'M' + (cx - sz) + ',' + ay + ' L' + (cx + sz) + ',' + ay + ' L' + cx + ',' + (ay - sz * 1.5) + 'Z'
                       : 'M' + (cx - sz) + ',' + (ay - sz * 1.5) + ' L' + (cx + sz) + ',' + (ay - sz * 1.5) + ' L' + cx + ',' + ay + 'Z';
            setAttr(b.arr, 'd', d); setOp(b.arr, op);
          });
        });

        /* tuning table + loss curve */
        tuneRows.forEach(function (r, k) {
          var vals = [vM[k], vB[k], vA[k], vO[k]];
          r.c.forEach(function (c, j) {
            setStyle(c.firstChild, 'width', (100 * vals[j]).toFixed(1) + '%');
            setText(c.lastChild, f2(vals[j]));
          });
          setCls(r.row, 'nd-win', selOp >= 1 && k === sel);
        });
        var d2 = '';
        for (var qn = 0; qn <= Math.floor(sIdx); qn++) d2 += (qn ? 'L' : 'M') + (200 * qn / K).toFixed(1) + ',' + (36 - 34 * L0[qn] / lmax).toFixed(1);
        if (sIdx % 1 > 0) d2 += 'L' + (200 * sIdx / K).toFixed(1) + ',' + (36 - 34 * lossNow / lmax).toFixed(1);
        setAttr(spark, 'd', lossShown ? d2 : '');
        setAttr(sparkDot, 'cx', (200 * sIdx / K).toFixed(1)); setAttr(sparkDot, 'cy', (36 - 34 * lossNow / lmax).toFixed(1));
        setOp(sparkDot, lossShown ? 1 : 0);
        setText(lossTxt, lossShown ? 'loss ' + f2(lossNow) + ' · step ' + Math.round(sIdx) : 'loss');
        boxEls.forEach(function (b, k) { setCls(b.r, 'nd-red', k === red); setCls(b.tb, 'nd-red', k === red); });
        setText(imgCap, cap);
        if (i === 7) { hot = ['iota']; }
        lineHot(lines); nodeHot(hot);
        dotsDone();
      };
    }

    var HOLD = 900;
    steps.forEach(function (st) { st.e = Math.max(0.5, 1 - HOLD / st.dur); });
    var TOT = 0, CUM = [];
    steps.forEach(function (st) { CUM.push(TOT); TOT += st.dur; });
    function globalPos(i, pr) { return (CUM[i] + clamp01(pr) * steps[i].dur) / TOT; }
    steps.forEach(function (st, k) {     // step boundaries as ticks on the single progress bar
      if (!k) return;
      var tk = h('b', 'nd-tick'); tk.style.left = (100 * CUM[k] / TOT).toFixed(3) + '%'; progEl.appendChild(tk);
    });
    var pillEls = steps.map(function (st, i) {
      var b = h('button', 'nd-pill', '<span>' + (i + 1) + '</span><em>' + st.t + '</em>');
      b.type = 'button'; b.setAttribute('aria-label', 'Step ' + (i + 1) + ': ' + st.t);
      b.addEventListener('click', function () { userJump(i); });
      pills.appendChild(b);
      return b;
    });
    function applyState(i, p) {
      applyMode(i, p);
      pillEls.forEach(function (b, k) { setCls(b, 'nd-on', k === i); setCls(b, 'nd-done', k < i); });
      if (capEl.dataset.step !== String(i)) { capEl.innerHTML = steps[i].cap; capEl.dataset.step = String(i); }
      setStyle(progBar, 'width', (100 * globalPos(i, p)).toFixed(3) + '%');
    }

    /* ================================================================ layout */
    function pickLayout() {
      var w = treeP.clientWidth || 600;
      var L = w < 520 ? 'narrow' : 'wide';
      if (L !== layoutName) { buildTree(L); return true; }
      return false;
    }
    pickLayout();
    new ResizeObserver(function () { pickLayout(); dirty = true; kick(); }).observe(treeP);

    /* ================================================================ player */
    /* one clock: (step, p) advances at a constant rate; the progress bar shows the global position */
    var step = 0, p = 0, playing = false, last = 0, visible = false, started = false, raf = 0, dirty = true, holdT = 0;
    function setPlayBtn() {
      bPlay.innerHTML = playing ? '<span aria-hidden="true">&#10074;&#10074;</span>' : '<span aria-hidden="true">&#9654;</span>';
      bPlay.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    }
    function play() {
      if (step === steps.length - 1 && p >= 1) { step = 0; p = reduce ? 1 : 0; }
      playing = true; started = true; last = performance.now(); setPlayBtn(); kick();
    }
    function pause() { playing = false; setPlayBtn(); dirty = true; kick(); }
    function go(i, pr) { step = Math.max(0, Math.min(steps.length - 1, i)); p = pr; last = performance.now(); holdT = 0; dirty = true; kick(); }
    function userJump(i) {      // land on the step's end state; when playing, hold there and then continue
      var j = Math.max(0, Math.min(steps.length - 1, i));
      go(j, playing && !reduce ? steps[j].e : 1);
    }
    bPlay.addEventListener('click', function () { if (playing) pause(); else play(); });
    bNext.addEventListener('click', function () { userJump(step + 1); });
    bPrev.addEventListener('click', function () { userJump(p > 0.25 && playing ? step : step - 1); });
    bRe.addEventListener('click', function () { go(0, reduce ? 1 : 0); play(); });

    function frame(now) {
      raf = 0;
      if (playing) {
        var dt = Math.min(100, now - last); last = now;
        if (reduce) { p = 1; holdT += dt; if (holdT >= steps[step].dur) { holdT = 0; advance(0); } }
        else { p += dt / steps[step].dur; if (p >= 1) advance((p - 1) * steps[step].dur); }
        dirty = true;
      }
      if (dirty) { applyState(step, Math.min(1, p)); dirty = playing; }
      if (visible && (playing || dirty)) raf = requestAnimationFrame(frame);
    }
    function advance(overMs) {   // carry the time past the step end into the next step, so the clock never stalls
      if (step < steps.length - 1) { step += 1; p = reduce ? 1 : Math.min(0.99, overMs / steps[step].dur); }
      else { p = 1; playing = false; setPlayBtn(); }
    }
    function kick() { if (!raf && visible) raf = requestAnimationFrame(frame); }
    var io = new IntersectionObserver(function (ents) {
      ents.forEach(function (e) {
        visible = e.isIntersecting;
        if (visible) { if (!started && !reduce) play(); else { dirty = true; kick(); } }
        else if (raf) { cancelAnimationFrame(raf); raf = 0; }
      });
    }, { threshold: 0.25 });
    io.observe(root);
    if (reduce) { step = 0; p = 1; root.classList.add('nd-reduced'); }
    setPlayBtn();
    applyState(step, p);

    /* hooks for automated checks */
    root.__nd = { go: function (i, pr) { playing = false; setPlayBtn(); go(i, pr); applyState(step, p); }, steps: steps.length,
                  ends: steps.map(function (st) { return st.e; }), durs: steps.map(function (st) { return st.dur; }),
                  state: function () { return { step: step, p: p, g: globalPos(step, Math.min(1, p)), playing: playing }; } };
  }

  /* ================================================================== boot */
  var roots = [document.getElementById('npt-fwd'), document.getElementById('npt-ref')].filter(Boolean);
  if (!roots.length) return;
  var src = roots[0].dataset.src;
  fetch(src + 'example.json').then(function (r) {
    if (!r.ok) throw new Error('example.json ' + r.status);
    return r.json();
  }).then(function (D) {
    roots.forEach(function (root) {
      try { new Widget(root, D, root.id === 'npt-ref' ? 'ref' : 'fwd'); }
      catch (err) { fail(root, err); }
    });
  }).catch(function (err) { roots.forEach(function (root) { fail(root, err); }); });
  function fail(root, err) {
    var m = document.createElement('p');
    m.className = 'nd-err';
    m.textContent = 'The animation could not load (' + (err && err.message ? err.message : err) + ').';
    root.appendChild(m);
    if (window.console) console.error(err);
  }
})();
