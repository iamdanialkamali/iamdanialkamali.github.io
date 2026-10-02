/* SATURN project page: captions and play-on-view for the rendered perspective-change clip. */
(function () {
  var box = document.getElementById('sat-video');
  if (!box) return;
  var video = box.querySelector('video');
  var cap = box.querySelector('.sat-video-cap span');
  var segs = JSON.parse(box.getAttribute('data-segments') || '[]');
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function update() {
    var t = video.currentTime, text = '';
    for (var i = 0; i < segs.length; i++) {
      if (t >= segs[i][0] && t < segs[i][1]) { text = segs[i][2]; }
    }
    if (cap.textContent !== text) { cap.textContent = text; }
    cap.style.display = text ? '' : 'none';
  }
  video.addEventListener('timeupdate', update);
  video.addEventListener('seeked', update);
  update();
  if (reduce || !('IntersectionObserver' in window)) { return; }
  new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        var p = video.play();
        if (p && p.catch) { p.catch(function () {}); }
      } else {
        video.pause();
      }
    });
  }, { threshold: 0.4 }).observe(box);
})();
