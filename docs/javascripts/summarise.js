// Summarise per-forecast scores for a selection, as relative_crps does in
// Python: each model's n and the mean of every metric over its own
// forecasts and, for relative metrics, its pairwise relative skill. The
// skill is computed once in Python (nowcasteval.scoring.pairwise_skill) as
// each model's geometric mean of ratios, per selection (d.skill, keyed by
// scoresummary.skill_key); here one model's is divided by the baseline's,
// which only rescales them. Each entry has cells, the baseline's n, and
// partial, whether the model has fewer forecasts than the baseline
// (mark_partial in Python), so its row is read apart from the full rows.
(function (root) {
  "use strict";

  // d.rows: [model, origin, release, place, week, ...one value per metric]
  // d.metrics: [{key, label, ref, relative}]
  // keep(row, index) selects the rows; baseline, a model index, is the
  // model relative metrics are scaled to (d.baseline if not given); group
  // is the selection's skill key (summarise.key), without which relative
  // metrics are missing. Returns one entry per model with rows.
  function summarise(d, keep, baseline, group) {
    if (baseline === undefined || baseline === null) baseline = d.baseline;
    var nm = d.metrics.length;
    var acc = {};
    d.rows.forEach(function (r, i) {
      if (!keep(r, i)) return;
      var a = acc[r[0]];
      if (!a) {
        a = acc[r[0]] = { n: 0, sum: [] };
        for (var j = 0; j < nm; j++) a.sum.push(0);
      }
      a.n += 1;
      for (var k = 0; k < nm; k++) a.sum[k] += r[5 + k];
    });
    var skill = group && d.skill ? d.skill[group] : null;
    var cells = acc[baseline] ? acc[baseline].n : null;
    return Object.keys(acc).map(Number).sort(function (x, y) { return x - y; })
      .map(function (m) {
        var a = acc[m];
        var out = { model: m, n: a.n, cells: cells,
          partial: cells !== null && a.n < cells, mean: {}, rel: {} };
        d.metrics.forEach(function (met, i) {
          out.mean[met.key] = a.sum[i] / a.n;
          if (!met.relative) return;
          var th = skill ? skill[met.key] : null;
          var mine = th ? th[m] : null, base = th ? th[baseline] : null;
          // No skill, or a baseline with none or of 0, gives no relative
          // score, as in Python.
          out.rel[met.key] = mine === null || mine === undefined ||
            base === null || base === undefined || !(base > 0) || !isFinite(base) ?
            null : mine / base;
        });
        return out;
      });
  }

  // The skill key of a selection (scoresummary.skill_key in Python): one
  // round index or null for every round, "p<pool>" or "x<place>", and a
  // horizon or undefined for every horizon.
  summarise.key = function (round, select, horizon) {
    return (round === null || round === undefined ? "all" : round) + "|" +
      select + "|" + (horizon === null || horizon === undefined ? "all" : horizon);
  };

  // A keep function for the rounds (origin indices) given, or every row
  // for null.
  summarise.inRounds = function (rounds) {
    if (rounds === null || rounds === undefined) return function () { return true; };
    var on = {};
    rounds.forEach(function (o) { on[o] = true; });
    return function (r) { return on[r[1]] === true; };
  };

  if (typeof module !== "undefined" && module.exports) module.exports = summarise;
  else root.bdbvSummarise = summarise;
})(this);
