/* AI writing detector front end.

   The scoring engine is the pure standard library aic package from this same
   repository. It is loaded into Pyodide and executed inside the visitor's own
   browser tab, so pasted text is never sent to a server. */
(function () {
  "use strict";

  var PYODIDE_JS = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
  var FALLBACK = "https://cdn.jsdelivr.net/gh/eswarr-dasi/academic-integrity-checker@main/src/aic/";
  var MODULES = [
    "__init__.py", "normalize.py", "fingerprint.py", "index.py",
    "similarity.py", "scoring.py", "ai_detect.py", "ingest.py",
    "report.py", "pipeline.py", "cli.py"
  ];

  var BOOT = [
    "import json, sys",
    "sys.path.insert(0, '/engine')",
    "from aic.normalize import normalize",
    "from aic import ai_detect",
    "_detector = ai_detect.AIDetector()",
    "def aic_check(text):",
    "    doc = normalize(text)",
    "    payload = _detector.analyze(doc).to_dict()",
    "    payload['scorable_words'] = len(doc.tokens)",
    "    payload['sentence_count'] = len(doc.sentences)",
    "    payload['excluded_spans'] = len(doc.exclusions)",
    "    payload['segment_words'] = ai_detect.SEGMENT_WORDS",
    "    payload['min_segment_words'] = ai_detect.MIN_SEGMENT_WORDS",
    "    return json.dumps(payload)",
    ""
  ].join("\n");

  var BAND_META = {
    "human": {
      label: "Human-like",
      cls: "b-human",
      color: "#16a34a",
      tint: "t0",
      note: "No meaningful AI-writing signal. The style markers here sit in the range typical of human academic prose."
    },
    "unclear": {
      label: "Unclear",
      cls: "b-unclear",
      color: "#d97706",
      tint: "t1",
      note: "Genuinely ambiguous. This band is deliberately wide because heavily edited human writing and lightly edited AI writing look almost identical to a statistical detector."
    },
    "likely-ai": {
      label: "Likely AI-assisted",
      cls: "b-likely",
      color: "#ea580c",
      tint: "t2",
      note: "Several style markers line up with generated prose. Treat this as a prompt to read the flagged passages, not as a finding."
    },
    "very-likely-ai": {
      label: "Very likely AI",
      cls: "b-very",
      color: "#dc2626",
      tint: "t3",
      note: "Strong AI-writing signals across most passages. Still not proof: check the passages and talk to the writer about their process."
    }
  };

  var FEATURE_LABEL = {
    sent_len_mean: "Average sentence length, in words",
    sent_len_cv: "Sentence length variation",
    type_token_ratio: "Vocabulary variety",
    hapax_ratio: "Share of words used only once",
    function_word_ratio: "Function word share",
    punct_entropy: "Punctuation variety",
    discourse_marker_rate: "Transition phrase rate",
    burstiness: "Burstiness, needs a language model",
    mean_log_prob: "Mean token log probability, needs a language model",
    log_prob_std: "Log probability spread, needs a language model",
    mean_log_rank: "Mean token rank, needs a language model",
    curvature: "Curvature, needs a language model",
    has_likelihood: "Language model features available"
  };

  var SAMPLE = [
    "In today's rapidly evolving educational landscape, the integration of artificial intelligence has become increasingly significant for institutions around the world. Educators and administrators alike are seeking to understand how these powerful tools can be leveraged effectively while maintaining rigorous academic standards.",
    "Furthermore, it is important to note that the adoption of such technologies presents both opportunities and challenges. On the one hand, automated systems can streamline assessment workflows and provide timely feedback to learners. On the other hand, concerns regarding accuracy, fairness, and transparency continue to be raised by stakeholders across the sector.",
    "Additionally, research has consistently demonstrated that student outcomes improve when feedback is delivered promptly and constructively. In this context, artificial intelligence offers a compelling avenue for scaling personalised support to large cohorts. However, careful consideration must be given to the pedagogical framework within which these tools are deployed.",
    "Moreover, a growing body of literature suggests that institutional readiness plays a pivotal role in determining whether new technologies deliver their intended benefits. Factors such as infrastructure, staff capacity and leadership support have all been identified as critical enablers. Consequently, institutions that invest in comprehensive training programmes are generally better positioned to realise sustainable improvements in teaching and learning outcomes over the long term.",
    "It is also worth noting that ethical considerations remain central to this discussion. Questions of data privacy, algorithmic bias and informed consent must be addressed proactively rather than retrospectively. Therefore, robust governance frameworks are widely regarded as an essential prerequisite for responsible adoption, ensuring that the interests of students and staff are protected throughout the implementation process.",
    "In addition, stakeholders have emphasised the importance of transparency whenever automated systems are used to inform academic judgements. Clear communication about how a decision was reached, combined with meaningful opportunities for appeal, can help to build the institutional trust that successful adoption ultimately depends upon.",
    "In conclusion, while artificial intelligence holds tremendous potential to transform the educational experience, it is essential that institutions approach implementation thoughtfully and deliberately. By establishing clear policies, investing in professional development, and centring the needs of learners, educational organisations can harness these innovations responsibly and sustainably."
  ].join("\n\n");

  var pyodide = null;
  var check = null;

  function el(id) { return document.getElementById(id); }

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function pct(x) { return (x * 100).toFixed(1) + "%"; }

  function countWords(t) {
    var m = String(t).match(/[0-9A-Za-z\u00c0-\u024f'\u2019]+/g);
    return m ? m.length : 0;
  }

  function status(msg, cls) {
    var e = el("engine");
    if (!e) { return; }
    e.textContent = msg;
    e.className = "engine " + (cls || "");
  }

  function meta(band) { return BAND_META[band] || BAND_META["unclear"]; }

  function kv(key, value) {
    return "<li><b>" + esc(key) + "</b><span>" + esc(value) + "</span></li>";
  }

  function segmentList(data, text) {
    var segs = data.segments || [];
    if (!segs.length) {
      return "<p class=\"bandnote\">Too short to fill a scoring window, so there is no passage breakdown.</p>";
    }
    return segs.map(function (s, i) {
      var m = meta(s.band);
      var snippet = text.slice(s.span[0], s.span[1]).replace(/\s+/g, " ").trim();
      if (snippet.length > 200) { snippet = snippet.slice(0, 200) + "..."; }
      return "<div class=\"seg\">" +
        "<div class=\"seg-top\"><span>Passage " + (i + 1) + " &middot; " + s.words + " words</span>" +
        "<strong>" + pct(s.score) + " <span class=\"chip " + m.cls + "\">" + m.label + "</span></strong></div>" +
        "<div class=\"bar\"><i style=\"width:" + Math.max(2, s.score * 100).toFixed(1) + "%;background:" + m.color + "\"></i></div>" +
        "<p>" + esc(snippet) + "</p></div>";
    }).join("");
  }

  function highlighted(data, text) {
    var segs = (data.segments || []).slice().sort(function (a, b) { return a.span[0] - b.span[0]; });
    if (!segs.length) { return ""; }
    var out = "";
    var cursor = 0;
    segs.forEach(function (s) {
      if (s.span[0] > cursor) { out += esc(text.slice(cursor, s.span[0])); }
      var m = meta(s.band);
      out += "<mark class=\"" + m.tint + "\" title=\"" + pct(s.score) + " - " + m.label + "\">" +
        esc(text.slice(s.span[0], s.span[1])) + "</mark>";
      cursor = s.span[1];
    });
    if (cursor < text.length) { out += esc(text.slice(cursor)); }
    return "<div class=\"blk\"><h3>Highlighted text</h3><div class=\"hl\">" + out + "</div></div>";
  }

  function diagnostics(data) {
    var f = data.features || {};
    var rows = Object.keys(f).map(function (k) {
      var v = f[k];
      if (typeof v === "boolean") { v = v ? "yes" : "no"; } else { v = Number(v).toFixed(4); }
      return "<tr><td>" + esc(FEATURE_LABEL[k] || k) + "</td><td class=\"num\">" + esc(v) + "</td></tr>";
    }).join("");
    return "<details class=\"diag\"><summary>Show the measured writing signals</summary>" +
      "<table><thead><tr><th>Signal</th><th class=\"num\">Value</th></tr></thead><tbody>" + rows +
      "</tbody></table><p class=\"bandnote\">These are whole-document diagnostics. The headline index is the word weighted mean of the passage scores, so a long document is never scored higher just for being long.</p></details>";
  }

  function render(data, text) {
    var m = meta(data.band);
    var p = data.index * 100;
    var ci = data.confidence_interval || [0, 1];
    var caveats = (data.caveats || []).slice();
    caveats.push("An AI-writing index is a style signal, not proof of misconduct. It must never be the sole basis for an accusation.");

    var html = "<div class=\"scorecard\">" +
      "<div class=\"ring\" style=\"--pct:" + p.toFixed(1) + ";--ringc:" + m.color + "\"><b>" +
      p.toFixed(1) + "%</b><small>AI index</small></div>" +
      "<div class=\"meta\"><span class=\"chip " + m.cls + "\">" + m.label + "</span>" +
      "<p class=\"bandnote\">" + m.note + "</p>" +
      "<p class=\"bandnote\">Confidence interval " + pct(ci[0]) + " to " + pct(ci[1]) + " &middot; " +
      (data.calibrated ? "calibrated" : "uncalibrated reference model") + " &middot; " +
      (data.used_language_model ? "with language model features" : "style features only") +
      "</p></div></div>";

    html += "<ul class=\"kv\">" +
      kv("Words scored", data.scorable_words) +
      kv("Sentences", data.sentence_count) +
      kv("Passages", (data.segments || []).length) +
      kv("Window size", data.segment_words + " words") +
      "</ul>";

    html += "<div class=\"notice\"><h4>Read this before acting on the number</h4><ul>" +
      caveats.map(function (c) { return "<li>" + esc(c) + "</li>"; }).join("") +
      "</ul></div>";

    html += "<div class=\"blk\"><h3>Passage breakdown</h3>" + segmentList(data, text) + "</div>";
    html += highlighted(data, text);
    html += diagnostics(data);
    return html;
  }

  function run() {
    var input = el("input");
    var text = input.value;
    if (!check) {
      status("The engine is still starting. Give it a few seconds.", "");
      return;
    }
    var n = countWords(text);
    if (n < 25) {
      status("Paste at least 25 words. Below roughly 300 words any AI score is very unstable.", "bad");
      return;
    }
    el("run").disabled = true;
    status("Scoring on this device...", "");
    setTimeout(function () {
      try {
        var data = JSON.parse(check(text));
        el("empty").hidden = true;
        var box = el("result");
        box.hidden = false;
        box.innerHTML = render(data, text);
        el("stamp").textContent = data.scorable_words.toLocaleString() + " words scored";
        status(n < 300
          ? "Done, but this text is short. Treat the number as indicative only."
          : "Done. Your text never left this device.", n < 300 ? "" : "ok");
      } catch (err) {
        status("Scoring failed: " + err.message, "bad");
      }
      el("run").disabled = false;
    }, 20);
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement("script");
      s.src = src;
      s.onload = function () { resolve(); };
      s.onerror = function () { reject(new Error("could not load " + src)); };
      document.head.appendChild(s);
    });
  }

  function grab(name) {
    return fetch("src/aic/" + name).then(function (r) {
      if (!r.ok) { throw new Error(String(r.status)); }
      return r.text();
    }).catch(function () {
      return fetch(FALLBACK + name).then(function (r) {
        if (!r.ok) { throw new Error("could not fetch " + name); }
        return r.text();
      });
    });
  }

  function boot() {
    status("Downloading the Python runtime, about 10 MB, once per visit...");
    loadScript(PYODIDE_JS).then(function () {
      return loadPyodide();
    }).then(function (py) {
      pyodide = py;
      status("Loading the detection engine...");
      pyodide.FS.mkdirTree("/engine/aic");
      return Promise.all(MODULES.map(grab));
    }).then(function (sources) {
      MODULES.forEach(function (name, i) {
        pyodide.FS.writeFile("/engine/aic/" + name, sources[i]);
      });
      return pyodide.runPythonAsync(BOOT);
    }).then(function () {
      check = pyodide.globals.get("aic_check");
      el("run").disabled = false;
      status("Engine ready. Nothing you paste is uploaded.", "ok");
    }).catch(function (err) {
      status("Could not start the engine: " + err.message, "bad");
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var input = el("input");
    function sync() {
      el("wc").textContent = countWords(input.value).toLocaleString() + " words";
    }
    input.addEventListener("input", sync);
    el("run").addEventListener("click", run);
    el("sample").addEventListener("click", function () {
      input.value = SAMPLE;
      sync();
    });
    el("clear").addEventListener("click", function () {
      input.value = "";
      sync();
      el("result").hidden = true;
      el("empty").hidden = false;
      el("stamp").textContent = "";
      status("Engine ready. Nothing you paste is uploaded.", "ok");
    });
    el("run").disabled = true;
    sync();
    boot();
  });
}());
