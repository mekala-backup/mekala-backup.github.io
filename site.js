const resources = {
  code: {
    href: "https://github.com/mekala-backup/mekala-backup.github.io",
    label: "Code",
    kicker: "Source",
    note: "Implementation and evaluation scripts",
  },
  arxiv: {
    href: "https://zenodo.org/records/21369544",
    label: "zenodo",
    kicker: "Paper",
    note: "Preprint and citation page",
  },
  huggingface: {
    href: "https://huggingface.co/mekala-2402/Less_is_More",
    label: "Hugging Face",
    kicker: "Models",
    note: "Weights, demos, or model cards",
  },
  dataset: {
    href: "https://huggingface.co/datasets/mekala-2402/Less_is_More_paper_datasets",
    label: "Dataset",
    kicker: "Datasets",
    note: "All 9 ratio points, test & original",
  },
};

const ratioData = [
  {
    rank: 0,
    slug: "synthetic",
    label: "100:0",
    mix: "100:0",
    name: "Synthetic only",
    mask50: 0.6161985609107543,
    mask5095: 0.38846215261422645,
    box50: 0.593207539813076,
    box5095: 0.369649575990301,
    precision: 0.6649483227236291,
    recall: 0.6533644593057717,
    speed: 813.984428740593,
  },
  {
    rank: 1,
    slug: "95",
    label: "95:5",
    mix: "95:5",
    name: "95% synthetic",
    mask50: 0.7335579381025401,
    mask5095: 0.5208147063302195,
    box50: 0.7017449899884445,
    box5095: 0.4657321656915765,
    precision: 0.8786750653170378,
    recall: 0.7297728178557195,
    speed: 870.7312354229819,
  },
  {
    rank: 2,
    slug: "90",
    label: "90:10",
    mix: "90:10",
    name: "90% synthetic",
    mask50: 0.7866850743319694,
    mask5095: 0.5745577041321692,
    box50: 0.7550414745032024,
    box5095: 0.5143600198068321,
    precision: 0.8898875600300807,
    recall: 0.7779992028696692,
    speed: 948.6965216603534,
  },
  {
    rank: 3,
    slug: "75",
    label: "75:25",
    mix: "75:25",
    name: "75% synthetic",
    mask50: 0.6985296829361993,
    mask5095: 0.5029899211734381,
    box50: 0.6670478924880574,
    box5095: 0.45067756161361616,
    precision: 0.8960573476702509,
    recall: 0.6974890394579514,
    speed: 994.7468311194808,
  },
  {
    rank: 4,
    slug: "50",
    label: "50:50",
    mix: "50:50",
    name: "Balanced",
    mask50: 0.7677913924822569,
    mask5095: 0.5502353408024083,
    box50: 0.7327078937458718,
    box5095: 0.48552855936706163,
    precision: 0.8783845800826067,
    recall: 0.7628537265842965,
    speed: 841.7615444150643,
  },
  {
    rank: 5,
    slug: "25",
    label: "25:75",
    mix: "25:75",
    name: "25% synthetic",
    mask50: 0.6722128779656278,
    mask5095: 0.4775463480616162,
    box50: 0.645847590741701,
    box5095: 0.42528194323813995,
    precision: 0.8416378885051801,
    recall: 0.6799521721801515,
    speed: 821.8307435738852,
  },
  {
    rank: 6,
    slug: "10",
    label: "10:90",
    mix: "10:90",
    name: "10% synthetic",
    mask50: 0.6044312097393465,
    mask5095: 0.44188699105931517,
    box50: 0.5908991218403067,
    box5095: 0.39329995142055085,
    precision: 0.7812812812812813,
    recall: 0.6221602231964927,
    speed: 819.2771642720542,
  },
  {
    rank: 7,
    slug: "5",
    label: "5:95",
    mix: "5:95",
    name: "5% synthetic",
    mask50: 0.6747526523705829,
    mask5095: 0.4857194027585262,
    box50: 0.6624682329855506,
    box5095: 0.42931263114276186,
    precision: 0.8494033505717059,
    recall: 0.6699066427781076,
    speed: 825.0469052515695,
  },
  {
    rank: 8,
    slug: "real",
    label: "0:100",
    mix: "0:100",
    name: "Real only",
    mask50: 0.6879835584210998,
    mask5095: 0.495504673293075,
    box50: 0.6519653935732832,
    box5095: 0.4365469266327904,
    precision: 0.87394737332658,
    recall: 0.6771622160223196,
    speed: 821.2728128537835,
  },
];

const metricDefs = {
  mask50: {
    title: "Segmentation mAP@50",
    label: "Segmentation mAP@50",
    suffix: "",
    direction: "max",
    description: "The main segmentation score peaks at 90:10, with 50:50 as a close secondary high point.",
    hoverText: "This is the paper's primary metric. Higher is better.",
    insights: [
      "90:10 gives the strongest segmentation mAP@50.",
      "50:50 stays competitive and forms a secondary peak.",
      "The curve is non-monotonic as real data increases.",
    ],
    showBaselines: true,
  },
  mask5095: {
    title: "Segmentation mAP@50-95",
    label: "Segmentation mAP@50-95",
    suffix: "",
    direction: "max",
    description: "The stricter segmentation metric follows the same mixed-ratio pattern.",
    hoverText: "A more demanding localization score. Higher is better.",
    insights: [
      "90:10 remains the top configuration.",
      "Mixed ratios consistently beat the pure-source extremes.",
      "The gap between top mixes is smaller than the gap to the extremes.",
    ],
    showBaselines: false,
  },
  box50: {
    title: "Box mAP@50",
    label: "Box mAP@50",
    suffix: "",
    direction: "max",
    description: "Box-level performance mirrors the segmentation-level trend without changing the overall story.",
    hoverText: "Higher is better.",
    insights: [
      "90:10 is the strongest box-level run.",
      "50:50 is the closest competitor.",
      "Pure synthetic lags the mixed regimes.",
    ],
    showBaselines: false,
  },
  box5095: {
    title: "Box mAP@50-95",
    label: "Box mAP@50-95",
    suffix: "",
    direction: "max",
    description: "The stricter box metric stays aligned with the mixed-ratio sweet spot.",
    hoverText: "Higher is better.",
    insights: [
      "90:10 stays in front.",
      "The balanced mix is the next strongest point.",
      "The ratio sweep stays clearly non-linear.",
    ],
    showBaselines: false,
  },
  precision: {
    title: "Precision",
    label: "Precision",
    suffix: "",
    direction: "max",
    description: "Precision is comparatively tight across the strongest ratios, with 75:25 edging out the rest.",
    hoverText: "Higher is better.",
    insights: [
      "75:25 is the highest precision point in the table.",
      "90:10 is still within a very small band of the top value.",
      "Precision varies less sharply than the mAP scores.",
    ],
    showBaselines: false,
  },
  recall: {
    title: "Recall",
    label: "Recall",
    suffix: "",
    direction: "max",
    description: "Recall favors the mixed-ratio regime and stays strongest near 90:10 and 50:50.",
    hoverText: "Higher is better.",
    insights: [
      "90:10 is the best recall score.",
      "50:50 is a strong second-tier option.",
      "Very real-heavy ratios lose more recall.",
    ],
    showBaselines: false,
  },
  speed: {
    title: "Inference speed",
    label: "Inference speed",
    suffix: " ms/image",
    direction: "min",
    description: "Runtime is a separate tradeoff: lower is better, and synthetic-only is the fastest point.",
    hoverText: "Lower is better.",
    insights: [
      "Synthetic-only is the fastest configuration.",
      "Speed differences are smaller than the accuracy differences.",
      "The 90:10 sweet spot trades some speed for stronger segmentation.",
    ],
    showBaselines: false,
  },
};

const figureCards = [
  {
    tag: "Clutter",
    title: "Unseen background, mixed ratios",
    caption: "Synthetic-heavy runs keep multiple transparent objects visible when the scene background changes.",
    image: "./images in research paper/synth_performs_good_in_a_different_backgroung.png",
  },
  {
    tag: "Clutter",
    title: "Cluttered tabletop comparison",
    caption: "Mixed-ratio models separate the bottles more cleanly than the pure-real baseline.",
    image: "./images in research paper/50_percent_and_synth.png",
  },
  {
    tag: "Orientation",
    title: "Floor clutter and random angles",
    caption: "Even a small real anchor helps the model stay stable in non-table scenes.",
    image: "./images in research paper/5_percent_good_in_floor_clutter.png",
  },
  {
    tag: "Occlusion",
    title: "Partial occlusion at 25 percent real",
    caption: "The 25:75 model keeps overlapping bottles separate instead of merging them.",
    image: "./images in research paper/adaption_for_occlusion_in_25.png",
  },
  {
    tag: "Recall",
    title: "Secondary object recovery",
    caption: "Higher real ratios start to recover background objects that lower-real runs miss.",
    image: "./images in research paper/50_percent_catching_objects_ignored_in_background.png",
  },
  {
    tag: "Generalization",
    title: "Novel floor environment",
    caption: "The synthetic-dominant and 50:50 runs generalize to a different floor setting.",
    image: "./images in research paper/synth_followed_by_50_percent_in_new_environments.png",
  },
  {
    tag: "Background",
    title: "Different background robustness",
    caption: "A synthetic-heavy model stays confident when the background shifts away from the training look.",
    image: "./images in research paper/25_percent_good.png",
  },
];

const els = {};
let activeMetric = "mask50";
let sortState = { key: "mask50", direction: "desc" };
let activeFigureIndex = 0;
let chartResizeHandler = null;
let chartResizeRaf = null;
let chartTooltipHideTimer = null;

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  hydrateImages();
  wireResourceLinks();
  initReveal();
  initScrollSpy();
  initMetricToggle();
  initTable();
  initGallery();
  initChartResize();
  renderChart(activeMetric);
  updateMetricFocus(activeMetric);
  syncNavState();
});

function cacheElements() {
  els.navLinks = [...document.querySelectorAll("[data-nav-link]")];
  els.revealBlocks = [...document.querySelectorAll(".reveal")];
  els.resourceCards = [...document.querySelectorAll("[data-resource]")];
  els.metricButtons = [...document.querySelectorAll("[data-metric]")];
  els.chartStage = document.getElementById("chartStage");
  els.chartCard = document.querySelector(".chart-card");
  els.chartTitle = document.getElementById("chartTitle");
  els.chartDescription = document.getElementById("chartDescription");
  els.chartMode = document.getElementById("chartMode");
  els.chartBadges = document.getElementById("chartBadges");
  els.chartLegend = document.getElementById("chartLegend");
  els.chartTooltip = document.getElementById("chartTooltip");
  els.metricSummary = document.getElementById("metricSummary");
  els.chartStageWrap = document.querySelector(".chart-stage-wrap");
  els.focusRatio = document.getElementById("focusRatio");
  els.focusText = document.getElementById("focusText");
  els.focusMetrics = document.getElementById("focusMetrics");
  els.chartInsights = document.getElementById("chartInsights");
  els.metricsBody = document.getElementById("metricsBody");
  els.figureGrid = document.getElementById("figureGrid");
  els.figureModal = document.getElementById("figureModal");
  els.modalImage = document.getElementById("modalImage");
  els.modalTitle = document.getElementById("modalTitle");
  els.modalCaption = document.getElementById("modalCaption");
  els.modalTag = document.getElementById("modalTag");
}

function hydrateImages() {
  document.querySelectorAll("[data-src]").forEach((img) => {
    const raw = img.getAttribute("data-src");
    if (!raw) return;
    img.src = encodeURI(raw);
  });
}

function wireResourceLinks() {
  els.resourceCards.forEach((card) => {
    const key = card.getAttribute("data-resource");
    const resource = resources[key];
    if (!resource) return;

    card.href = resource.href;
    const kicker = card.querySelector(".resource-kicker");
    const title = card.querySelector("strong");
    const note = card.querySelector(".resource-note");
    if (kicker) kicker.textContent = resource.kicker;
    if (title) title.textContent = resource.label;
    if (note) note.textContent = resource.note;
  });
}

function initReveal() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  els.revealBlocks.forEach((node) => observer.observe(node));
}

function initScrollSpy() {
  const sections = [...document.querySelectorAll("[data-section]")];
  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!visible) return;
      const id = visible.target.id;
      els.navLinks.forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === `#${id}`);
      });
    },
    {
      rootMargin: "-25% 0px -55% 0px",
      threshold: [0.12, 0.22, 0.32, 0.5],
    }
  );

  sections.forEach((section) => observer.observe(section));
}

function syncNavState() {
  const activeSection = [...document.querySelectorAll("[data-section]")].find((section) => {
    const rect = section.getBoundingClientRect();
    return rect.top < window.innerHeight * 0.34 && rect.bottom > window.innerHeight * 0.34;
  });

  if (!activeSection) return;

  els.navLinks.forEach((link) => {
    link.classList.toggle("active", link.getAttribute("href") === `#${activeSection.id}`);
  });
}

function initMetricToggle() {
  els.metricButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const metric = button.getAttribute("data-metric");
      if (!metric || metric === activeMetric) return;
      activeMetric = metric;
      sortState = { key: metric, direction: metricDefs[metric].direction === "min" ? "asc" : "desc" };
      setMetricToggleState();
      renderChart(activeMetric);
      updateMetricFocus(activeMetric);
      renderTable();
    });
  });
}

function initChartResize() {
  if (chartResizeHandler) return;
  chartResizeHandler = () => {
    if (chartResizeRaf !== null) return;
    chartResizeRaf = window.requestAnimationFrame(() => {
      chartResizeRaf = null;
      renderChart(activeMetric);
    });
  };
  window.addEventListener("resize", chartResizeHandler, { passive: true });
}

function setMetricToggleState() {
  els.metricButtons.forEach((button) => {
    const active = button.getAttribute("data-metric") === activeMetric;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
}

function updateMetricFocus(metricKey, hoveredRow = null) {
  const def = metricDefs[metricKey];
  const rows = [...ratioData];
  const score = (row) => row[metricKey];
  const bestRow = def.direction === "min" ? rows.reduce((best, row) => (score(row) < score(best) ? row : best), rows[0]) : rows.reduce((best, row) => (score(row) > score(best) ? row : best), rows[0]);
  const secondRow = [...rows]
    .sort((a, b) => (def.direction === "min" ? score(a) - score(b) : score(b) - score(a)))
    .find((row) => row !== bestRow);
  const row = hoveredRow || bestRow;

  els.focusRatio.textContent = row.mix;
  els.focusText.textContent = hoveredRow
    ? metricKey === "speed"
      ? `${row.name} gives ${formatMetricValue(row[metricKey], metricKey)} ms/image.`
      : `${row.name} gives ${formatMetricValue(row[metricKey], metricKey)} on ${def.label}.`
    : defaultFocusText(metricKey);

  els.focusMetrics.innerHTML = "";
  const chips = [
    {
      label: "Best",
      value: `${bestRow.mix} ${formatMetricValue(bestRow[metricKey], metricKey)} ${metricSuffix(metricKey)}`.trim(),
    },
    {
      label: "Runner-up",
      value: `${secondRow.mix} ${formatMetricValue(secondRow[metricKey], metricKey)} ${metricSuffix(metricKey)}`.trim(),
    },
  ];

  if (metricKey === "mask50") {
    chips.push({
      label: "Baselines",
      value: `Pure real ${formatMetricValue(ratioData[8].mask50, metricKey)} | Pure synthetic ${formatMetricValue(ratioData[0].mask50, metricKey)}`,
    });
  } else if (metricKey === "speed") {
    chips.push({
      label: "Fastest",
      value: `${bestRow.name} ${formatMetricValue(bestRow[metricKey], metricKey)} ms/image`,
    });
  } else {
    chips.push({
      label: "Range",
      value: `${formatMetricValue(Math.max(...ratioData.map((r) => r[metricKey])) - Math.min(...ratioData.map((r) => r[metricKey])), metricKey)} ${metricSuffix(metricKey)}`.trim(),
    });
  }

  chips.forEach((chip) => {
    const el = document.createElement("div");
    el.className = "focus-chip";
    el.innerHTML = `<strong>${chip.label}</strong><span>${chip.value}</span>`;
    els.focusMetrics.appendChild(el);
  });

  els.chartInsights.innerHTML = "";
  const insights = def.insights;
  insights.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    els.chartInsights.appendChild(li);
  });

  els.chartTitle.textContent = def.title;
  els.chartDescription.textContent = def.description;
}

function defaultFocusText(metricKey) {
  const def = metricDefs[metricKey];
  if (metricKey === "speed") {
    return "Hover a bar to compare runtime against the best configuration.";
  }
  return def.hoverText;
}

function metricSuffix(metricKey) {
  return metricDefs[metricKey].suffix || "";
}

function formatMetricValue(value, metricKey) {
  if (metricKey === "speed") {
    return value.toFixed(1);
  }
  return value.toFixed(3);
}

function formatMetricValueShort(value, metricKey) {
  if (metricKey === "speed") {
    return value.toFixed(0);
  }
  return value.toFixed(3);
}

function renderChart(metricKey) {
  const def = metricDefs[metricKey];
  const isSeg = metricKey.startsWith("mask");
  const isBox = metricKey.startsWith("box");
  const palette = isSeg
    ? {
        main: "#f97316",
        mainDark: "#c2410c",
        soft: "#ffedd5",
        softDark: "#fed7aa",
        grid: "rgba(249, 115, 22, 0.12)",
        label: "#ea580c",
        text: "#9a3412",
      }
    : isBox
      ? {
          main: "#0f172a",
          mainDark: "#334155",
          soft: "#e2e8f0",
          softDark: "#cbd5e1",
          grid: "rgba(15, 23, 42, 0.12)",
          label: "#0f172a",
          text: "#0f172a",
        }
      : {
          main: "#475569",
          mainDark: "#1e293b",
          soft: "#e2e8f0",
          softDark: "#cbd5e1",
          grid: "rgba(100, 116, 139, 0.12)",
          label: "#334155",
          text: "#0f172a",
        };
  const values = ratioData.map((row) => row[metricKey]);
  const bestIndex = def.direction === "min"
    ? values.indexOf(Math.min(...values))
    : values.indexOf(Math.max(...values));

  const width = Math.max(320, els.chartStage.clientWidth || 640);
  const height = 420;
  const margin = { top: 36, right: 18, bottom: 74, left: 60 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, metricKey === "speed" ? 1 : 0.001);
  const pad = metricKey === "speed" ? 18 : span * 0.22;
  const domainMin = min - pad;
  const domainMax = max + pad;
  const y = (value) => margin.top + innerHeight - ((value - domainMin) / (domainMax - domainMin)) * innerHeight;
  const step = innerWidth / ratioData.length;
  const barWidth = Math.min(70, step * 0.64);

  if (els.chartCard) {
    els.chartCard.classList.remove("mode-seg", "mode-box", "mode-other");
    els.chartCard.classList.add(isSeg ? "mode-seg" : isBox ? "mode-box" : "mode-other");
  }
  if (els.chartMode) {
    els.chartMode.textContent = isSeg ? "Segmentation view" : isBox ? "Box view" : "Metric view";
  }

  const gridLines = 5;
  let svg = `
      <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${def.title} chart">
      <defs>
        <linearGradient id="barFill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="${palette.main}"></stop>
          <stop offset="100%" stop-color="${palette.mainDark}"></stop>
        </linearGradient>
        <linearGradient id="barFillMuted" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="${palette.soft}"></stop>
          <stop offset="100%" stop-color="${palette.softDark}"></stop>
        </linearGradient>
      </defs>
  `;

  for (let i = 0; i < gridLines; i += 1) {
    const value = domainMin + ((domainMax - domainMin) / (gridLines - 1)) * i;
    const yy = y(value);
    svg += `
      <line x1="${margin.left}" y1="${yy}" x2="${width - margin.right}" y2="${yy}" stroke="${palette.grid}" stroke-dasharray="4 6"></line>
      <text x="${margin.left - 10}" y="${yy + 4}" text-anchor="end" fill="${palette.label}" font-size="12">${formatChartAxisValue(value, metricKey)}</text>
    `;
  }

  

  ratioData.forEach((row, index) => {
    const value = row[metricKey];
    const barTop = y(value);
    const barHeight = margin.top + innerHeight - barTop;
    const xCenter = margin.left + step * index + step / 2;
    const x = xCenter - barWidth / 2;
    const isBest = index === bestIndex;
    const fill = isBest ? "url(#barFill)" : "url(#barFillMuted)";
    const stroke = isBest ? palette.mainDark : palette.grid;
    const labelY = height - 26;
if (def.showBaselines) {
    const pureReal = ratioData[8][metricKey];
    const pureSynth = ratioData[0][metricKey];
    const realY = y(pureReal);
    const synthY = y(pureSynth);
    svg += `
      <line x1="${margin.left}" y1="${realY}" x2="${width - margin.right}" y2="${realY}" stroke="rgba(249, 115, 22, 0.8)" stroke-dasharray="8 6" stroke-width="1.5"></line>
      <line x1="${margin.left}" y1="${synthY}" x2="${width - margin.right}" y2="${synthY}" stroke="rgba(15, 23, 42, 0.34)" stroke-dasharray="8 6" stroke-width="1.5"></line>
      <rect x="${margin.left}" y="${realY - 22}" width="150" height="18" rx="9" fill="rgba(255,255,255,0.92)"></rect>
      <text x="${margin.left + 8}" y="${realY - 8}" text-anchor="start" fill="${palette.text}" font-size="12" font-weight="700">Pure real ${formatMetricValue(pureReal, metricKey)}</text>
      <rect x="${margin.left}" y="${synthY - 22}" width="170" height="18" rx="9" fill="rgba(255,255,255,0.92)"></rect>
      <text x="${margin.left + 8}" y="${synthY - 8}" text-anchor="start" fill="#64748b" font-size="12" font-weight="700">Pure synthetic ${formatMetricValue(pureSynth, metricKey)}</text>
    `;
  }
    svg += `
      <g class="bar-group" data-index="${index}" tabindex="0" role="button" focusable="true" aria-label="${row.label}, ${def.title}, ${formatMetricValue(value, metricKey)}${metricSuffix(metricKey)}">
        <rect x="${x}" y="${barTop}" width="${barWidth}" height="${barHeight}" rx="16" fill="${fill}" stroke="${stroke}" stroke-width="1"></rect>
        <text x="${xCenter}" y="${labelY}" text-anchor="middle" fill="#64748b" font-size="12">${row.label}</text>
      </g>
    `;
  });

  svg += `
      <text x="${margin.left}" y="${height - 8}" fill="#64748b" font-size="12">Synthetic -> Real ratio</text>
      <text x="${width - margin.right}" y="${22}" text-anchor="end" fill="${palette.label}" font-size="12">${def.label}${def.suffix ? ` (${def.suffix.trim()})` : ""}</text>
    </svg>
  `;

  els.chartStage.innerHTML = svg;

  const svgEl = els.chartStage.querySelector("svg");
  svgEl.querySelectorAll(".bar-group").forEach((group) => {
    const index = Number(group.getAttribute("data-index"));
    const row = ratioData[index];
    const show = () => {
      updateMetricFocus(metricKey, row);
      showChartTooltip(metricKey, row, group);
    };
    const hide = () => {
      updateMetricFocus(metricKey);
      hideChartTooltip();
    };
    group.addEventListener("mouseenter", show);
    group.addEventListener("mouseleave", hide);
    group.addEventListener("focusin", show);
    group.addEventListener("focusout", hide);
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        show();
      }
    });
  });

  els.chartLegend.innerHTML = "";
  const legendItems = [
    { label: "Best bar", active: true },
    { label: "Other mixes", active: false },
  ];
  if (def.showBaselines) {
    legendItems.push({ label: "Pure real reference", active: false });
    legendItems.push({ label: "Pure synthetic reference", active: false });
  }
  legendItems.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "legend-item";
    chip.innerHTML = `<span class="legend-swatch ${item.active ? "active" : ""}"></span><span>${item.label}</span>`;
    els.chartLegend.appendChild(chip);
  });

  els.chartBadges.innerHTML = "";
  const badgeTexts = [
    `Best: ${ratioData[bestIndex].mix} ${formatMetricValue(ratioData[bestIndex][metricKey], metricKey)}${metricSuffix(metricKey)}`,
    metricKey === "speed" ? "Lower is better" : "Higher is better",
  ];
  if (def.showBaselines) {
    badgeTexts.push("Dashed lines = pure-source baselines");
  }

  badgeTexts.forEach((text) => {
    const badge = document.createElement("span");
    badge.className = "pill";
    badge.textContent = text;
    els.chartBadges.appendChild(badge);
  });

  renderMetricSummary(metricKey, bestIndex);
  updateMetricFocus(metricKey);
}

function showChartTooltip(metricKey, row, targetEl) {
  if (!els.chartTooltip || !els.chartStageWrap) return;
  clearTimeout(chartTooltipHideTimer);
  chartTooltipHideTimer = null;

  const value = formatMetricValue(row[metricKey], metricKey);
  const unit = metricSuffix(metricKey).trim();
  const detail = unit ? `${value}${unit}` : value;
  const category = metricDefs[metricKey].title;

  els.chartTooltip.innerHTML = `
    <strong>${row.label}</strong>
    <span>${row.name}</span>
    <em>${detail}</em>
    <small>${category}</small>
  `;

  els.chartTooltip.classList.add("visible");
  els.chartTooltip.setAttribute("aria-hidden", "false");

  const wrapRect = els.chartStageWrap.getBoundingClientRect();
  const targetRect = targetEl.getBoundingClientRect();

  els.chartTooltip.style.visibility = "hidden";
  els.chartTooltip.style.left = "12px";
  els.chartTooltip.style.top = "12px";

  window.requestAnimationFrame(() => {
    const tooltipRect = els.chartTooltip.getBoundingClientRect();
    const centerX = targetRect.left - wrapRect.left + targetRect.width / 2;
    const left = clamp(centerX - tooltipRect.width / 2, 12, Math.max(12, wrapRect.width - tooltipRect.width - 12));
    const spaceAbove = targetRect.top - wrapRect.top;
    const spaceBelow = wrapRect.bottom - targetRect.bottom;
    let top;

    if (spaceAbove >= tooltipRect.height + 12) {
      top = targetRect.top - wrapRect.top - tooltipRect.height - 12;
    } else if (spaceBelow >= tooltipRect.height + 12) {
      top = targetRect.bottom - wrapRect.top + 12;
    } else {
      top = clamp(
        targetRect.top - wrapRect.top - tooltipRect.height - 12,
        12,
        Math.max(12, wrapRect.height - tooltipRect.height - 12)
      );
    }

    els.chartTooltip.style.left = `${left}px`;
    els.chartTooltip.style.top = `${top}px`;
    els.chartTooltip.style.visibility = "visible";
  });
}

function hideChartTooltip() {
  if (!els.chartTooltip) return;
  clearTimeout(chartTooltipHideTimer);
  chartTooltipHideTimer = window.setTimeout(() => {
    els.chartTooltip.classList.remove("visible");
    els.chartTooltip.setAttribute("aria-hidden", "true");
    chartTooltipHideTimer = null;
  }, 60);
}

function renderMetricSummary(metricKey, bestIndex) {
  const def = metricDefs[metricKey];
  const bestRow = ratioData[bestIndex];
  const pureReal = ratioData[8];
  const pureSynth = ratioData[0];
  const values = ratioData.map((row) => row[metricKey]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min;

  const liftReal = metricLift(bestRow[metricKey], pureReal[metricKey], metricKey);
  const liftSynth = metricLift(bestRow[metricKey], pureSynth[metricKey], metricKey);

  const cards = [
    {
      label: "Best mix",
      value: `${bestRow.mix} · ${formatMetricValue(bestRow[metricKey], metricKey)}${metricSuffix(metricKey)}`,
      note: `Top point for ${def.title.toLowerCase()}.`,
      width: 100,
    },
    {
      label: "Lift vs pure real",
      value: liftReal.text,
      note: liftReal.note,
      width: liftReal.percent,
    },
    {
      label: "Lift vs pure synthetic",
      value: liftSynth.text,
      note: liftSynth.note,
      width: liftSynth.percent,
    },
    {
      label: "Sweep spread",
      value: metricKey === "speed" ? `${spread.toFixed(1)} ms/image` : `${formatMetricValue(spread, metricKey)}`,
      note: "Range between the best and worst points across the sweep.",
      width: max === 0 ? 0 : (spread / max) * 100,
      accent: true,
    },
  ];

  els.metricSummary.innerHTML = cards
    .map(
      (card) => `
        <article class="summary-card ${card.accent ? "summary-card-accent" : ""}">
          <span class="summary-label">${card.label}</span>
          <strong>${card.value}</strong>
          <p>${card.note}</p>
          <div class="summary-bar" aria-hidden="true"><span style="width:${clamp(card.width, 0, 100)}%"></span></div>
        </article>
      `
    )
    .join("");

}

function metricLift(best, baseline, metricKey) {
  if (metricKey === "speed") {
    const delta = baseline - best;
    const percent = baseline === 0 ? 0 : (delta / baseline) * 100;
    return {
      text: delta >= 0 ? `${delta.toFixed(1)} ms faster` : `${Math.abs(delta).toFixed(1)} ms slower`,
      note: percent >= 0 ? `${percent.toFixed(1)}% faster` : `${Math.abs(percent).toFixed(1)}% slower`,
      percent: Math.abs(percent),
    };
  }

  const delta = best - baseline;
  const percent = baseline === 0 ? 0 : (delta / baseline) * 100;
  return {
    text: `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`,
    note: `${Math.abs(percent).toFixed(1)}% ${delta >= 0 ? "better" : "worse"}`,
    percent: Math.abs(percent),
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatChartAxisValue(value, metricKey) {
  if (metricKey === "speed") {
    return value.toFixed(0);
  }
  return value.toFixed(3);
}

function initTable() {
  const headers = document.querySelectorAll(".metrics-table thead th");
  headers.forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.getAttribute("data-sort-key");
      if (!key) return;
      if (sortState.key === key) {
        sortState.direction = sortState.direction === "asc" ? "desc" : "asc";
      } else {
        sortState.key = key;
        sortState.direction = key === "speed" ? "asc" : "desc";
      }
      renderTable();
      updateHeaderIndicators();
    });
  });

  renderTable();
  updateHeaderIndicators();
}

function renderTable() {
  const metricKey = sortState.key;
  const direction = sortState.direction;
  const rows = [...ratioData].sort((a, b) => {
    if (metricKey === "rank") {
      return direction === "asc" ? a.rank - b.rank : b.rank - a.rank;
    }
    const av = a[metricKey];
    const bv = b[metricKey];
    if (av === bv) return 0;
    return direction === "asc" ? av - bv : bv - av;
  });

  const bestRow = metricDefs[activeMetric].direction === "min"
    ? [...ratioData].reduce((best, row) => (row[activeMetric] < best[activeMetric] ? row : best), ratioData[0])
    : [...ratioData].reduce((best, row) => (row[activeMetric] > best[activeMetric] ? row : best), ratioData[0]);

  els.metricsBody.innerHTML = rows
    .map((row) => {
      const isBest = row.slug === bestRow.slug;
      return `
        <tr class="${isBest ? "is-best" : ""}">
          <td><strong>${row.label}</strong><div class="muted">${row.mix}</div></td>
          <td>${formatMetricValue(row.mask50, "mask50")}</td>
          <td>${formatMetricValue(row.mask5095, "mask5095")}</td>
          <td>${formatMetricValue(row.box50, "box50")}</td>
          <td>${formatMetricValue(row.box5095, "box5095")}</td>
          <td>${formatMetricValue(row.precision, "precision")}</td>
          <td>${formatMetricValue(row.recall, "recall")}</td>
          <td>${formatMetricValue(row.speed, "speed")}</td>
        </tr>
      `;
    })
    .join("");
}

function updateHeaderIndicators() {
  document.querySelectorAll(".metrics-table thead th").forEach((th) => {
    const key = th.getAttribute("data-sort-key");
    if (!key) return;
    th.querySelector(".sort-indicator")?.remove();
    if (sortState.key === key) {
      const indicator = document.createElement("span");
      indicator.className = "sort-indicator";
      indicator.textContent = sortState.direction === "asc" ? "↑" : "↓";
      th.appendChild(indicator);
    }
  });
}

function initGallery() {
  els.figureGrid.innerHTML = figureCards
    .map((figure, index) => `
      <button type="button" class="figure-card" data-figure-index="${index}">
        <img data-src="${figure.image}" alt="${figure.title}" loading="lazy">
        <span class="figure-tag">${figure.tag}</span>
        <h3>${figure.title}</h3>
        <p>${figure.caption}</p>
      </button>
    `)
    .join("");

  hydrateImages();

  els.figureGrid.querySelectorAll("[data-figure-index]").forEach((card) => {
    card.addEventListener("click", () => {
      const index = Number(card.getAttribute("data-figure-index"));
      openFigure(index);
    });
  });

  els.figureModal.addEventListener("click", (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && target.getAttribute("data-close") === "true") {
      closeFigure();
    }
    if (target instanceof HTMLElement && target.getAttribute("data-step")) {
      const step = Number(target.getAttribute("data-step"));
      openFigure(activeFigureIndex + step);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (!els.figureModal.classList.contains("is-open")) return;
    if (event.key === "Escape") closeFigure();
    if (event.key === "ArrowLeft") openFigure(activeFigureIndex - 1);
    if (event.key === "ArrowRight") openFigure(activeFigureIndex + 1);
  });
}

function openFigure(index) {
  activeFigureIndex = (index + figureCards.length) % figureCards.length;
  const figure = figureCards[activeFigureIndex];
  els.modalImage.src = encodeURI(figure.image);
  els.modalImage.alt = figure.title;
  els.modalTitle.textContent = figure.title;
  els.modalCaption.textContent = figure.caption;
  els.modalTag.textContent = figure.tag;
  els.figureModal.classList.add("is-open");
  els.figureModal.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
}

function closeFigure() {
  els.figureModal.classList.remove("is-open");
  els.figureModal.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
}
