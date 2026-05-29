const performanceStages = [
  {
    stage: "Initial serial",
    improvement: "순차 처리 baseline",
    fps: 3.145,
    frameSeconds: 0.320,
    fpsDelta: "-",
    timeDelta: "-",
  },
  {
    stage: "Camera parallel",
    improvement: "cam1/cam2 파이프라인 병렬화",
    fps: 5.488,
    frameSeconds: 0.183,
    fpsDelta: "+74.5%",
    timeDelta: "43.0% 감소",
  },
  {
    stage: "Model parallel",
    improvement: "pose/custom YOLO 추론 병렬화",
    fps: 6.335,
    frameSeconds: 0.158,
    fpsDelta: "+15.4%",
    timeDelta: "13.5% 감소",
  },
  {
    stage: "Input tuning",
    improvement: "custom YOLO 입력 크기 512 기준 최적화",
    fps: 7.364,
    frameSeconds: 0.136,
    fpsDelta: "+16.2%",
    timeDelta: "14.2% 감소",
  },
  {
    stage: "Pose skip/cache",
    improvement: "pose를 2프레임에 1회 추론하고 캐시 재사용",
    fps: 10.148,
    frameSeconds: 0.099,
    fpsDelta: "+37.8%",
    timeDelta: "27.0% 감소",
  },
];

function createBarChart(rootId, valueKey, suffix, invert = false) {
  const root = document.getElementById(rootId);
  if (!root) return;

  const values = performanceStages.map((item) => item[valueKey]);
  const max = Math.max(...values);

  performanceStages.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("span");
    label.textContent = item.stage;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    const ratio = invert ? (max - item[valueKey] + Math.min(...values)) / max : item[valueKey] / max;
    fill.style.width = `${Math.max(8, ratio * 100).toFixed(1)}%`;

    const value = document.createElement("strong");
    value.textContent = `${item[valueKey].toFixed(3)}${suffix}`;

    track.appendChild(fill);
    row.append(label, track, value);
    root.appendChild(row);
  });
}

function renderPerformanceTable() {
  const tbody = document.getElementById("perfTableBody");
  if (!tbody) return;

  performanceStages.forEach((item, index) => {
    const tr = document.createElement("tr");
    const cells = [
      `${index}`,
      item.improvement,
      `${item.fps.toFixed(3)} FPS`,
      item.fpsDelta,
      `${item.frameSeconds.toFixed(3)}s`,
      item.timeDelta,
    ];

    cells.forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });
}

function syncVideoPlayback() {
  const videos = Array.from(document.querySelectorAll("video"));
  videos.forEach((video) => {
    video.addEventListener("mouseenter", () => {
      if (video.paused && video.muted) {
        video.play().catch(() => {});
      }
    });
  });
}

createBarChart("fpsChart", "fps", "");
createBarChart("latencyChart", "frameSeconds", "s", true);
renderPerformanceTable();
syncVideoPlayback();
