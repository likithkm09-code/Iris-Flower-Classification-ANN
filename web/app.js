/**
 * Iris Classifier Studio - Client Side Reactivity & Deep Learning Visualizer
 */

document.addEventListener("DOMContentLoaded", () => {
  // Dual-control inputs
  const slRange = document.getElementById("sl-range");
  const swRange = document.getElementById("sw-range");
  const plRange = document.getElementById("pl-range");
  const pwRange = document.getElementById("pw-range");

  const slNum = document.getElementById("sl-num");
  const swNum = document.getElementById("sw-num");
  const plNum = document.getElementById("pl-num");
  const pwNum = document.getElementById("pw-num");

  const normTags = document.getElementById("norm-tags");
  const predSpeciesName = document.getElementById("pred-species-name");
  const predAlias = document.getElementById("pred-alias");
  const predBotanicalDesc = document.getElementById("pred-botanical-desc");
  const predConfidence = document.getElementById("pred-confidence");
  const confBadge = document.getElementById("conf-badge");
  const predFlowerImg = document.getElementById("pred-flower-img");
  const predictionStatus = document.getElementById("prediction-status");
  const flowerImageInput = document.getElementById("flower-image-input");
  const flowerUploadZone = document.getElementById("flower-upload-zone");
  const uploadedPreview = document.getElementById("uploaded-preview");
  const uploadedImage = document.getElementById("uploaded-image");
  const uploadedFileName = document.getElementById("uploaded-file-name");
  const imageStatus = document.getElementById("image-status");
  const clearUploadButton = document.getElementById("btn-clear-upload");
  const imageAnalysisResult = document.getElementById("image-analysis-result");
  const imageSpecies = document.getElementById("image-species");
  const imageConfidence = document.getElementById("image-confidence");
  const imageLength = document.getElementById("image-length");
  const imageWidth = document.getElementById("image-width");

  const probValSetosa = document.getElementById("prob-val-setosa");
  const probBarSetosa = document.getElementById("prob-bar-setosa");
  const probValVersicolor = document.getElementById("prob-val-versicolor");
  const probBarVersicolor = document.getElementById("prob-bar-versicolor");
  const probValVirginica = document.getElementById("prob-val-virginica");
  const probBarVirginica = document.getElementById("prob-bar-virginica");

  const rowSetosa = document.getElementById("row-setosa");
  const rowVersicolor = document.getElementById("row-versicolor");
  const rowVirginica = document.getElementById("row-virginica");

  // Botanical species metadata
  const SPECIES_DETAILS = {
    "Iris setosa": {
      alias: "Arctic / Bristle-Pointed Iris",
      img: "/assets/setosa.jpg",
      color: "#38bdf8",
      desc: "Distinguished by small, delicate petals (1.0-1.9 cm) and wide sepals with a bright yellow signal patch. Extremely cold-hardy, native to subarctic wetlands."
    },
    "Iris versicolor": {
      alias: "Blue Flag / Harlequin Blueflag",
      img: "/assets/versicolor.jpg",
      color: "#10b981",
      desc: "Displays intermediate petal lengths (3.0-5.1 cm) with distinct violet-blue reticulated veining. Indigenous to sedge meadows and eastern North American marshes."
    },
    "Iris virginica": {
      alias: "Southern / Giant Blue Flag",
      img: "/assets/virginica.jpg",
      color: "#f43f5e",
      desc: "Characterized by elongated, robust petals (4.5-6.9 cm) and dark violet-blue falls with vibrant golden crests. Typically the largest of the three species."
    }
  };

  let modelMetadata = null;
  let debounceTimer = null;
  let uploadedImageUrl = null;

  const referenceSpecies = [
    { name: "Iris setosa", shortName: "setosa", index: 0, image: "/assets/setosa.jpg" },
    { name: "Iris versicolor", shortName: "versicolor", index: 1, image: "/assets/versicolor.jpg" },
    { name: "Iris virginica", shortName: "virginica", index: 2, image: "/assets/virginica.jpg" }
  ];

  // Initialize Network Graph
  initNetworkGraph();

  // Load Model Metadata
  fetch("/api/model_info")
    .then(r => r.json())
    .then(data => {
      modelMetadata = data;
      if (data.metrics && data.metrics.accuracy) {
        const accEl = document.getElementById("nav-acc");
        if (accEl) accEl.textContent = (data.metrics.accuracy * 100).toFixed(1) + "%";
      }
      runPrediction();
    })
    .catch(() => runPrediction());

  // Synchronize Slider & Numeric Input
  function bindDualInput(rangeEl, numEl) {
    rangeEl.addEventListener("input", () => {
      numEl.value = parseFloat(rangeEl.value).toFixed(1);
      onInputChanged();
    });

    numEl.addEventListener("input", () => {
      let val = parseFloat(numEl.value);
      if (!isNaN(val)) {
        rangeEl.value = Math.min(Number(rangeEl.max), Math.max(Number(rangeEl.min), val));
        onInputChanged();
      }
    });
  }

  bindDualInput(slRange, slNum);
  bindDualInput(swRange, swNum);
  bindDualInput(plRange, plNum);
  bindDualInput(pwRange, pwNum);

  // Image comparison is intentionally separate from the measurement-trained ANN.
  if (flowerImageInput) {
    flowerImageInput.addEventListener("change", () => {
      if (flowerImageInput.files[0]) classifyUploadedImage(flowerImageInput.files[0]);
    });
  }

  if (flowerUploadZone) {
    ["dragenter", "dragover"].forEach(eventName => flowerUploadZone.addEventListener(eventName, event => {
      event.preventDefault();
      flowerUploadZone.classList.add("dragging");
    }));
    ["dragleave", "drop"].forEach(eventName => flowerUploadZone.addEventListener(eventName, event => {
      event.preventDefault();
      flowerUploadZone.classList.remove("dragging");
    }));
    flowerUploadZone.addEventListener("drop", event => {
      const file = event.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) classifyUploadedImage(file);
    });
  }

  if (clearUploadButton) clearUploadButton.addEventListener("click", clearUploadedImage);

  function onInputChanged() {
    updateStandardizationTags();
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runPrediction, 40);
  }

  function updateStandardizationTags() {
    if (modelMetadata && modelMetadata.mean && modelMetadata.std) {
      const vals = [
        parseFloat(slNum.value),
        parseFloat(swNum.value),
        parseFloat(plNum.value),
        parseFloat(pwNum.value)
      ];
      if (vals.some(Number.isNaN)) return;
      const zScores = vals.map((v, i) => (v - modelMetadata.mean[i]) / modelMetadata.std[i]);
      normTags.innerHTML = zScores.map((z, idx) => {
        const sign = z >= 0 ? "+" : "";
        return `<span class="norm-tag">Z<sub>${idx+1}</sub>: ${sign}${z.toFixed(2)}</span>`;
      }).join("");
    }
  }

  // Anatomy Guide Drawer Toggle
  const btnToggleGuide = document.getElementById("btn-toggle-guide");
  const anatomyDrawer = document.getElementById("anatomy-drawer");
  if (btnToggleGuide && anatomyDrawer) {
    btnToggleGuide.addEventListener("click", () => {
      anatomyDrawer.classList.toggle("open");
      btnToggleGuide.textContent = anatomyDrawer.classList.contains("open") ? "✖ Close Guide" : "📖 Anatomy Guide";
    });
  }

  // Species Quick Picker Cards
  document.querySelectorAll(".species-card").forEach(card => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".species-card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");

      slRange.value = card.dataset.sl; slNum.value = card.dataset.sl;
      swRange.value = card.dataset.sw; swNum.value = card.dataset.sw;
      plRange.value = card.dataset.pl; plNum.value = card.dataset.pl;
      pwRange.value = card.dataset.pw; pwNum.value = card.dataset.pw;

      onInputChanged();
    });
  });

  // Random sample button
  const btnRandom = document.getElementById("btn-random-sample");
  if (btnRandom) {
    btnRandom.addEventListener("click", () => {
      document.querySelectorAll(".species-card").forEach(c => c.classList.remove("active"));
      const r_sl = (4.3 + Math.random() * 3.4).toFixed(1);
      const r_sw = (2.0 + Math.random() * 2.2).toFixed(1);
      const r_pl = (1.0 + Math.random() * 5.8).toFixed(1);
      const r_pw = (0.1 + Math.random() * 2.4).toFixed(1);

      slRange.value = r_sl; slNum.value = r_sl;
      swRange.value = r_sw; swNum.value = r_sw;
      plRange.value = r_pl; plNum.value = r_pl;
      pwRange.value = r_pw; pwNum.value = r_pw;

      onInputChanged();
    });
  }

  // Reset button
  const btnReset = document.getElementById("btn-reset-values");
  if (btnReset) {
    btnReset.addEventListener("click", () => {
      document.getElementById("card-setosa").click();
    });
  }

  // Run Prediction via Backend REST API
  function runPrediction() {
    const values = [slNum, swNum, plNum, pwNum].map(input => parseFloat(input.value));
    if (values.some(value => !Number.isFinite(value))) {
      setPredictionStatus("Enter four numeric measurements to classify the sample.", true);
      return;
    }

    setPredictionStatus("Running ANN inference...", false);
    const payload = {
      sepal_length: values[0],
      sepal_width: values[1],
      petal_length: values[2],
      petal_width: values[3]
    };

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(res => res.json())
      .then(data => {
        if (data.error) throw new Error(data.error);
        renderPrediction(data);
        animateNetworkFlow(data);
        setPredictionStatus(`Latest sample classified as ${data.short_name} (${(data.confidence * 100).toFixed(1)}% confidence).`, false);
      })
      .catch(err => {
        console.error("Inference Error:", err);
        setPredictionStatus(`Inference unavailable: ${err.message}`, true);
      });
  }

  function setPredictionStatus(message, isError) {
    if (!predictionStatus) return;
    predictionStatus.textContent = message;
    predictionStatus.classList.toggle("error", isError);
  }

  async function classifyUploadedImage(file) {
    if (file.size > 10 * 1024 * 1024) {
      setImageStatus("Please choose an image smaller than 10 MB.", true);
      return;
    }

    if (uploadedImageUrl) URL.revokeObjectURL(uploadedImageUrl);
    uploadedImageUrl = URL.createObjectURL(file);
    uploadedImage.src = uploadedImageUrl;
    uploadedFileName.textContent = file.name;
    uploadedPreview.hidden = false;
    setImageStatus("Comparing the upload with the reference species...", false);

    try {
      const uploadedBitmap = await loadImage(uploadedImageUrl);
      const uploadedSignature = await imageSignature(uploadedBitmap);
      const dimensions = estimateFlowerBounds(uploadedBitmap);
      const referenceSignatures = await Promise.all(referenceSpecies.map(async species => ({
        species,
        signature: await imageSignature(await loadImage(species.image))
      })));
      const scores = referenceSignatures.map(item => 1 / (1 + signatureDistance(uploadedSignature, item.signature)));
      const total = scores.reduce((sum, score) => sum + score, 0);
      const probabilities = scores.map(score => score / total);
      const winnerIndex = probabilities.indexOf(Math.max(...probabilities));
      const winner = referenceSpecies[winnerIndex];
      const visualResult = {
        predicted_index: winner.index,
        predicted_species: winner.name,
        short_name: winner.shortName,
        confidence: probabilities[winnerIndex],
        probabilities: Object.fromEntries(referenceSpecies.map((species, index) => [species.name, probabilities[index]]))
      };
      renderPrediction(visualResult);
      animateNetworkFlow(visualResult);
      imageSpecies.textContent = winner.name;
      imageConfidence.textContent = `${(probabilities[winnerIndex] * 100).toFixed(1)}% visual match`;
      imageLength.textContent = `${dimensions.length} px`;
      imageWidth.textContent = `${dimensions.width} px`;
      imageAnalysisResult.hidden = false;
      setImageStatus(`Image classified as likely ${winner.name}.`, false);
    } catch (error) {
      console.error("Image comparison error:", error);
      setImageStatus("This image could not be analyzed. Try another JPG or PNG.", true);
    }
  }

  function loadImage(source) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = source;
    });
  }

  function imageSignature(image) {
    return new Promise(resolve => {
      const canvas = document.createElement("canvas");
      canvas.width = 32;
      canvas.height = 32;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(image, 0, 0, 32, 32);
      const pixels = context.getImageData(0, 0, 32, 32).data;
      const histogram = new Array(24).fill(0);
      for (let index = 0; index < pixels.length; index += 4) {
        histogram[Math.floor(pixels[index] / 256 * 8)]++;
        histogram[8 + Math.floor(pixels[index + 1] / 256 * 8)]++;
        histogram[16 + Math.floor(pixels[index + 2] / 256 * 8)]++;
      }
      const total = 32 * 32;
      resolve(histogram.map(value => value / total));
    });
  }

  function signatureDistance(first, second) {
    return Math.sqrt(first.reduce((sum, value, index) => sum + (value - second[index]) ** 2, 0));
  }

  function estimateFlowerBounds(image) {
    const canvas = document.createElement("canvas");
    const sampleSize = 128;
    canvas.width = sampleSize;
    canvas.height = sampleSize;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(image, 0, 0, sampleSize, sampleSize);
    const pixels = context.getImageData(0, 0, sampleSize, sampleSize).data;
    const corners = [0, (sampleSize - 1) * 4, (sampleSize * (sampleSize - 1)) * 4, (sampleSize * sampleSize - 1) * 4];
    const background = corners.reduce((average, offset) => {
      average[0] += pixels[offset];
      average[1] += pixels[offset + 1];
      average[2] += pixels[offset + 2];
      return average;
    }, [0, 0, 0]).map(value => value / corners.length);
    let minX = sampleSize;
    let minY = sampleSize;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < sampleSize; y++) {
      for (let x = 0; x < sampleSize; x++) {
        const offset = (y * sampleSize + x) * 4;
        const distance = Math.sqrt(
          (pixels[offset] - background[0]) ** 2 +
          (pixels[offset + 1] - background[1]) ** 2 +
          (pixels[offset + 2] - background[2]) ** 2
        );
        const maxChannel = Math.max(pixels[offset], pixels[offset + 1], pixels[offset + 2]);
        const minChannel = Math.min(pixels[offset], pixels[offset + 1], pixels[offset + 2]);
        if (distance > 32 || maxChannel - minChannel > 35) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }

    if (maxX < 0) return { width: image.naturalWidth, length: image.naturalHeight };
    return {
      width: Math.max(1, Math.round((maxX - minX + 1) * image.naturalWidth / sampleSize)),
      length: Math.max(1, Math.round((maxY - minY + 1) * image.naturalHeight / sampleSize))
    };
  }

  function setImageStatus(message, isError) {
    if (!imageStatus) return;
    imageStatus.textContent = message;
    imageStatus.style.color = isError ? "#fb7185" : "";
  }

  function clearUploadedImage() {
    if (uploadedImageUrl) URL.revokeObjectURL(uploadedImageUrl);
    uploadedImageUrl = null;
    flowerImageInput.value = "";
    uploadedPreview.hidden = true;
    imageAnalysisResult.hidden = true;
    setImageStatus("", false);
  }

  function renderPrediction(data) {
    const species = data.predicted_species;
    const conf = (data.confidence * 100).toFixed(1);
    const details = SPECIES_DETAILS[species] || SPECIES_DETAILS["Iris setosa"];

    predSpeciesName.textContent = species;
    predAlias.textContent = details.alias;
    predBotanicalDesc.textContent = details.desc;
    predConfidence.textContent = conf + "%";

    // Switch image smoothly
    if (predFlowerImg && predFlowerImg.src !== details.img) {
      predFlowerImg.style.opacity = "0.2";
      predFlowerImg.style.transform = "scale(0.96)";
      setTimeout(() => {
        predFlowerImg.src = details.img;
        predFlowerImg.style.opacity = "1.0";
        predFlowerImg.style.transform = "scale(1.0)";
      }, 150);
    }

    // Update active species card
    document.querySelectorAll(".species-card").forEach(c => {
      c.classList.toggle("active", c.querySelector("h3").textContent.trim() === species);
    });

    // Update confidence badge styling
    confBadge.style.borderColor = details.color;
    predConfidence.style.color = details.color;

    // Probabilities
    const probs = data.probabilities;
    const pSetosa = ((probs["Iris setosa"] || 0) * 100).toFixed(1);
    const pVersicolor = ((probs["Iris versicolor"] || 0) * 100).toFixed(1);
    const pVirginica = ((probs["Iris virginica"] || 0) * 100).toFixed(1);

    probValSetosa.textContent = pSetosa + "%";
    probBarSetosa.style.width = pSetosa + "%";

    probValVersicolor.textContent = pVersicolor + "%";
    probBarVersicolor.style.width = pVersicolor + "%";

    probValVirginica.textContent = pVirginica + "%";
    probBarVirginica.style.width = pVirginica + "%";

    rowSetosa.classList.toggle("active-winner", species === "Iris setosa");
    rowVersicolor.classList.toggle("active-winner", species === "Iris versicolor");
    rowVirginica.classList.toggle("active-winner", species === "Iris virginica");
  }

  // Tabs switching
  document.querySelectorAll(".tab-btn").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.add("active");
    });
  });

  // Modal zoom for plots
  const modal = document.getElementById("image-modal");
  const modalImg = document.getElementById("modal-img");
  const modalCap = document.getElementById("modal-caption");
  const modalClose = document.getElementById("modal-close");

  document.querySelectorAll(".zoomable-img").forEach(img => {
    img.addEventListener("click", () => {
      modalImg.src = img.src;
      modalCap.textContent = img.alt;
      modal.classList.add("open");
    });
  });

  if (modalClose) {
    modalClose.addEventListener("click", () => modal.classList.remove("open"));
  }
  if (modal) {
    modal.addEventListener("click", e => {
      if (e.target === modal) modal.classList.remove("open");
    });
  }

  // Dynamic SVG Neural Network Architecture Visualizer
  function initNetworkGraph() {
    const svg = document.getElementById("ann-svg");
    if (!svg) return;

    const layersConfig = [4, 16, 8, 3];
    const layerNames = ["Input (4)", "Dense 1: 16 (ReLU)", "Dense 2: 8 (ReLU)", "Output: 3 (Softmax)"];
    const width = 760;
    const height = 310;
    const paddingX = 85;
    const stepX = (width - 2 * paddingX) / (layersConfig.length - 1);

    svg.innerHTML = "";

    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.innerHTML = `
      <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="4.0" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    `;
    svg.appendChild(defs);

    const layerPositions = [];

    layersConfig.forEach((count, lIdx) => {
      const x = paddingX + lIdx * stepX;
      const positions = [];
      const usableHeight = height - 70;
      const stepY = usableHeight / (count + 1);

      for (let n = 0; n < count; n++) {
        const y = 35 + (n + 1) * stepY;
        positions.push({ x, y, layer: lIdx, nodeIndex: n });
      }
      layerPositions.push(positions);
    });

    // Links Layer
    const linksGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    linksGroup.setAttribute("class", "links-layer");

    for (let l = 0; l < layerPositions.length - 1; l++) {
      const curLayer = layerPositions[l];
      const nextLayer = layerPositions[l + 1];

      curLayer.forEach(src => {
        nextLayer.forEach(dst => {
          const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
          line.setAttribute("x1", src.x);
          line.setAttribute("y1", src.y);
          line.setAttribute("x2", dst.x);
          line.setAttribute("y2", dst.y);
          line.setAttribute("stroke", "rgba(255, 255, 255, 0.05)");
          line.setAttribute("stroke-width", "0.75");
          line.setAttribute("class", `link l-${l}-${l+1}`);
          linksGroup.appendChild(line);
        });
      });
    }
    svg.appendChild(linksGroup);

    // Layer Headers
    const labelsGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    layerPositions.forEach((layerNodes, lIdx) => {
      const x = layerNodes[0].x;
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", x);
      text.setAttribute("y", 18);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("fill", "#94a3b8");
      text.setAttribute("font-size", "10");
      text.setAttribute("font-weight", "600");
      text.textContent = layerNames[lIdx];
      labelsGroup.appendChild(text);
    });
    svg.appendChild(labelsGroup);

    // Nodes Layer
    const nodesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    nodesGroup.setAttribute("class", "nodes-layer");

    const inputLabels = ["SL", "SW", "PL", "PW"];
    const outputLabels = ["Setosa", "Versicolor", "Virginica"];

    layerPositions.forEach((layerNodes, lIdx) => {
      layerNodes.forEach((node, nIdx) => {
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("class", `node-group node-l${lIdx}-n${nIdx}`);

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", node.x);
        circle.setAttribute("cy", node.y);
        const radius = lIdx === 0 ? 12 : (lIdx === 3 ? 14 : 7);
        circle.setAttribute("r", radius);

        let fill = "#38bdf8";
        if (lIdx === 1) fill = "#818cf8";
        if (lIdx === 2) fill = "#c084fc";
        if (lIdx === 3) fill = "#34d399";

        circle.setAttribute("fill", fill);
        circle.setAttribute("fill-opacity", "0.85");
        circle.setAttribute("stroke", "#ffffff");
        circle.setAttribute("stroke-width", "1.2");
        circle.setAttribute("id", `node-circle-${lIdx}-${nIdx}`);
        g.appendChild(circle);

        if (lIdx === 0) {
          const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
          t.setAttribute("x", node.x - 22);
          t.setAttribute("y", node.y + 4);
          t.setAttribute("text-anchor", "end");
          t.setAttribute("fill", "#cbd5e1");
          t.setAttribute("font-size", "9");
          t.setAttribute("font-family", "var(--font-mono)");
          t.textContent = inputLabels[nIdx];
          g.appendChild(t);
        } else if (lIdx === 3) {
          const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
          t.setAttribute("x", node.x + 22);
          t.setAttribute("y", node.y + 4);
          t.setAttribute("text-anchor", "start");
          t.setAttribute("fill", "#cbd5e1");
          t.setAttribute("font-size", "10");
          t.setAttribute("font-weight", "600");
          t.textContent = outputLabels[nIdx];
          g.appendChild(t);
        }

        nodesGroup.appendChild(g);
      });
    });
    svg.appendChild(nodesGroup);
  }

  function animateNetworkFlow(data) {
    const winnerIdx = data.predicted_index;

    // Highlight winner output node with dynamic glowing halo
    for (let i = 0; i < 3; i++) {
      const circle = document.getElementById(`node-circle-3-${i}`);
      if (!circle) continue;
      if (i === winnerIdx) {
        circle.setAttribute("stroke", "#ffffff");
        circle.setAttribute("stroke-width", "3.5");
        circle.setAttribute("filter", "url(#glow)");
        circle.setAttribute("fill-opacity", "1.0");
        circle.setAttribute("r", "16");
      } else {
        circle.removeAttribute("filter");
        circle.setAttribute("stroke-width", "1");
        circle.setAttribute("fill-opacity", "0.35");
        circle.setAttribute("r", "12");
      }
    }
  }

  // Initial trigger
  updateStandardizationTags();
});
