import { useEffect, useMemo, useState } from "react";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
const LOCAL_HOSTNAMES = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

const OCR_MODES = [
  {
    id: "normal",
    label: "Normal OCR",
    autoCrop: false,
    cropUse: "gray",
  },
  {
    id: "gray",
    label: "Auto crop gray",
    autoCrop: true,
    cropUse: "gray",
  },
  {
    id: "bw",
    label: "Auto crop bw",
    autoCrop: true,
    cropUse: "bw",
  },
  {
    id: "color",
    label: "Auto crop color",
    autoCrop: true,
    cropUse: "color",
  },
];

function getApiBaseUrl() {
  const envValue = import.meta.env.VITE_API_BASE_URL?.trim();

  if (envValue) {
    return envValue.replace(/\/+$/, "");
  }

  if (typeof window !== "undefined") {
    const hostname = window.location.hostname;

    if (hostname && !LOCAL_HOSTNAMES.has(hostname)) {
      return `http://${hostname}:8000`;
    }
  }

  return DEFAULT_API_BASE_URL;
}

function getErrorMessage(error, fallback = "Request failed") {
  if (!error) {
    return fallback;
  }

  if (typeof error === "string") {
    return error;
  }

  if (Array.isArray(error)) {
    return error.map((item) => item.msg || JSON.stringify(item)).join("; ");
  }

  if (typeof error === "object") {
    return error.detail || error.message || JSON.stringify(error);
  }

  return String(error);
}

function toAssetUrl(path, apiBaseUrl) {
  if (!path || typeof path !== "string") {
    return "";
  }

  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  const normalizedPath = path.replace(/\\/g, "/");

  if (normalizedPath.startsWith("/outputs/")) {
    return `${apiBaseUrl}${normalizedPath}`;
  }

  if (normalizedPath.startsWith("outputs/")) {
    return `${apiBaseUrl}/${normalizedPath}`;
  }

  return "";
}

function getClipboardImageFile(clipboardData) {
  if (!clipboardData) {
    return null;
  }

  const pastedFile = Array.from(clipboardData.files || []).find((file) =>
    file.type.startsWith("image/"),
  );

  if (pastedFile) {
    return pastedFile;
  }

  const imageItem = Array.from(clipboardData.items || []).find(
    (item) => item.kind === "file" && item.type.startsWith("image/"),
  );

  return imageItem?.getAsFile() ?? null;
}

function withDefaultImageName(file, fallbackName = "pasted-image") {
  if (file.name) {
    return file;
  }

  const extension = file.type === "image/jpeg" ? "jpg" : "png";
  return new File([file], `${fallbackName}.${extension}`, {
    type: file.type || "image/png",
    lastModified: Date.now(),
  });
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();

    image.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(image);
    };

    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Cannot read this image. Please choose a browser-supported image file."));
    };

    image.src = objectUrl;
  });
}

function canvasToJpegBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
          return;
        }

        reject(new Error("Cannot convert image to JPEG."));
      },
      "image/jpeg",
      0.92,
    );
  });
}

async function convertImageFileToJpeg(file) {
  if (!file?.type?.startsWith("image/")) {
    throw new Error("Selected file is not an image.");
  }

  const image = await loadImageFromFile(file);
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;

  if (!width || !height) {
    throw new Error("Cannot read image dimensions.");
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");

  if (!context) {
    throw new Error("This browser cannot convert images with canvas.");
  }

  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  context.drawImage(image, 0, 0, width, height);

  const blob = await canvasToJpegBlob(canvas);
  const baseName = (file.name || "ocr-image").replace(/\.[^.]+$/, "");

  return new File([blob], `${baseName}.jpg`, {
    type: "image/jpeg",
    lastModified: Date.now(),
  });
}

function App() {
  const apiBaseUrl = useMemo(getApiBaseUrl, []);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [modeId, setModeId] = useState("normal");
  const [padX, setPadX] = useState(15);
  const [padY, setPadY] = useState(25);
  const [dilateIter, setDilateIter] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const selectedMode = OCR_MODES.find((mode) => mode.id === modeId) ?? OCR_MODES[0];

  function setImageFile(file) {
    setSelectedFile(file);
    setResult(null);
    setError("");
  }

  useEffect(() => {
    function handlePaste(event) {
      const imageFile = getClipboardImageFile(event.clipboardData);

      if (!imageFile) {
        return;
      }

      event.preventDefault();
      setImageFile(withDefaultImageName(imageFile));
    }

    window.addEventListener("paste", handlePaste);

    return () => window.removeEventListener("paste", handlePaste);
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  function handleFileChange(event) {
    const file = event.target.files?.[0] ?? null;
    setImageFile(file);
  }

  async function handlePredict(event) {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please choose or capture an image before predicting.");
      return;
    }

    setIsLoading(true);
    setResult(null);
    setError("");

    try {
      const uploadFile = await convertImageFileToJpeg(selectedFile);
      const formData = new FormData();
      formData.append("file", uploadFile);
      formData.append("auto_crop", String(selectedMode.autoCrop));
      formData.append("crop_use", selectedMode.cropUse);
      formData.append("pad_x", String(padX));
      formData.append("pad_y", String(padY));
      formData.append("dilate_iter", String(dilateIter));

      const response = await fetch(`${apiBaseUrl}/predict/line`, {
        method: "POST",
        body: formData,
      });

      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json")
        ? await response.json()
        : await response.text();

      if (!response.ok) {
        throw new Error(
          `API error ${response.status}: ${getErrorMessage(payload, response.statusText)}`,
        );
      }

      setResult(payload);
    } catch (err) {
      setError(getErrorMessage(err, "Cannot connect to OCR API."));
    } finally {
      setIsLoading(false);
    }
  }

  const usedImageUrl = toAssetUrl(result?.used_image, apiBaseUrl);
  const cropLinks = ["bbox", "gray"]
    .map((key) => ({
      key,
      path: result?.crop_paths?.[key],
      url: toAssetUrl(result?.crop_paths?.[key], apiBaseUrl),
    }))
    .filter((link) => link.path && link.url);

  return (
    <main className="app-shell">
      <section className="workspace" aria-labelledby="app-title">
        <header className="app-header">
          <div>
            <p className="eyebrow">OCR demo</p>
            <h1 id="app-title">Vietnamese Handwriting OCR</h1>
          </div>
          <span className="api-chip">{apiBaseUrl}</span>
        </header>

        <form className="ocr-layout" onSubmit={handlePredict}>
          <div className="panel upload-panel">
            <div className="section-heading">
              <h2>Upload / Camera</h2>
            </div>

            <label className="file-drop" htmlFor="image-input">
              <input
                id="image-input"
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleFileChange}
              />
              <span className="file-drop-title">
                {selectedFile ? selectedFile.name : "Choose, capture, or paste image"}
              </span>
              <span className="file-drop-meta">
                Ctrl+V supported. Images are sent as JPG.
              </span>
            </label>

            <div className="preview-frame">
              {previewUrl ? (
                <img src={previewUrl} alt="Selected handwriting preview" />
              ) : (
                <span>No image selected</span>
              )}
            </div>
          </div>

          <div className="panel options-panel">
            <div className="section-heading">
              <h2>Options</h2>
            </div>

            <fieldset className="mode-group">
              <legend>Mode</legend>
              <div className="segmented-control">
                {OCR_MODES.map((mode) => (
                  <label key={mode.id} className={mode.id === modeId ? "active" : ""}>
                    <input
                      type="radio"
                      name="ocr-mode"
                      value={mode.id}
                      checked={mode.id === modeId}
                      onChange={() => {
                        setModeId(mode.id);
                        setResult(null);
                        setError("");
                      }}
                    />
                    <span>{mode.label}</span>
                  </label>
                ))}
              </div>
            </fieldset>

            <div className="number-grid">
              <label>
                <span>pad_x</span>
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={padX}
                  onChange={(event) => setPadX(event.target.value)}
                />
              </label>
              <label>
                <span>pad_y</span>
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={padY}
                  onChange={(event) => setPadY(event.target.value)}
                />
              </label>
              <label>
                <span>dilate_iter</span>
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={dilateIter}
                  onChange={(event) => setDilateIter(event.target.value)}
                />
              </label>
            </div>

            <button className="predict-button" type="submit" disabled={isLoading}>
              {isLoading ? "Predicting..." : "Predict"}
            </button>
          </div>

          <div className="panel result-panel">
            <div className="section-heading">
              <h2>Result</h2>
            </div>

            {error && <div className="message error-message">{error}</div>}

            {isLoading && <div className="message loading-message">Loading OCR result...</div>}

            {!error && !isLoading && !result && (
              <div className="empty-result">Prediction will appear here.</div>
            )}

            {result && (
              <div className="result-content">
                <div>
                  <span className="result-label">Prediction</span>
                  <p className="prediction-text">{result.prediction || "(empty)"}</p>
                </div>

                {result.used_image && (
                  <div className="path-block">
                    <span className="result-label">used_image</span>
                    {usedImageUrl ? (
                      <a href={usedImageUrl} target="_blank" rel="noreferrer">
                        {result.used_image}
                      </a>
                    ) : (
                      <code>{result.used_image}</code>
                    )}
                  </div>
                )}

                {cropLinks.length > 0 && (
                  <div className="crop-links">
                    <span className="result-label">Crop / Debug</span>
                    <div>
                      {cropLinks.map((link) => (
                        <a key={link.key} href={link.url} target="_blank" rel="noreferrer">
                          {link.key}: {link.path}
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </form>
      </section>
    </main>
  );
}

export default App;
