import React, { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [blueprint, setBlueprint] = useState(null);
  const [referenceBom, setReferenceBom] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    if (!blueprint) {
      setError("Please select a blueprint image.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", blueprint);

      if (referenceBom) {
        formData.append("reference_bom_file", referenceBom);
      }

      const response = await fetch(
        `${API_URL}/analyze-blueprint`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Analysis failed.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err.message || "Unable to connect to backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setBlueprint(null);
    setReferenceBom(null);
    setResult(null);
    setError("");

    const blueprintInput =
      document.getElementById("blueprint-input");
    const referenceInput =
      document.getElementById("reference-input");

    if (blueprintInput) blueprintInput.value = "";
    if (referenceInput) referenceInput.value = "";
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>BOM Mismatch Detection System</h1>
          <p>
            Deep Learning-Based Engineering Blueprint Analysis
          </p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Backend Connected
        </div>
      </header>

      <main className="container">
        <section className="upload-section">
          <div className="upload-card">
            <div className="card-label">INPUT 01</div>

            <h2>Upload Blueprint</h2>

            <p className="description">
              Upload an engineering blueprint image
              containing a Bill of Materials.
            </p>

            <label
              htmlFor="blueprint-input"
              className="file-box"
            >
              <span className="upload-icon">📄</span>

              <strong>
                {blueprint
                  ? blueprint.name
                  : "Choose blueprint image"}
              </strong>

              <span>
                PNG, JPG or JPEG
              </span>
            </label>

            <input
              id="blueprint-input"
              type="file"
              accept=".png,.jpg,.jpeg"
              onChange={(e) =>
                setBlueprint(
                  e.target.files[0] || null
                )
              }
            />
          </div>

          <div className="upload-card">
            <div className="card-label">INPUT 02</div>

            <h2>Reference BOM</h2>

            <p className="description">
              Optional. Upload the expected BOM
              as a CSV file for comparison.
            </p>

            <label
              htmlFor="reference-input"
              className="file-box"
            >
              <span className="upload-icon">📊</span>

              <strong>
                {referenceBom
                  ? referenceBom.name
                  : "Choose reference CSV"}
              </strong>

              <span>
                CSV format
              </span>
            </label>

            <input
              id="reference-input"
              type="file"
              accept=".csv"
              onChange={(e) =>
                setReferenceBom(
                  e.target.files[0] || null
                )
              }
            />
          </div>
        </section>

        <section className="action-section">
          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={loading}
          >
            {loading
              ? "Analyzing Blueprint..."
              : "Analyze Blueprint"}
          </button>

          <button
            className="reset-button"
            onClick={handleReset}
            disabled={loading}
          >
            Reset
          </button>
        </section>

        {error && (
          <div className="error-box">
            <strong>Error:</strong> {error}
          </div>
        )}

        {loading && (
          <div className="loading-box">
            <div className="spinner"></div>

            <div>
              <strong>
                Processing blueprint...
              </strong>

              <p>
                OCR and table analysis are extracting
                BOM information.
              </p>
            </div>
          </div>
        )}

        {result && (
          <Results result={result} />
        )}
      </main>
    </div>
  );
}

function Results({ result }) {
  const bom = result.extracted_bom || [];
  const comparison = result.comparison;

  const mismatchedFields =
    comparison?.summary?.mismatched_fields || 0;

  const matchingFields =
    comparison?.summary?.matching_fields || 0;

  const totalFields =
    comparison?.summary?.total_fields_checked || 0;

  const hasMismatch = mismatchedFields > 0;

  return (
    <section className="results">
      <div className="results-header">
        <div>
          <div className="section-label">
            ANALYSIS REPORT
          </div>

          <h2>Analysis Results</h2>

          <p className="filename">
            {result.filename}
          </p>
        </div>

        <div
          className={
            hasMismatch
              ? "result-status warning"
              : "result-status success"
          }
        >
          <span className="status-icon">
            {hasMismatch ? "!" : "✓"}
          </span>

          <div>
            <strong>
              {comparison
                ? hasMismatch
                  ? "Differences Detected"
                  : "BOM Matched"
                : "BOM Extracted"}
            </strong>

            <span>
              {comparison
                ? hasMismatch
                  ? `${mismatchedFields} field${
                      mismatchedFields !== 1
                        ? "s"
                        : ""
                    } differ`
                  : "All compared fields match"
                : "Reference comparison unavailable"}
            </span>
          </div>
        </div>
      </div>

      {comparison && (
        <div className="summary-message">
          <div className="summary-message-icon">
            {hasMismatch ? "!" : "✓"}
          </div>

          <div>
            <strong>
              {hasMismatch
                ? "Differences were found between the extracted BOM and the reference BOM."
                : "The extracted BOM matches the reference BOM."}
            </strong>

            <p>
              {totalFields} fields were compared across{" "}
              {result.extracted_bom_rows} BOM rows.
            </p>
          </div>
        </div>
      )}

      <div className="stats">
        <StatCard
          label="OCR Detections"
          value={result.ocr_detections}
          icon="◎"
        />

        <StatCard
          label="OCR Rows"
          value={result.ocr_rows}
          icon="≡"
        />

        <StatCard
          label="Extracted BOM"
          value={result.extracted_bom_rows}
          icon="▤"
        />

        <StatCard
          label="Reference Rows"
          value={result.reference_bom_rows}
          icon="✓"
        />

        {comparison && (
          <>
            <StatCard
              label="Matching Fields"
              value={matchingFields}
              icon="✓"
              type="success"
            />

            <StatCard
              label="Different Fields"
              value={mismatchedFields}
              icon="!"
              type={
                mismatchedFields > 0
                  ? "warning"
                  : "success"
              }
            />
          </>
        )}
      </div>

      <div className="table-section">
        <div className="section-heading">
          <div>
            <h3>Extracted BOM</h3>
            <p>
              Structured BOM information extracted from
              the blueprint.
            </p>
          </div>

          <span className="row-count">
            {bom.length} rows
          </span>
        </div>

        {bom.length === 0 ? (
          <div className="empty-state">
            No BOM rows were extracted.
          </div>
        ) : (
          <div className="table-wrapper">
            <table className="bom-table">
              <thead>
                <tr>
                  <th>Part No</th>
                  <th>Description</th>
                  <th>Material</th>
                  <th>UOM</th>
                  <th>Qty</th>
                </tr>
              </thead>

              <tbody>
                {bom.map((row, index) => (
                  <tr key={index}>
                    <td className="part-number">
                      {row.PART_NO || "—"}
                    </td>

                    <td>
                      {row.DESCRIPTION || "—"}
                    </td>

                    <td>
                      {row.MATERIAL || "—"}
                    </td>

                    <td>
                      {row.UOM || "—"}
                    </td>

                    <td className="quantity">
                      {row.QTY || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {comparison && (
        <ComparisonReport
          comparison={comparison}
        />
      )}
    </section>
  );
}

function StatCard({
  label,
  value,
  icon,
  type = "",
}) {
  return (
    <div className={`stat-card ${type}`}>
      <div className="stat-icon">
        {icon}
      </div>

      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function ComparisonReport({ comparison }) {
  const [filter, setFilter] = useState("ALL");

  const rows = comparison.results || [];

  const filteredRows = rows.filter((item) => {
    if (filter === "MATCH") {
      return item.STATUS === "MATCH";
    }

    if (filter === "MISMATCH") {
      return item.STATUS !== "MATCH";
    }

    return true;
  });

  const matching =
    comparison.summary?.matching_fields || 0;

  const mismatched =
    comparison.summary?.mismatched_fields || 0;

  const total =
    comparison.summary?.total_fields_checked || 0;

  return (
    <div className="comparison-section">
      <div className="comparison-header">
        <div>
          <div className="section-label">
            REFERENCE VALIDATION
          </div>

          <h3>BOM Comparison</h3>

          <p>
            Field-by-field comparison between the
            extracted BOM and reference BOM.
          </p>
        </div>

        <div
          className={
            mismatched > 0
              ? "comparison-status warning"
              : "comparison-status success"
          }
        >
          {mismatched > 0
            ? "Differences Detected"
            : "All Fields Match"}
        </div>
      </div>

      <div className="comparison-summary">
        <div className="comparison-card">
          <span>Fields Checked</span>
          <strong>{total}</strong>
        </div>

        <div className="comparison-card success-card">
          <span>Matching</span>
          <strong>{matching}</strong>
        </div>

        <div className="comparison-card warning-card">
          <span>Different</span>
          <strong>{mismatched}</strong>
        </div>
      </div>

      <div className="comparison-toolbar">
        <div>
          <strong>Detailed Comparison</strong>
          <span>
            {filteredRows.length} results shown
          </span>
        </div>

        <div className="filter-buttons">
          <button
            className={
              filter === "ALL"
                ? "filter-button active"
                : "filter-button"
            }
            onClick={() => setFilter("ALL")}
          >
            All
          </button>

          <button
            className={
              filter === "MATCH"
                ? "filter-button active"
                : "filter-button"
            }
            onClick={() => setFilter("MATCH")}
          >
            Matches
          </button>

          <button
            className={
              filter === "MISMATCH"
                ? "filter-button active"
                : "filter-button"
            }
            onClick={() =>
              setFilter("MISMATCH")
            }
          >
            Differences
          </button>
        </div>
      </div>

      <div className="table-wrapper">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Part No</th>
              <th>Field</th>
              <th>Expected</th>
              <th>Detected</th>
              <th>Status</th>
            </tr>
          </thead>

          <tbody>
            {filteredRows.map(
              (item, index) => {
                const isMatch =
                  item.STATUS === "MATCH";

                return (
                  <tr
                    key={index}
                    className={
                      isMatch
                        ? ""
                        : "difference-row"
                    }
                  >
                    <td className="part-number">
                      {item.PART_NO || "—"}
                    </td>

                    <td>
                      <span className="field-name">
                        {item.FIELD}
                      </span>
                    </td>

                    <td>
                      {item.EXPECTED || "—"}
                    </td>

                    <td>
                      {item.DETECTED || "—"}
                    </td>

                    <td>
                      <span
                        className={
                          isMatch
                            ? "status-pill match"
                            : "status-pill mismatch"
                        }
                      >
                        {isMatch
                          ? "MATCH"
                          : "DIFFERENCE"}
                      </span>
                    </td>
                  </tr>
                );
              }
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default App;

