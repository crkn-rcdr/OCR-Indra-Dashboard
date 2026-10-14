# CRKN Digital Collections: Indra OCR Quality Benchmark

Comprehensive empirical evaluation, automated lexical validity scoring, and spatial bounding-box alignment analysis comparing **Indra OCR** against the **ABBYY FineReader** commercial baseline across **7,712 historical digital archive pages** from the Canadian Research Knowledge Network (CRKN / RCDR) collections.

**Live Interactive Dashboard**: [https://crkn-rcdr.github.io/OCR-Indra-Dashboard/](https://crkn-rcdr.github.io/OCR-Indra-Dashboard/)

---

## 1. Executive Summary & Quality Verdict

Indra OCR is an ultra-fast, lightweight OCR engine designed to execute on consumer and server CPUs via ONNX runtimes. Across the complete CRKN benchmark test suite (**7,712 historical pages**, **1,809,905 recognized word tokens**), the benchmark revealed clear architectural strengths and performance boundaries:

```
+-----------------------------------------------------------------------------------------+
|                                    ARCHIVAL BENCHMARK                                   |
|                                                                                         |
|   1. Annuals (Clean Modern Print):       93.0% Concordance | 0.958 Sim | 22.7% QA Win   |
|   2. Heritage (Degraded Microfilm):      54.7% Concordance | 0.726 Sim |  9.7% QA Win   |
|   3. Newspapers (Multi-Column Serials):  63.2% Concordance | 0.678 Sim | 23.9% QA Win   |
|                                                                                         |
|   Overall Archive Concordance: 58.3%  |  Indra QA Win Rate: 10.8% (vs 89.2% ABBYY)     |
|   CPU Processing Throughput: ~8 to 12 pages/sec across 6 worker processes               |
+-----------------------------------------------------------------------------------------+
```

### Key Takeaway
* **Production-Grade on Clean Monograph Stock**: On standardized book publication streams (**Annuals**), Indra closely tracks commercial quality (**93.0% clean text concordance**, **89.7% exact character match**, and **94.6% spatial bounding-box alignment**). It reliably serves as an ultra-fast, zero-GPU drop-in engine for clean, well-lit typography.
* **Significant Degradation on Archival Microfilm**: On historical microfilm reels (**Heritage**), performance drops sharply to **54.7% clean concordance** and **9.7% lexical win rate** (ABBYY won 187,306 word disputes vs. Indra's 20,219). Indra lacks specialized computer vision pre-processing (adaptive CLAHE, contrast equalization, stroke reconnection), causing it to drop faint ink, suffer from microfilm grain noise, and miss low-contrast historical script.
* **Complex Broadsheet Segmentation**: On multi-column newspapers (**Newspapers**), 3×2 quadrant tiling prevented truncated text lines, but complex reading order arbitration and micro-font resolution lagged behind ABBYY (23.9% win rate).

---

## 2. Quantitative Performance Breakdown

### A. High-Level Metrics vs. ABBYY FineReader

| Metric | Indra OCR (Fast CPU) | ABBYY FineReader (Baseline) | PaddleOCR (PP-OCRv6 + Filters)* |
| :--- | :---: | :---: | :---: |
| **Pages Evaluated** | **7,712** | 7,712 | 4,154 |
| **Total Words Extracted** | **1,809,905** | 1,854,006 | 1,600,728 |
| **Spatial IoU Alignment (≥0.25)** | **70.0%** (1,267,165 words) | Baseline | 72.0% |
| **Clean Text Concordance** | **58.26%** | Baseline | **80.39%** |
| **Exact Text Concordance** | **55.02%** | Baseline | **77.71%** |
| **Mean Character Similarity** | **0.7464** | Baseline | **0.8785** |
| **Mean Bounding Box IoU** | **0.4940** | Baseline | **0.4857** |
| **Total Discrepant Tokens** | 404,256 | — | 183,136 |
| **Decisive QA Win Rate** | **10.79%** (24,283 wins) | **89.21%** (200,706 wins) | **63.65%** (48,505 wins) |
| **Processing Speed** | **~8–12 pages/sec (CPU)** | ~0.2–0.5 pages/sec | ~0.8–1.5 pages/sec (GPU) |

*\*PaddleOCR benchmark data included for cross-engine architectural comparison.*

---

### B. Collection-by-Collection Performance Analysis

#### 1. Annuals (Clean Printed Books & Yearbooks - 701 pages)
* **Spatial Alignment**: 94.6%
* **Clean Concordance**: 92.99% (Exact: 89.65%)
* **Mean Character Similarity**: 0.958
* **Lexical Arbitration**: 2,112 Indra wins vs. 7,182 ABBYY wins (22.7% win rate)
* **Analysis**: Modern typography and high-contrast printed paper represent Indra's sweet spot. With sub-second execution (~0.5s/page), Indra extracts clean, publication-grade text with minimal spelling variance.

#### 2. Heritage (Degraded Archival Microfilm - 6,951 pages)
* **Spatial Alignment**: 62.3%
* **Clean Concordance**: 54.71% (Exact: 51.52%)
* **Mean Character Similarity**: 0.726
* **Lexical Arbitration**: 20,219 Indra wins vs. 187,306 ABBYY wins (9.74% win rate)
* **Analysis**: Historical microfilm exhibits uneven lighting, dark celluloid borders, faded ink, and bleed-through. Because Indra does not apply adaptive local thresholding or morphological denoising, its detection boxes frequently miss faint text or fragment ligatures. For microfilm collections, heavy pre-processing (CLAHE + stroke reconnection) or fine-tuned GPU models (Paddle PP-OCRv6) remain necessary.

#### 3. Newspapers (Dense Broadsheet Serials - 60 pages)
* **Spatial Alignment**: 87.9%
* **Clean Concordance**: 63.19% (Exact: 59.84%)
* **Mean Character Similarity**: 0.678
* **Lexical Arbitration**: 1,952 Indra wins vs. 6,218 ABBYY wins (23.89% win rate)
* **Analysis**: Standard single-pass OCR truncates micro-print across 8-column broadsheets. Implementing 3×2 quadrant tiling resolved box clipping and restored spatial alignment to 87.9%, but column-flow arbitration and dense font recognition remain challenging on century-old newsprint.

---

## 3. Operational Throughput & Parallelism

Indra's ONNX models run purely on CPU without requiring Nvidia CUDA or GPU memory. 

By utilizing page-level multiprocessing with Python's `ProcessPoolExecutor`:
* **Single Core / Sequential**: ~0.55s per page (~1.8 pages/sec)
* **6 Worker Processes**: ~8.5 to 11.2 pages/sec aggregate throughput
* **Entire 7,712-Page Archive**: Completed in **under 35 minutes** on standard workstation hardware.

---

## 4. Repository Structure & Dashboard Deployment

This repository hosts the static evaluation dashboard deployed via GitHub Pages from `/docs`:

```
OCR-Indra-Dashboard/
├── README.md                           # Benchmark documentation and architectural analysis
└── docs/                               # GitHub Pages deployment root
    ├── index.html                      # Interactive Master AI Quality Dashboard
    ├── comparison_summary.json         # Complete 7,712-page structured metrics
    ├── comparison_summary.csv          # Exportable metrics spreadsheet
    ├── Annuals/                        # Interactive side-by-side page diff viewers
    ├── Heritage/                       # Interactive side-by-side page diff viewers
    └── Newspapers/                     # Interactive side-by-side page diff viewers
```

### Navigating the Interactive Dashboard
1. **Executive Verdict & Macro KPIs**: Real-time summary of archive-wide text recall, concordance, and word win rate.
2. **Collection Filters**: Filter instantly between Annuals, Heritage, and Newspapers.
3. **Interactive Side-by-Side Viewers**: Click any page link to open the synchronized side-by-side comparison viewer showing the original scan, pipeline bounding boxes, ABBYY bounding boxes, and automated lexical discrepancy adjudications.

---

## 5. Summary Recommendation for CRKN Digital Collections

| Use Case | Recommended Engine | Rationale |
| :--- | :---: | :--- |
| **Clean Book & Periodical Digitization** | **Indra OCR** | Extreme throughput (~10 pgs/s on CPU), zero GPU dependency, 93% concordance on standard print. |
| **Archival Microfilm & Census Records** | **Paddle PP-OCRv6** | Adaptive CLAHE pre-processing, stroke reconnection, and higher recognition accuracy on degraded ink. |
| **High-Volume Newspaper Broadsheets** | **Paddle + 3×2 Tiling** | Superior multi-column layout parsing and micro-print resolution on 8-column broadsheets. |
| **Rapid Searchable PDF Triage** | **Indra OCR** | Fast conversion from scanned image to ALTO XML and searchable PDF/A with minimal resource footprint. |
