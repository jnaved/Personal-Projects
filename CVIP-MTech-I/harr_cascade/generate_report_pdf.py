"""
generate_report_pdf.py: Generates a high-quality PDF report for the Haar Cascade Face Detection Assignment.
Utilizes ReportLab to render a clean, structured document matching the SVNIT M.Tech curriculum standards.
"""

import os
import sys
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = "harr_cascade/haar_cascade_face_detection_report.pdf"
CSV_PATH = "harr_cascade/outputs/experiments_summary.csv"
OUTPUT_DIR = "harr_cascade/outputs"


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' and header."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8.5)
        self.setStrokeColor(colors.HexColor("#bbbbbb"))
        self.setFillColor(colors.HexColor("#444444"))
        self.setLineWidth(0.5)

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.line(40, A4[1] - 42, A4[0] - 40, A4[1] - 42)
            self.drawString(40, A4[1] - 36, "SVNIT Surat | M.Tech (CSE) | CVIP (CS6111) — Haar Cascade Face Detection Lab")
            self.drawRightString(A4[0] - 40, A4[1] - 36, "Shaik Jnaved (P26DS001)")

        # Footer (all pages)
        self.line(40, 42, A4[0] - 40, 42)
        self.drawString(40, 30, "Confidential & Academic Laboratory Report | Department of Computer Science & Engineering")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 40, 30, page_str)
        self.restoreState()


def build_pdf():
    os.makedirs(os.path.dirname(PDF_OUTPUT_PATH), exist_ok=True)
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=52,
        bottomMargin=52
    )

    styles = getSampleStyleSheet()
    
    # Custom Color Palette
    primary_color = colors.HexColor("#0f2d59")   # Deep navy
    secondary_color = colors.HexColor("#1b6ca8") # Vibrant blue
    accent_color = colors.HexColor("#b33939")    # Deep red / crimson
    bg_light = colors.HexColor("#f8f9fa")        # Clean card gray
    border_color = colors.HexColor("#dcdde1")

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        alignment=1, # Center
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        alignment=1,
        spaceAfter=14
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#2f3640"),
        alignment=1
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e272e"),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e272e"),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )
    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#2c3e50")
    )
    callout_style = ParagraphStyle(
        'Callout_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2c3e50")
    )

    story = []

    # ---------------------------------------------------------
    # COVER / HEADER BLOCK
    # ---------------------------------------------------------
    story.append(Paragraph("SARDAR VALLABHBHAI NATIONAL INSTITUTE OF TECHNOLOGY, SURAT", ParagraphStyle('Univ', fontName='Helvetica-Bold', fontSize=11, leading=14, alignment=1, textColor=primary_color)))
    story.append(Paragraph("Department of Computer Science and Engineering | M.Tech. I — Semester I", ParagraphStyle('Dept', fontName='Helvetica', fontSize=9.5, leading=13, alignment=1, textColor=colors.HexColor("#57606f"))))
    story.append(Paragraph("Computer Vision and Image Processing (CS6111 / CSDS119)", ParagraphStyle('Course', fontName='Helvetica-Bold', fontSize=10, leading=14, alignment=1, textColor=secondary_color)))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph("Laboratory Report: Haar Cascade Face Detection", title_style))
    story.append(Paragraph("Empirical Analysis of Attentional Cascade Models, Parameter Sensitivity, and Perturbation Robustness under Scale, Illumination, and Orientation Variations", subtitle_style))
    
    # Metadata Card Table
    meta_data = [
        [
            Paragraph("<b>Student Name:</b> Shaik Jnaved", meta_style),
            Paragraph("<b>Enrollment No.:</b> P26DS001", meta_style),
            Paragraph("<b>Course Code:</b> CS6111 / CSDS119", meta_style)
        ],
        [
            Paragraph("<b>Degree:</b> M.Tech. CSE (Data Science)", meta_style),
            Paragraph("<b>Submission Date:</b> October 2026", meta_style),
            Paragraph("<b>Model Evaluated:</b> Haar Frontal Cascade", meta_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[170, 170, 175])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('BOX', (0, 0), (-1, -1), 0.8, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 1: PROBLEM STATEMENT & OBJECTIVES
    # ---------------------------------------------------------
    story.append(Paragraph("1. Problem Statement and Assigned Task", h1_style))
    story.append(Paragraph(
        "Face detection serves as the critical foundational stage in modern visual perception pipelines, biometrics, surveillance, "
        "and human-computer interaction (HCI). The assigned laboratory assignment mandates the rigorous examination of classical "
        "Haar Cascade object detection (Viola-Jones architecture), code-understanding and debugging via LLM assistance, implementation "
        "redesign, and empirical benchmarking across multiple detection configurations and environmental perturbations.",
        body_style
    ))
    story.append(Paragraph("<b>Prescribed Core Objectives:</b>", body_style))
    story.append(Paragraph("&bull; <b>Study and Execute:</b> Inspect the baseline GitHub repository (<code>main.py</code>, <code>run.py</code>, <code>haarcascade_frontalface_default.xml</code>) and diagnose its structural/runtime limitations.", bullet_style))
    story.append(Paragraph("&bull; <b>LLM Task Collaboration:</b> Request a comprehensive theoretical and mathematical breakdown of the boosted cascade model and the exact operational role of the parameters used in OpenCV's <code>detectMultiScale</code>.", bullet_style))
    story.append(Paragraph("&bull; <b>Code Redesign and Enhancement:</b> Modernize the implementation (<code>main_improved.py</code>) into a modular, multi-modal engine supporting images, directories, video, CLI parameter tuning, and telemetry.", bullet_style))
    story.append(Paragraph("&bull; <b>Controlled Systematic Experiments:</b> Test at least three parameter settings across systematic variations in <b>scale</b>, <b>illumination</b>, and <b>face orientation</b>, quantitatively recording successful and failed detections.", bullet_style))
    story.append(Paragraph("&bull; <b>Comprehensive Reporting:</b> Compile code diffs, input/output visualizations, latency benchmarks, and analytical observations into a formal report.", bullet_style))
    story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # SECTION 2: THEORETICAL FOUNDATION OF HAAR CASCADE
    # ---------------------------------------------------------
    story.append(Paragraph("2. Theoretical Foundation: The Viola-Jones Framework", h1_style))
    story.append(Paragraph(
        "Introduced by Paul Viola and Michael Jones in their seminal 2001 paper (<i>'Rapid Object Detection using a Boosted Cascade of Simple Features'</i>), "
        "the Haar Cascade framework pioneered the first real-time capable object detection algorithm in computer vision history. "
        "Its operational efficiency is underpinned by four core mathematical pillars:",
        body_style
    ))
    
    theory_items = [
        ("Haar-like Features",
         "Rather than raw pixel intensities, the detector computes rectangular contrasts consisting of adjacent light and dark boxes "
         "(two-rectangle, three-rectangle, and four-rectangle diagonal filters). These features encode generic facial structural invariants: "
         "the eye region is consistently darker than the forehead and upper cheeks, and the nasal bridge is brighter than the bilateral nasal sidewalls."),
        ("Integral Image (Summed-Area Table)",
         "Evaluating rectangular features at multiple scales and positions across an image would be computationally catastrophic. "
         "The integral image represents each point (x, y) as the cumulative sum of all pixels above and to the left: "
         "<i>II(x, y) = &sum;<sub>x'&le;x, y'&le;y</sub> I(x', y')</i>. Using the four corner coordinates of any rectangle, its area pixel sum "
         "is evaluated in exact <b>O(1) constant time</b> with only 3 additions/subtractions, completely independent of filter scale."),
        ("AdaBoost Machine Learning (Feature Selection)",
         "A standard 24&times;24 detector sub-window generates over 160,000 candidate Haar rectangle features. AdaBoost (Adaptive Boosting) "
         "is leveraged to select a small, optimal subset of 'weak classifiers' (single-feature decision stumps) and assign optimal hypothesis weights "
         "&alpha;<sub>t</sub> to minimize empirical weighted classification error over extensive positive (faces) and negative (non-faces) training sets."),
        ("The Attentional Cascade Structure",
         "The defining innovation is the degenerate decision tree (cascade) of sequential stages. In a natural image, &gt;99.9% of scanned sub-windows "
         "are background non-faces. The cascade organizes classifiers into dozens of sequential stages of increasing complexity: Stage 1 may test only 1–2 "
         "features, Stage 2 tests ~10 features, while deeper stages test hundreds. If any stage rejects a sub-window, it is <b>immediately discarded</b> "
         "without evaluating subsequent stages. Only face-like candidates penetrate to deeper stages, yielding sub-millisecond average rejection time.")
    ]
    for title, desc in theory_items:
        story.append(Paragraph(f"&bull; <b>{title}:</b> {desc}", bullet_style))
    story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # SECTION 3: ROLE OF DETECTION PARAMETERS
    # ---------------------------------------------------------
    story.append(Paragraph("3. Role and Sensitivity of OpenCV Detection Parameters", h1_style))
    story.append(Paragraph(
        "The primary detection interface in OpenCV is <code>cv2.CascadeClassifier.detectMultiScale(image, scaleFactor, minNeighbors, minSize, maxSize)</code>. "
        "Each parameter directly dictates the trade-off between sensitivity, spatial precision, and computational execution time:",
        body_style
    ))

    param_table_data = [
        [
            Paragraph("<b>Parameter</b>", code_style),
            Paragraph("<b>Physical Role in Algorithm</b>", code_style),
            Paragraph("<b>Low Value Impact (&darr;)</b>", code_style),
            Paragraph("<b>High Value Impact (&uarr;)</b>", code_style)
        ],
        [
            Paragraph("<b>scaleFactor</b><br/>(e.g., 1.05, 1.10, 1.30)", body_style),
            Paragraph("Defines the step-size ratio used to scale down the image in the multi-scale image pyramid at each scan iteration.", body_style),
            Paragraph("<b>Dense scale sampling:</b> High granularity, detects subtle/intermediate face scales. <i>Cost:</i> Severe latency overhead.", body_style),
            Paragraph("<b>Coarse scale sampling:</b> Extremely fast processing. <i>Cost:</i> Steps over faces whose scale falls between pyramid levels.", body_style)
        ],
        [
            Paragraph("<b>minNeighbors</b><br/>(e.g., 2, 4, 8)", body_style),
            Paragraph("Spatial clustering threshold. Specifies how many overlapping candidate detection rectangles must agree to form a retained face cluster.", body_style),
            Paragraph("<b>Permissive:</b> High recall; catches faint, occluded, or low-contrast faces. <i>Cost:</i> Generates numerous false positives.", body_style),
            Paragraph("<b>Conservative:</b> Strict precision; eliminates spurious noise and texture false alarms. <i>Cost:</i> Misses genuine faces.", body_style)
        ],
        [
            Paragraph("<b>minSize / maxSize</b><br/>(e.g., (20,20), (40,40))", body_style),
            Paragraph("Sets the explicit pixel bounding box lower and upper search constraints.", body_style),
            Paragraph("Allows tiny/distant face search. <i>Cost:</i> Increases window evaluations and false alarms in high-frequency textures.", body_style),
            Paragraph("Prunes search space drastically, accelerating execution and avoiding noise. <i>Cost:</i> Ignores distant faces.", body_style)
        ]
    ]
    ptable = Table(param_table_data, colWidths=[95, 160, 130, 130])
    ptable.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(ptable)
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 4: LLM INTERACTION & CODE MODIFICATION
    # ---------------------------------------------------------
    story.append(Paragraph("4. LLM Interaction, Code Understanding, and Modifications", h1_style))
    story.append(Paragraph(
        "<b>Diagnostic Review of the Original Implementation:</b><br/>"
        "The original repository provided two scripts: <code>main.py</code> (36 lines) and <code>run.py</code> (10 lines). "
        "Upon thorough examination with the LLM, critical architectural deficiencies were identified:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Rigid Input Modality:</b> <code>main.py</code> hardcoded <code>cv2.VideoCapture(0)</code>, rendering automated benchmarking, static image evaluation, or batch testing impossible.", bullet_style))
    story.append(Paragraph("&bull; <b>Headless/WSL Incompatibility:</b> The script unconditionally called <code>cv2.imshow()</code> without checking GUI server availability, causing fatal crashes in headless servers, WSL, or non-desktop environments.", bullet_style))
    story.append(Paragraph("&bull; <b>Hardcoded Hyperparameters:</b> <code>detectMultiScale(gray, 1.1, 4)</code> provided no CLI arguments or runtime hooks to vary <code>scaleFactor</code> or <code>minNeighbors</code>.", bullet_style))
    story.append(Paragraph("&bull; <b>Zero Quantitative Telemetry:</b> Did not record bounding box coordinates, IoU metrics, inference latency, or export processed results to disk.", bullet_style))
    story.append(Paragraph("&bull; <b>Brittle Path Resolution:</b> Loading <code>'haarcascade_frontalface_default.xml'</code> via a relative path failed if the script was invoked from an exterior working directory.", bullet_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Architectural Modifications in <code>main_improved.py</code>:</b>", h2_style))
    mod_points = [
        ("Multi-Modal Input Handler", "Implemented flexible CLI arguments supporting <code>--image &lt;path&gt;</code>, <code>--dir &lt;folder&gt;</code>, <code>--video &lt;path&gt;</code>, and <code>--webcam [id]</code>."),
        ("Hyperparameter Exposing", "CLI flags for <code>--scaleFactor</code>, <code>--minNeighbors</code>, <code>--minSize</code>, and <code>--maxSize</code> enabling programmatic parameter sweeps."),
        ("Telemetry & Performance Timing", "Integrated <code>time.perf_counter()</code> benchmarking to quantify exact classifier latency in milliseconds alongside bounding box logging."),
        ("High-Tech HUD Annotation", "Engineered an overlay HUD with cyan/emerald corner accents, face sequence tags, dimension readouts, and a top status bar displaying detector settings."),
        ("Headless Batch Processing", "Added <code>--output &lt;path&gt;</code> headless image/video rendering and saving, bypassing GUI window requirements completely.")
    ]
    for title, desc in mod_points:
        story.append(Paragraph(f"&bull; <b>{title}:</b> {desc}", bullet_style))

    story.append(PageBreak()) # New page for experimental design and results

    # ---------------------------------------------------------
    # SECTION 5: EXPERIMENTAL DESIGN
    # ---------------------------------------------------------
    story.append(Paragraph("5. Systematic Experimental Design and Benchmark Suite", h1_style))
    story.append(Paragraph(
        "To satisfy the core requirement of testing across parameter settings and physical perturbation axes, "
        "an automated benchmarking engine (<code>run_experiments.py</code>) was developed. "
        "The experiment systematically evaluates four distinct parameter configurations against 18 controlled test conditions (72 total evaluation runs):",
        body_style
    ))

    config_table_data = [
        [
            Paragraph("<b>Configuration Name</b>", code_style),
            Paragraph("<b>scaleFactor</b>", code_style),
            Paragraph("<b>minNeighbors</b>", code_style),
            Paragraph("<b>minSize</b>", code_style),
            Paragraph("<b>Intended Operational Regime</b>", code_style)
        ],
        [
            Paragraph("<b>Config 1: Baseline (Default)</b>", body_style),
            Paragraph("1.10", body_style),
            Paragraph("4", body_style),
            Paragraph("(30, 30)", body_style),
            Paragraph("Standard balanced default. Balanced precision and recall for frontal faces.", body_style)
        ],
        [
            Paragraph("<b>Config 2: High Sensitivity</b>", body_style),
            Paragraph("1.05", body_style),
            Paragraph("2", body_style),
            Paragraph("(20, 20)", body_style),
            Paragraph("Permissive regime. Denser pyramid and low consensus to catch faint/small faces.", body_style)
        ],
        [
            Paragraph("<b>Config 3: High Precision</b>", body_style),
            Paragraph("1.30", body_style),
            Paragraph("8", body_style),
            Paragraph("(40, 40)", body_style),
            Paragraph("Conservative regime. Fast coarse pyramid and strict clustering to reject all noise.", body_style)
        ],
        [
            Paragraph("<b>Config 4: Fine-Scale Strict</b>", body_style),
            Paragraph("1.03", body_style),
            Paragraph("6", body_style),
            Paragraph("(30, 30)", body_style),
            Paragraph("Ultra-fine pyramid with moderate-high neighbor clustering for high fidelity.", body_style)
        ]
    ]
    ctable = Table(config_table_data, colWidths=[130, 65, 75, 65, 180])
    ctable.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(ctable)
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 6: SCALE VARIATION RESULTS
    # ---------------------------------------------------------
    story.append(Paragraph("6. Experimental Evaluation 1: Face Scale Variations", h1_style))
    story.append(Paragraph(
        "The standard Lena portrait (512&times;512, ground-truth face width = 169 px) was scaled across five discrete resolution factors: "
        "<b>Tiny</b> (0.18&times;, ~30 px face), <b>Small</b> (0.30&times;, ~50 px face), <b>Medium</b> (0.50&times;, ~85 px face), "
        "<b>Standard</b> (1.00&times;, 169 px face), and <b>Large</b> (1.60&times;, ~270 px face).",
        body_style
    ))

    scale_img_path = os.path.join(OUTPUT_DIR, "scale_experiment_grid_opt.jpg")
    if not os.path.exists(scale_img_path):
        scale_img_path = os.path.join(OUTPUT_DIR, "scale_experiment_grid.png")
    if os.path.exists(scale_img_path):
        story.append(Image(scale_img_path, width=7.2 * inch, height=3.6 * inch))
        story.append(Paragraph("<b>Figure 1:</b> Multi-scale face detection grid comparing Configs 1–4 across 5 scale levels. Bounding boxes show Ground Truth (yellow) and Detections (green for IoU &ge; 0.35).", callout_style))
        story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Scale Analysis & Observations:</b><br/>"
        "&bull; <b>Small-Scale Failure Threshold:</b> At Tiny scale (~30 px face), <b>Config 3 completely failed</b> (0 detections) because its <code>minSize=(40,40)</code> strictly forbade evaluating sub-windows below 40 pixels. "
        "In contrast, <b>Config 2 succeeded with high IoU (0.879)</b> due to its reduced <code>minSize=(20,20)</code> and permissive neighbor threshold (<code>minNeighbors=2</code>).<br/>"
        "&bull; <b>Execution Latency Scaling:</b> Latency scales inversely with <code>scaleFactor</code>. Config 3 evaluated in <b>5.9 ms</b> at standard scale, whereas Config 2 required <b>53.5 ms</b> and Config 4 required <b>63.5 ms</b>. "
        "At large image scale, Config 4 consumed <b>121.9 ms</b>—a 12&times; increase over Config 3 due to the dense pyramid steps.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # SECTION 7: ILLUMINATION VARIATION RESULTS
    # ---------------------------------------------------------
    story.append(Paragraph("7. Experimental Evaluation 2: Illumination & Contrast Variations", h1_style))
    story.append(Paragraph(
        "Six illumination environments were synthetically modeled using non-linear gamma curves, brightness shifts, and directional gradients: "
        "Severe Underexposure (&Delta;I=-70, &gamma;=2.5), Moderate Dim Light (&Delta;I=-35, &gamma;=1.5), Normal Diffuse, "
        "Moderate Overexposure (&Delta;I=+40, &gamma;=0.6), Specular Burn/Flash (&Delta;I=+80, &gamma;=0.35), and Directional Side Shadow.",
        body_style
    ))

    illum_img_path = os.path.join(OUTPUT_DIR, "illumination_experiment_grid_opt.jpg")
    if not os.path.exists(illum_img_path):
        illum_img_path = os.path.join(OUTPUT_DIR, "illumination_experiment_grid.png")
    if os.path.exists(illum_img_path):
        story.append(Image(illum_img_path, width=7.2 * inch, height=4.0 * inch))
        story.append(Paragraph("<b>Figure 2:</b> Illumination robustness grid comparing detection success across severe underexposure, dim light, normal, overexposure, specular burn, and directional side shadow.", callout_style))
        story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Illumination Analysis & Observations:</b><br/>"
        "&bull; <b>Robustness to Linear Shifts:</b> Across moderate dim lighting, moderate overexposure, and directional shadows, all four configurations achieved successful detections (IoU &gt; 0.83). "
        "Because Haar features calculate the <i>difference</i> between rectangle sums, linear illumination shifts cancel out across adjacent boxes.<br/>"
        "&bull; <b>Failure Modes under Non-Linear Saturation:</b> Under <b>Severe Underexposure</b> and <b>Specular Flash</b>, <b>Config 3 completely failed</b> (0 detections). "
        "In extreme darkness or highlight clipping, the contrast between eye sockets and nasal bridge falls below the AdaBoost decision stump thresholds. Overlapping detections drop below <code>minNeighbors=8</code>, leading to false negatives.",
        body_style
    ))

    story.append(PageBreak()) # New page for orientation, latency, and conclusions

    # ---------------------------------------------------------
    # SECTION 8: ORIENTATION VARIATION RESULTS
    # ---------------------------------------------------------
    story.append(Paragraph("8. Experimental Evaluation 3: In-Plane Roll & Pose Rotation", h1_style))
    story.append(Paragraph(
        "Face orientation was tested across 7 in-plane angles: 0&deg; (Frontal), +15&deg; (Slight tilt), -15&deg; (Opposite tilt), "
        "+30&deg; (Moderate tilt), +45&deg; (Steep diagonal), +90&deg; (Sideways), and 180&deg; (Upside-down inverted).",
        body_style
    ))

    orient_img_path = os.path.join(OUTPUT_DIR, "orientation_experiment_grid_opt.jpg")
    if not os.path.exists(orient_img_path):
        orient_img_path = os.path.join(OUTPUT_DIR, "orientation_experiment_grid.png")
    if os.path.exists(orient_img_path):
        story.append(Image(orient_img_path, width=7.2 * inch, height=4.2 * inch))
        story.append(Paragraph("<b>Figure 3:</b> Orientation evaluation grid revealing the critical architectural vulnerability of Haar Cascades under in-plane head roll rotations &ge; 30&deg;.", callout_style))
        story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Orientation Analysis & Fatal Vulnerability:</b><br/>"
        "&bull; <b>Slight Tilts (&plusmn;15&deg;):</b> Config 1 detected +15&deg; (IoU 0.698) but failed on -15&deg; due to asymmetric hair/shadow features. "
        "Config 2 and Config 4 detected the face at both &plusmn;15&deg;, but produced 1–2 <b>False Positives</b> in surrounding background textures.<br/>"
        "&bull; <b>Complete Failure at &ge; 30&deg;:</b> <b>At +30&deg;, +45&deg;, +90&deg;, and 180&deg;, ALL CONFIGURATIONS FAILED COMPLETELY.</b> "
        "<i>Root Cause:</i> Haar rectangle filters are rigidly <b>axis-aligned</b> (horizontal and vertical). When the head rotates &ge; 30&deg;, "
        "the bilateral symmetry and horizontal dark eye band no longer align with the rectangular integration grid, causing early-stage cascade rejection.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # SECTION 9: QUANTITATIVE SUMMARY CHARTS & METRICS
    # ---------------------------------------------------------
    story.append(Paragraph("9. Quantitative Benchmarks & Outcome Distribution", h1_style))

    chart_img_path = os.path.join(OUTPUT_DIR, "latency_and_accuracy_summary_opt.jpg")
    if not os.path.exists(chart_img_path):
        chart_img_path = os.path.join(OUTPUT_DIR, "latency_and_accuracy_summary.png")
    if os.path.exists(chart_img_path):
        story.append(Image(chart_img_path, width=7.2 * inch, height=2.8 * inch))
        story.append(Paragraph("<b>Figure 4:</b> Left: Mean inference latency (ms) per configuration across all test scenarios. Right: Detection outcome distribution (Success vs Partial vs Failed) across all 18 test variations.", callout_style))
        story.append(Spacer(1, 6))

    # Summary Statistics Table
    df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else None
    if df is not None:
        stats_rows = [
            [
                Paragraph("<b>Configuration</b>", code_style),
                Paragraph("<b>Mean Latency</b>", code_style),
                Paragraph("<b>Successes (TP&ge;1, FP=0)</b>", code_style),
                Paragraph("<b>Partial (TP+FP)</b>", code_style),
                Paragraph("<b>Failures (Missed)</b>", code_style),
                Paragraph("<b>Overall Success Rate</b>", code_style)
            ]
        ]
        for cfg_name in ["Config 1 (Baseline)", "Config 2 (High Sensitivity)", "Config 3 (High Precision)", "Config 4 (Fine Strict)"]:
            sub = df[df['Config'] == cfg_name]
            mean_lat = sub['Latency_ms'].mean()
            n_succ = sum("SUCCESS" in s for s in sub['Status'])
            n_part = sum("PARTIAL" in s for s in sub['Status'])
            n_fail = sum("FAILED" in s for s in sub['Status'])
            rate = (n_succ / len(sub)) * 100.0 if len(sub) > 0 else 0
            stats_rows.append([
                Paragraph(f"<b>{cfg_name}</b>", body_style),
                Paragraph(f"{mean_lat:.1f} ms", body_style),
                Paragraph(f"{n_succ} / {len(sub)}", body_style),
                Paragraph(f"{n_part} / {len(sub)}", body_style),
                Paragraph(f"{n_fail} / {len(sub)}", body_style),
                Paragraph(f"<b>{rate:.1f}%</b>", body_style)
            ])

        stable = Table(stats_rows, colWidths=[130, 70, 80, 70, 70, 95])
        stable.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, border_color),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(stable)
        story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 10: OBSERVATIONS & CONCLUSION
    # ---------------------------------------------------------
    story.append(Paragraph("10. Analytical Observations, Failure Modes, and Conclusion", h1_style))
    story.append(Paragraph(
        "<b>Synthesis of Experimental Findings:</b><br/>"
        "1. <b>The Precision-Recall Trade-off in Cascade Models:</b> No single static parameter setting is optimal across all environments. "
        "Config 2 (<code>scaleFactor=1.05, minNeighbors=2</code>) maximizes recall, catching small and degraded faces, but suffers high false alarm rates under tilted angles. "
        "Config 3 (<code>scaleFactor=1.30, minNeighbors=8</code>) offers lightning-fast execution (~6 ms) and zero false positives, but blindly ignores tiny faces (&le; 30 px) and clipped lighting.<br/>"
        "2. <b>Rotational Invariance Limitation:</b> The Haar Cascade's fundamental architectural flaw is its lack of in-plane rotational invariance. "
        "While modern Convolutional Neural Networks (CNNs) and vision transformers learn rotationally robust feature embeddings, Viola-Jones detectors "
        "require either manual image orientation uprighting or training separate multi-angle cascade models.<br/>"
        "3. <b>Profile Pose Discrepancy:</b> Experiments with <code>haarcascade_profileface.xml</code> demonstrated that frontal models fail on lateral side poses, "
        "necessitating a dual-cascade pipeline where frontal and profile classifiers run concurrently to detect arbitrary head poses.<br/>"
        "4. <b>Summary:</b> The improved implementation (<code>main_improved.py</code>) and automated benchmark suite successfully resolved all original code limitations, "
        "providing a rigorous scientific framework to evaluate and verify classical face detection principles.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Sign-off block
    sign_block = [
        [
            Paragraph("<b>Report Prepared By:</b> Shaik Jnaved (P26DS001)<br/>M.Tech. I — Computer Science and Engineering", body_style),
            Paragraph("<b>Evaluated At:</b> SVNIT Surat<br/>Academic Session 2026-2027", body_style)
        ]
    ]
    sign_table = Table(sign_block, colWidths=[260, 255])
    sign_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 1, primary_color),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(sign_table)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF report at: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
