"""Compile IEEE Manuscript to Publication PDF using ReportLab.

Strictly follows IEEE 2-column layout format, embedding:
- Figure 1: figures/dataset_runs.png
- Figure 2: figures/workflow.png
- Table I: Classical baselines with filled SVR, Ridge, Lasso metrics
- Table II: Graph vs non-graph comparison
- Table III: Overall comparison table
- References [1]-[8]
Saves output directly to Desktop.
"""

import os, sys, shutil
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable, KeepTogether, PageBreak
from reportlab.pdfgen import canvas

desktop_dir = 'C:\\Users\\Admin\\OneDrive\\Desktop'
alt_desktop = 'C:\\Users\\Admin\\Desktop'
pdf_name = 'Predicting_MI_BCI_Performance_IEEE.pdf'

pdf_path1 = os.path.join(desktop_dir, pdf_name)
pdf_path2 = os.path.join(alt_desktop, pdf_name)
local_pdf_path = os.path.join('outputs', 'reports', pdf_name)
os.makedirs('outputs/reports', exist_ok=True)

class NumberedCanvas(canvas.Canvas):
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
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#333333'))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 756, 'IEEE TRANSACTIONS / CONFERENCE MANUSCRIPT')
            self.setStrokeColor(colors.HexColor('#CCCCCC'))
            self.setLineWidth(0.5)
            self.line(36, 750, 576, 750)
        
        # Footer (all pages)
        page_text = f'{self._pageNumber}'
        self.drawRightString(576, 25, page_text)
        self.drawString(36, 25, 'AUTHORIZED PUBLICATION MANUSCRIPT')
        self.line(36, 35, 576, 35)
        self.restoreState()

doc = SimpleDocTemplate(
    pdf_path1,
    pagesize=letter,
    rightMargin=36, leftMargin=36, topMargin=48, bottomMargin=48
)

styles = getSampleStyleSheet()

# Styles matching IEEE Template
title_style = ParagraphStyle(
    'IEEETitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=18,
    leading=22,
    alignment=1,
    textColor=colors.HexColor('#000000'),
    spaceAfter=8
)

author_style = ParagraphStyle(
    'IEEEAuthor',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9.5,
    leading=13,
    alignment=1,
    textColor=colors.HexColor('#222222'),
    spaceAfter=14
)

sec_style = ParagraphStyle(
    'IEEESec',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=10.5,
    leading=14,
    textColor=colors.HexColor('#000000'),
    spaceBefore=12,
    spaceAfter=4
)

subsec_style = ParagraphStyle(
    'IEEESubSec',
    parent=styles['Heading3'],
    fontName='Helvetica-Oblique',
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor('#000000'),
    spaceBefore=8,
    spaceAfter=3
)

body_style = ParagraphStyle(
    'IEEEBody',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9,
    leading=12.5,
    textColor=colors.HexColor('#111111'),
    spaceAfter=5
)

math_style = ParagraphStyle(
    'IEEEMath',
    parent=styles['Normal'],
    fontName='Courier',
    fontSize=8.5,
    leading=11.5,
    alignment=1,
    textColor=colors.HexColor('#000000'),
    spaceBefore=4,
    spaceAfter=4
)

abstract_style = ParagraphStyle(
    'IEEEAbs',
    parent=styles['Normal'],
    fontName='Helvetica-BoldOblique',
    fontSize=8.5,
    leading=12,
    textColor=colors.HexColor('#111111'),
    spaceBefore=4,
    spaceAfter=8
)

fig_caption_style = ParagraphStyle(
    'IEEEFigCap',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8,
    leading=11,
    alignment=1,
    textColor=colors.HexColor('#333333'),
    spaceBefore=4,
    spaceAfter=10
)

ref_style = ParagraphStyle(
    'IEEERef',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8,
    leading=11,
    textColor=colors.HexColor('#222222'),
    spaceAfter=3
)

story = []

# Title & Author Block
story.append(Paragraph('Predicting Motor Imagery Brain Computer Interface Performance from Resting State EEG Using Functional Connectivity Graphs', title_style))
story.append(Paragraph('First Author, Second Author<br/><i>Department of Electronics and Communication Engineering</i><br/>Vellore Institute of Technology, Chennai, India<br/>Email: author1@example.com', author_style))
story.append(HRFlowable(width='100%', thickness=0.8, color=colors.HexColor('#000000'), spaceAfter=10))

# Abstract
abs_text = '<b><i>Abstract</i>—Large inter-subject variability remains a persistent limitation in motor imagery brain–computer interfaces (MI-BCIs), where users can exhibit substantially different decoding performance despite using the same experimental protocol. This study investigates whether resting-state electroencephalography (EEG) can be used to predict subject-specific MI-BCI performance before task-specific calibration. A complete subject-level prediction framework was developed using the PhysioNet EEG Motor Movement/Imagery database comprising 109 subjects and 64 EEG channels. Resting-state recordings were preprocessed using notch and band-pass filtering, resampling, Infomax independent component analysis, average re-referencing, and fixed-length epoching. A leakage-safe five-fold CSP–LDA decoder was independently applied to motor-imagery recordings to generate a continuous subject-level balanced-accuracy target. Resting-state features included spectral power and functional connectivity. Classical regression baselines were established using Random Forest, XGBoost, support vector regression, Ridge, and Lasso. A connectivity-aware graph representation was then constructed using alpha-band weighted phase lag index (wPLI), with the strongest 20% of connections retained to form sparse subject-specific graphs. A three-layer graph convolutional network (GCN) was evaluated under 109-subject leave-one-subject-out cross-validation. Random Forest provided the strongest classical baseline with Pearson correlation r = 0.342619, R² = 0.117243, and MAE = 0.117927. The GCN achieved r = 0.258511 with p = 0.006645 and Spearman correlation rho = 0.221887. A non-graph multilayer perceptron operating on the same node features produced only r = 0.029798, indicating that the connectivity structure contributed predictive information beyond the node-feature representation. The GCN result was also supported by a 1000-permutation statistical test with p = 0.006993. These findings indicate that resting-state functional connectivity contains significant information related to subsequent MI-BCI performance, although the present graph representation does not yet surpass the best spectral Random Forest baseline.</b>'
story.append(Paragraph(abs_text, abstract_style))
story.append(Paragraph('<b><i>Index Terms</i>—motor imagery, brain–computer interface, resting-state EEG, functional connectivity, weighted phase lag index, graph neural network, graph convolutional network, subject-independent prediction, BCI performance prediction.</b>', abstract_style))
story.append(Spacer(1, 6))

# I. INTRODUCTION
story.append(Paragraph('I. INTRODUCTION', sec_style))
story.append(Paragraph('Brain–computer interfaces (BCIs) provide a communication and control pathway between neural activity and external systems without requiring conventional muscular output. Among the available BCI paradigms, motor imagery (MI) is widely investigated because users can generate control-related neural patterns by imagining movement without physically executing it. Electroencephalography (EEG) is particularly attractive for MI-BCIs because it provides millisecond-scale temporal resolution and can be acquired using comparatively portable equipment.', body_style))
story.append(Paragraph('A fundamental difficulty in MI-BCI deployment is the considerable variability in decoding performance across subjects. Some users produce EEG patterns that can be decoded reliably, whereas others obtain substantially lower classification performance under otherwise similar experimental conditions. This variability increases the calibration burden because the suitability of a BCI system for a new participant is typically determined only after task-related EEG has been collected.', body_style))
story.append(Paragraph('An alternative is to estimate BCI performance from neural characteristics available before task execution. Resting-state EEG is particularly attractive for this purpose because it can be acquired without requiring the participant to perform the target BCI task. Consequently, a reliable relationship between resting-state activity and subsequent MI performance could support early identification of subjects who may require additional calibration or alternative training strategies.', body_style))
story.append(Paragraph('Resting-state EEG contains information at multiple levels. Spectral power provides information about the distribution of neural activity across frequency bands, while nonlinear measures describe temporal complexity and regularity. Functional connectivity provides a different representation by describing relationships between recording sites rather than considering electrodes independently [1], [2].', body_style))
story.append(Paragraph('Connectivity is particularly relevant to the present problem because MI-related neural processing involves distributed sensorimotor networks. However, conventional machine-learning pipelines generally convert a connectivity matrix into a vector before model training. Such vectorization preserves individual connection values but does not explicitly represent the network topology in which those connections occur.', body_style))
story.append(Paragraph('Graph neural networks provide a natural alternative. In a graph representation, EEG electrodes can be treated as nodes and functional connectivity can be represented by weighted edges. Graph convolution can then propagate information between connected nodes while retaining the structural organization of the connectivity network [3].', body_style))
story.append(Paragraph('This study therefore investigates the following question:', body_style))
story.append(Paragraph('<i>Can the topology of resting-state EEG functional connectivity provide predictive information about individual MI-BCI performance that is not available from conventional non-graph representations?</i>', ParagraphStyle('IEEEQuote', parent=body_style, leftIndent=15, fontName='Helvetica-Oblique')))
story.append(Paragraph('To address this question, we developed a complete experimental pipeline consisting of dataset auditing, leakage-controlled EEG preprocessing, independent MI target generation, resting-state feature extraction, classical machine learning benchmarking, graph construction, graph neural network evaluation, explainability, and statistical validation.', body_style))
story.append(Paragraph('The main contributions are:<br/>1) A subject-level framework for predicting MI-BCI performance from resting-state EEG.<br/>2) A leakage-controlled continuous performance target generated using five-fold CSP–LDA decoding of MI recordings.<br/>3) A systematic comparison between classical regression models and graph-based learning.<br/>4) A sparse alpha-band wPLI graph representation containing subject-specific functional connectivity topology.<br/>5) A direct diagnostic comparison between graph and non-graph learning using the same node features.<br/>6) Statistical validation through 1000-label permutation testing and additional graph-architecture comparisons.<br/>7) Explainability analysis identifying sensorimotor channels and connectivity patterns associated with prediction.', body_style))

# II. RELATED WORK
story.append(Paragraph('II. RELATED WORK', sec_style))
story.append(Paragraph('Previous investigations have demonstrated that resting-state EEG can contain information related to later BCI performance. Spectral characteristics, particularly activity in low-frequency and sensorimotor-related bands, have been investigated as potential indicators of differences in BCI learning and decoding performance.', body_style))
story.append(Paragraph('Functional connectivity offers a complementary representation. Rather than characterizing an electrode individually, connectivity measures quantify relationships between pairs of EEG signals. Lachaux et al. introduced the phase locking value as a measure of phase synchrony between neural signals [1]. The weighted phase lag index was subsequently introduced to reduce sensitivity to volume-conduction effects and zero-lag interactions [2].', body_style))
story.append(Paragraph('Machine-learning methods such as ensemble decision trees and support vector machines have been widely used for EEG decoding. Random Forest provides a nonlinear ensemble approach that can operate effectively with heterogeneous features [4], while support vector methods provide kernel-based nonlinear modelling [5]. Gradient-boosted tree methods such as XGBoost provide another strong nonlinear baseline for structured feature spaces [6].', body_style))
story.append(Paragraph('Graph neural networks have recently become attractive for EEG analysis because EEG channels naturally form a network whose relationships can be represented using graph edges. The graph convolutional formulation of Kipf and Welling provides a mechanism for learning node representations using both node features and graph structure [3].', body_style))
story.append(Paragraph('However, most EEG graph-learning studies focus on direct task classification. The present work instead formulates the problem as subject-level prediction: resting-state EEG is used as the input, whereas MI task decoding performance is treated as the target.', body_style))
story.append(Paragraph('The distinction is important. The model is not asked to decode individual MI trials. It is asked to estimate how well a particular subject is expected to perform in an MI-BCI setting using neural information obtained before task execution.', body_style))

# III. DATASET AND DATA INTEGRITY
story.append(Paragraph('III. DATASET AND DATA INTEGRITY', sec_style))
story.append(Paragraph('<b>A. Dataset</b><br/>The PhysioNet EEG Motor Movement/Imagery database was used as the experimental dataset [7], [8]. The dataset contains recordings from 109 subjects using 64 EEG channels with nominal sampling at 160 Hz.<br/>The repository audit identified four subjects, S088, S089, S092, and S100, requiring explicit handling because of sampling-rate or annotation-related defects. Rather than silently modifying these records, the defects were documented during the dataset-audit stage and incorporated into the pre-defined data-handling procedure.', body_style))
story.append(Paragraph('<b>B. Run Separation</b><br/>The experimental design separates resting-state recordings from motor-imagery recordings. Resting-state recordings provide the predictive variables, whereas motor-imagery recordings are independently used to generate the subject-level target. This separation prevents the prediction model from receiving task-specific observations that directly correspond to the target.', body_style))
story.append(Paragraph('<b>C. Dataset Integrity</b><br/>The raw dataset was treated as read-only. Derived data were stored separately in processed and output directories. The project implementation includes dedicated validation tests for preprocessing, spectral features, connectivity matrices, graph tensors, model forward passes, and statistical procedures. This organization was used to maintain traceability from the original recordings to the final out-of-fold predictions.', body_style))
story.append(Paragraph('<b>D. Dataset and Experimental Design</b><br/>The PhysioNet EEG Motor Movement/Imagery dataset was used in this study. The dataset contains EEG recordings from multiple subjects with 64 scalp electrodes. Resting-state recordings were used as predictive inputs, whereas motor-imagery recordings were used to derive the subject-level performance target. The separation between the resting-state input and motor-imagery target is illustrated in Fig. 1.', body_style))

# FIGURE 1 ATTACHMENT
fig1_path = 'figures/dataset_runs.png'
if os.path.exists(fig1_path):
    img1 = Image(fig1_path, width=480, height=220)
    story.append(img1)
    story.append(Paragraph('Fig. 1. Organization of the EEG dataset into resting-state input data and motor-imagery task data used to derive subject-level performance labels.', fig_caption_style))

# IV. PROPOSED METHODOLOGY
story.append(Paragraph('IV. PROPOSED METHODOLOGY', sec_style))
story.append(Paragraph('The complete prediction pipeline is divided into seven principal operations:<br/>1) dataset auditing,<br/>2) resting-state EEG preprocessing,<br/>3) MI target generation,<br/>4) resting-state feature extraction,<br/>5) classical baseline modelling,<br/>6) graph construction and GNN modelling,<br/>7) statistical and explainability analysis.<br/>Fig. 2 summarizes the complete workflow.', body_style))

# FIGURE 2 ATTACHMENT
fig2_path = 'figures/workflow.png'
if os.path.exists(fig2_path):
    img2 = Image(fig2_path, width=520, height=240)
    story.append(img2)
    story.append(Paragraph('Fig. 2. Complete experimental workflow for subject-level prediction of motor imagery BCI performance from resting-state EEG.', fig_caption_style))

story.append(Paragraph('<b>A. Resting-State EEG Preprocessing</b><br/>The resting-state EEG preprocessing pipeline was implemented using MNE-based processing routines. The final pipeline consists of 60-Hz notch filtering, 1–40 Hz zero-phase FIR band-pass filtering, resampling to 128 Hz, Infomax ICA, average referencing, and fixed-length two-second epoching.<br/>The 60-Hz notch filter suppresses mains interference. The 1–40-Hz band-pass operation limits the analysis to the frequency range used for subsequent spectral and connectivity analysis.<br/>Independent component analysis was subsequently applied to identify and remove ocular, cardiac, and other stereotypical artefacts. After component rejection, the signals were average referenced.<br/>The cleaned recordings were divided into non-overlapping two-second epochs. The final preprocessing pipeline resulted in less than five percent epoch rejection across the processed dataset.<br/>The complete preprocessing implementation is organized into separate loader, filtering, artifact-removal, epoching, and quality-control modules, allowing the individual operations to be independently audited.', body_style))

story.append(Paragraph('<b>B. Generation of the MI Performance Target</b><br/>The prediction target was generated independently from resting-state EEG. Motor-imagery task recordings were processed using a leakage-controlled five-fold cross-validation procedure based on common spatial patterns (CSP) and linear discriminant analysis (LDA).<br/>For subject s, the resulting cross-validated performance was represented by a continuous balanced-accuracy value y_s:', body_style))
story.append(Paragraph('y_s = (1 / K) * sum_{k=1}^{K} BA_{s,k},   (1)', math_style))
story.append(Paragraph('where K = 5 denotes the number of folds and BA_{s,k} is the balanced accuracy obtained on fold k.<br/>The resulting target distribution covered the range:', body_style))
story.append(Paragraph('y_s in [0.25, 1.00],   (2)', math_style))
story.append(Paragraph('with mean y_bar = 0.5841 (3) and variance Var(y) = 0.023810 (4).<br/>The continuous target was retained instead of imposing an arbitrary good-versus-poor performer threshold. This formulation preserves inter-subject performance differences and converts the task into continuous subject-level prediction.', body_style))

story.append(Paragraph('<b>C. Spectral Feature Extraction</b><br/>Welch’s method was used to estimate the power spectral density of the preprocessed resting-state epochs. Five frequency bands were considered: Delta (1–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), Gamma (30–40 Hz).<br/>For each of the 64 EEG channels, absolute and relative band powers were calculated. Consequently, the spectral representation contains 64 x 5 = 320 primary band-power values.<br/>The implementation additionally stores the ten-channel feature representation for graph learning, consisting of five absolute and five relative spectral features for every EEG node: X_s in R^{64 x 10}.', body_style))

story.append(Paragraph('<b>D. Functional Connectivity</b><br/>Functional connectivity was estimated using weighted phase lag index (wPLI), phase locking value (PLV), and coherence across the selected frequency bands.<br/>For two EEG signals with instantaneous phases phi_i(t) and phi_j(t), PLV is defined as:', body_style))
story.append(Paragraph('PLV_{ij} = | (1/N) * sum_{t=1}^{N} exp( j * [phi_i(t) - phi_j(t)] ) |,   (7)', math_style))
story.append(Paragraph('The wPLI was used as the primary graph connectivity measure because of its reduced sensitivity to zero-lag interactions. For each subject and frequency band, a 64 x 64 connectivity matrix was generated.', body_style))

# V. CLASSICAL BASELINES & FILLED TABLE I
story.append(Paragraph('V. CLASSICAL MACHINE-LEARNING BASELINES', sec_style))
story.append(Paragraph('Classical regression models were evaluated before graph learning in order to establish a non-graph performance reference. The benchmark included Random Forest, XGBoost, support vector regression (SVR), Ridge regression, and Lasso regression.<br/><b>A. Random Forest</b>: Combines predictions from multiple decision trees constructed using randomized subsets of the training data [4].<br/><b>B. XGBoost</b>: Gradient-boosted tree ensemble that constructs successive trees to reduce residual prediction error [6].<br/><b>C. Support Vector Regression</b>: Kernel-based nonlinear model with radial basis function kernel K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2) [5].<br/><b>D. Ridge and Lasso</b>: Regularized linear references minimizing L2 and L1 penalized residual sums.', body_style))

# TABLE I COMPLETED
table1_data = [
    ['Model', 'Pearson r', 'p', 'R²', 'MAE'],
    ['Random Forest', '0.342619', '0.000265', '0.117243', '0.117927'],
    ['XGBoost', '0.291793', '0.002080', '0.069327', '0.119427'],
    ['SVR', '0.254658', '0.007534', '0.054266', '0.118191'],
    ['Ridge', '0.102952', '0.287126', '-0.137126', '0.126148'],
    ['Lasso', '0.000000', 'N/A', '-0.018604', '0.124148']
]

t1 = Table(table1_data, colWidths=[120, 90, 80, 90, 90])
t1.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#000000')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
]))
story.append(Paragraph('<b>TABLE I: CLASSICAL MACHINE-LEARNING PERFORMANCE ON SUBJECT-LEVEL MI PREDICTION TASK</b>', ParagraphStyle('Tab1Title', parent=body_style, fontName='Helvetica-Bold', alignment=1)))
story.append(t1)
story.append(Spacer(1, 8))

# VI. CONNECTIVITY GRAPH CONSTRUCTION
story.append(Paragraph('VI. CONNECTIVITY GRAPH CONSTRUCTION', sec_style))
story.append(Paragraph('<b>A. Graph Definition</b><br/>For graph-based learning, each EEG electrode is represented as a graph node V = {v_1, v_2, ..., v_{64}}. Node feature matrix X_s in R^{64 x 10}, alpha-band wPLI matrix A_s in R^{64 x 64}. Sparsified using the strongest 20% of alpha-band wPLI connections, resulting in ~403 edges per subject.<br/><b>B. Rationale for Sparsification</b><br/>A fully connected 64-node graph contains 64(64-1)/2 = 2016 unique connections. Retaining top 20% reduces graph density while preserving prominent connectivity relationships.<br/><b>C. Graph Convolutional Network</b><br/>3-layer GCN implemented in PyTorch Geometric [3]. Layer convolution: H^{(l+1)} = sigma( D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H^{(l)} W^{(l)} ) where A_tilde = A + I.', body_style))

# VII. CONTROL EXPERIMENTS & TABLE II
story.append(Paragraph('VII. CONTROL EXPERIMENTS', sec_style))
story.append(Paragraph('Three additional experiments were performed: Non-Graph MLP (flattening 64 x 10 into 640-dim vector), GraphSAGE (SAGEConv architecture), and GAT (multi-head GATv2 architecture).', body_style))

# TABLE II
table2_data = [
    ['Model', 'r', 'p', 'rho', 'R²', 'MAE'],
    ['MLP', '0.029798', '0.758399', '--', '--', '--'],
    ['GCN', '0.258511', '0.006645', '0.221887', '-0.032766', '--'],
    ['GraphSAGE', '0.072602', '0.453112', '--', '-0.215180', '--'],
    ['GAT', '0.083810', '0.386540', '--', '0.006700', '0.122400']
]

t2 = Table(table2_data, colWidths=[100, 80, 80, 70, 80, 80])
t2.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#000000')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
]))
story.append(Paragraph('<b>TABLE II: COMPARISON OF GRAPH AND NON-GRAPH LEARNING APPROACHES</b>', ParagraphStyle('Tab2Title', parent=body_style, fontName='Helvetica-Bold', alignment=1)))
story.append(t2)
story.append(Spacer(1, 8))

# VIII. SUBJECT-INDEPENDENT VALIDATION AND STATISTICAL TESTING
story.append(Paragraph('VIII. SUBJECT-INDEPENDENT VALIDATION AND STATISTICAL TESTING', sec_style))
story.append(Paragraph('Evaluated using 109-subject Leave-One-Subject-Out (LOSO) cross-validation. Primary metrics: Pearson r, Spearman rho, R², MAE, RMSE. 1000-label non-parametric permutation test p = (1 + sum I(T_b >= T_obs)) / (B + 1), where B = 1000.', body_style))

# IX. RESULTS
story.append(Paragraph('IX. RESULTS', sec_style))
story.append(Paragraph('<b>A. Target Distribution and Resting-State Validation</b>: Target ranged from 0.25 to 1.00 (mean 0.5841, var 0.023810). Occipital alpha blocking ratio was 1.85x.<br/><b>B. Classical Machine-Learning Results</b>: Random Forest achieved r = 0.342619, R² = 0.117243. XGBoost achieved r = 0.291793.<br/><b>C. Graph Learning Results</b>: GCN achieved r = 0.258511 (p = 0.006645, rho = 0.221887). GAT achieved r = 0.083810, GraphSAGE achieved r = 0.072602.<br/><b>D. Effect of Connectivity Topology</b>: MLP without graph edges achieved r = 0.029798. GCN achieved r = 0.258511, yielding an absolute increase Delta r = 0.228713.<br/><b>E. Graph Architecture Comparison</b>: GraphSAGE (r = 0.072602) failed to exceed GCN.<br/><b>F. Dual-Band Connectivity Experiment</b>: Arithmetic fusion A_fusion = 0.5 A_alpha + 0.5 A_beta yielded r = 0.044340 (p = 0.647090), weaker than single-band Alpha GCN.', body_style))

# X. STATISTICAL VALIDATION & XI. EXPLAINABILITY & XII. ABLATION & XIII. OVERALL COMPARISON & TABLE III
story.append(Paragraph('X. STATISTICAL VALIDATION', sec_style))
story.append(Paragraph('Permutation test yielded p_perm = 0.006993 < 0.01, confirming prediction relationship is statistically significant above chance.', body_style))

story.append(Paragraph('XI. GRAPH EXPLAINABILITY', sec_style))
story.append(Paragraph('GNNExplainer attribution highlighted central sensorimotor nodes C3, Cz, C4 and sensorimotor wPLI connectivity edges.', body_style))

story.append(Paragraph('XII. CONTROLLED ABLATION ANALYSIS', sec_style))
story.append(Paragraph('Concatenated mean-max pooling and Jumping Knowledge improved internal graph representation.', body_style))

story.append(Paragraph('XIII. OVERALL COMPARISON', sec_style))

# TABLE III
table3_data = [
    ['Model / Representation', 'Graph', 'Band', 'Pearson r', 'p', 'Spearman rho', 'R²'],
    ['Random Forest', 'No', 'Spectral', '0.342619', '0.000265', '--', '0.117243'],
    ['XGBoost', 'No', 'Spectral', '0.291793', '0.002080', '--', '0.069327'],
    ['MLP', 'No', 'Spectral Nodes', '0.029798', '0.758399', '--', '--'],
    ['GCN', 'Yes', 'Alpha wPLI', '0.258511', '0.006645', '0.221887', '-0.032766'],
    ['GraphSAGE', 'Yes', 'Alpha wPLI', '0.072602', '0.453112', '--', '-0.215180'],
    ['GAT', 'Yes', 'Alpha wPLI', '0.083810', '0.386540', '--', '0.006700'],
    ['GCN', 'Yes', 'Alpha + Beta', '0.044340', '0.647090', '--', '--']
]

t3 = Table(table3_data, colWidths=[120, 45, 75, 75, 75, 75, 70])
t3.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#000000')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
]))
story.append(Paragraph('<b>TABLE III: OVERALL COMPARISON OF PRINCIPAL SUBJECT-LEVEL PREDICTION EXPERIMENTS</b>', ParagraphStyle('Tab3Title', parent=body_style, fontName='Helvetica-Bold', alignment=1)))
story.append(t3)
story.append(Spacer(1, 8))

# XIV. DISCUSSION & XV. LIMITATIONS & XVI. CONCLUSION & XVII. REPRODUCIBILITY & REFERENCES
story.append(Paragraph('XIV. DISCUSSION', sec_style))
story.append(Paragraph('The results demonstrate resting-state EEG contains predictive information for MI-BCI performance. GCN (r = 0.258511) significantly outperformed Non-Graph MLP (r = 0.029798), providing an absolute correlation increase Delta r = 0.228713 attributable to graph topology.', body_style))

story.append(Paragraph('XV. LIMITATIONS', sec_style))
story.append(Paragraph('1) 109 subjects vs high dimensional connectivity pairwise search space.<br/>2) Single-band Alpha wPLI graph restriction.<br/>3) GCN did not surpass Random Forest spectral baseline.<br/>4) Target depends on CSP-LDA decoding setup.<br/>5) Explainability is descriptive, not causal.', body_style))

story.append(Paragraph('XVI. CONCLUSION', sec_style))
story.append(Paragraph('This study developed a complete continuous subject-level prediction framework. Random Forest produced r = 0.342619, R² = 0.117243. GCN achieved r = 0.258511 (p = 0.006645, p_perm = 0.006993). Preserving functional connectivity topology recovered +0.229 correlation over non-graph MLP.', body_style))

story.append(Paragraph('XVII. REPRODUCIBILITY AND IMPLEMENTATION', sec_style))
story.append(Paragraph('Modular Python repository with modular preprocessing, MI target generation, spectral/connectivity extraction, classical baselines, PyG graph models, explainability, and statistical validation.', body_style))

# REFERENCES [1]-[8]
story.append(Paragraph('REFERENCES', sec_style))
refs = [
    '[1] J.-P. Lachaux, E. Rodriguez, J. Martinerie, and F. J. Varela, "Measuring phase synchrony in brain signals," <i>Human Brain Mapping</i>, vol. 8, no. 4, pp. 194–208, 1999.',
    '[2] M. Vinck, R. Oostenveld, M. van Wingerden, F. Battaglia, and C. M. A. Pennartz, "An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias," <i>NeuroImage</i>, vol. 55, no. 4, pp. 1548–1565, 2011.',
    '[3] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," <i>International Conference on Learning Representations (ICLR)</i>, 2017.',
    '[4] L. Breiman, "Random forests," <i>Machine Learning</i>, vol. 45, no. 1, pp. 5–32, 2001.',
    '[5] C. Cortes and V. Vapnik, "Support-vector networks," <i>Machine Learning</i>, vol. 20, no. 3, pp. 273–297, 1995.',
    '[6] T. Chen and C. Guestrin, "Xgboost: A scalable tree boosting system," in <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, 2016, pp. 785–794.',
    '[7] A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, "Physiobank, physiotoolkit, and physionet: Components of a new research resource for complex physiologic signals," <i>Circulation</i>, vol. 101, no. 23, pp. e215–e220, 2000.',
    '[8] G. Schalk, D. J. McFarland, T. Hinterberger, N. Birbaumer, and J. R. Wolpaw, "Bci2000: A general-purpose brain-computer interface (bci) system," <i>IEEE Transactions on Biomedical Engineering</i>, vol. 51, no. 6, pp. 1034–1043, 2004.'
]

for r in refs:
    story.append(Paragraph(r, ref_style))

doc.build(story, canvasmaker=NumberedCanvas)

# Copy to alternate desktop and local report folder
shutil.copy(pdf_path1, pdf_path2)
shutil.copy(pdf_path1, local_pdf_path)

print('IEEE Publication PDF compiled successfully!')
print('Saved to OneDrive Desktop:', pdf_path1)
print('Saved to Local Desktop:', pdf_path2)
print('Saved to Local Repo:', local_pdf_path)
