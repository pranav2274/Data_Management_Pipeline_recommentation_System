from pathlib import Path
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports'
results = pd.read_csv(REPORTS / 'evaluation_results.csv')
validation = json.loads((REPORTS / 'validation_report.json').read_text())
run = json.loads((REPORTS / 'run_summary.json').read_text())

pdf_path = REPORTS / 'final_report.pdf'
doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=17*mm, leftMargin=17*mm, topMargin=15*mm, bottomMargin=15*mm)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=19, leading=23, alignment=TA_CENTER, textColor=colors.HexColor('#20364b'), spaceAfter=4))
styles.add(ParagraphStyle(name='SubTitle', parent=styles['Normal'], fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor('#526579'), spaceAfter=12))
styles.add(ParagraphStyle(name='Section', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=colors.HexColor('#20364b'), spaceBefore=6, spaceAfter=5))
styles.add(ParagraphStyle(name='BodySmall', parent=styles['BodyText'], fontSize=8.5, leading=12, spaceAfter=5))
styles.add(ParagraphStyle(name='Tiny', parent=styles['BodyText'], fontSize=7, leading=9))
styles.add(ParagraphStyle(name='CaptionCustom', parent=styles['Normal'], fontSize=7, leading=9, alignment=TA_CENTER, textColor=colors.HexColor('#526579'), spaceAfter=4))

story=[]
story.append(Paragraph('End-to-End Data Management Pipeline for a Recommendation System', styles['ReportTitle']))
story.append(Paragraph('Data Management for Machine Learning - Assignment 2 | Final project report', styles['SubTitle']))
story.append(Paragraph('1. Objective and implementation summary', styles['Section']))
story.append(Paragraph(
    'This project implements a modular recommendation pipeline for a generic e-commerce catalog. It ingests user, item and interaction CSV files; validates and quarantines invalid events; cleans and transforms the accepted data; creates reusable user, item and interaction features; trains two recommendation approaches; evaluates top-K rankings; and records artifacts and run metadata. Deterministic synthetic data is used so the complete workflow can be reproduced offline.', styles['BodySmall']))
story.append(Paragraph(
    f"The successful run processed {run['counts']['users']:,} users, {run['counts']['items']:,} items and {run['counts']['raw_interactions']:,} raw interaction rows. After validation and quarantine, {run['counts']['valid_interactions']:,} valid interactions remained. The chronological per-user split produced {run['counts']['train_interactions']:,} training and {run['counts']['test_interactions']:,} test interactions.", styles['BodySmall']))
story.append(Paragraph('2. Pipeline architecture', styles['Section']))
img_path=REPORTS/'architecture.png'
if img_path.exists():
    im=Image(str(img_path), width=174*mm, height=68*mm)
    im.hAlign='CENTER'
    story.append(im)
    story.append(Paragraph('Figure 1. End-to-end workflow and model branches.', styles['CaptionCustom']))
story.append(Paragraph('3. Data quality, preparation and feature management', styles['Section']))
failed = [k for k,v in validation['checks'].items() if not v['passed']]
fail_text = ', '.join(failed) if failed else 'none'
story.append(Paragraph(
    f"Validation checks cover missing values, duplicate keys/IDs, numeric types, rating bounds, timestamp parsing and referential integrity. The raw interaction file contained {validation['summary']['quarantined_interactions']} records requiring quarantine: duplicate interaction IDs, one invalid timestamp, one out-of-range rating, one unknown user ID and one unknown item ID (some defects occur on the same duplicate rows). The raw checks flag these anomalies ({len(failed)} failing check categories: {fail_text}); the row-level error dataset records reasons, and all invalid rows are excluded. Post-quarantine clean-data checks passed.", styles['BodySmall']))
story.append(Paragraph(
    'Preparation standardizes IDs and text, parses timestamps as UTC, converts numeric fields, fills missing dimension attributes with safe defaults and removes invalid/duplicate interactions. A per-user chronological 80/20 split holds out each user\'s latest events. User activity summaries, item popularity, and a user-item rating matrix are generated from training data only. Feature tables include user activity count, mean user rating, preferred category, recency, item popularity, mean item rating, category, price, event type and timestamp.', styles['BodySmall']))

story.append(PageBreak())
story.append(Paragraph('4. Models and evaluation methodology', styles['Section']))
story.append(Paragraph(
    'The baseline ranks items using a Bayesian-smoothed combination of interaction count and average rating. The personalized model uses item-item collaborative filtering: it builds an item similarity matrix from positive (4-5 star) user interactions using cosine similarity, scores unseen items from each user\'s history, and uses popularity as a fallback when needed. Both models return top-5 items not already observed by the user.', styles['BodySmall']))
story.append(Paragraph(
    'A held-out event is relevant when its rating is at least 4. Precision@5 and Recall@5 measure relevance among the recommendations and held-out relevant items; NDCG@5 accounts for rank position. Coverage is the proportion of the catalog recommended to at least one evaluated user. Diversity is the average within-list fraction of item pairs with different categories. All 300 users with test events are included. These are ranking models and do not predict numeric ratings, so RMSE and MAE are not applicable.', styles['BodySmall']))
story.append(Paragraph('5. Comparative results', styles['Section']))
cols=['Model','Precision@5','Recall@5','NDCG@5','Coverage','Diversity']
data=[cols]
for _,r in results.iterrows():
    short='Popularity baseline' if 'Popularity' in r['model'] else 'Item-item CF'
    data.append([short, f"{r['precision@5']:.4f}", f"{r['recall@5']:.4f}", f"{r['ndcg@5']:.4f}", f"{r['coverage']:.4f}", f"{r['diversity']:.4f}"])
t=Table(data, colWidths=[43*mm, 25*mm, 23*mm, 23*mm, 23*mm, 23*mm], repeatRows=1)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#20364b')),
    ('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
    ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
    ('FONTSIZE',(0,0),(-1,-1),7.5),
    ('ALIGN',(1,1),(-1,-1),'CENTER'),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#f0f4f7')]),
    ('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#c5ced6')),
    ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
]))
story.append(t)
story.append(Spacer(1,4))
cf=results[results['model'].str.contains('collaborative', case=False)].iloc[0]
pop=results[results['model'].str.contains('Popularity', case=False)].iloc[0]
story.append(Paragraph(
    f"The item-item CF model achieved Precision@5 of {cf['precision@5']:.4f} versus {pop['precision@5']:.4f} for popularity, and Recall@5 of {cf['recall@5']:.4f} versus {pop['recall@5']:.4f}. It also recommended a much larger fraction of the catalog ({cf['coverage']:.1%} versus {pop['coverage']:.1%}). Its category diversity was lower ({cf['diversity']:.4f} versus {pop['diversity']:.4f}), which is a trade-off: personalized recommendations concentrate on the preferences inferred from a user\'s history, whereas the popularity list is more category-spread but less relevant in this test. The absolute metric values are specific to this synthetic data generator and should not be generalized to a live platform.", styles['BodySmall']))
story.append(Paragraph('6. Orchestration, tracking and reproducibility', styles['Section']))
story.append(Paragraph(
    'The dependency-aware Python workflow executes ingestion, validation, preparation, chronological transformation, feature building, both training jobs and evaluation in sequence. It writes a pipeline log and run summary, and fails with a captured error if a task raises an exception. A local JSON experiment tracker records a run ID, timestamp, model parameters, evaluation metrics and artifact paths. The workflow can be scheduled with cron or Windows Task Scheduler; setup commands and an example schedule are provided in README.md.', styles['BodySmall']))
story.append(Paragraph('7. Limitations and next steps', styles['Section']))
story.append(Paragraph(
    'The dataset is synthetic and intentionally contains controlled defects for validation demonstration. The collaborative model has cold-start limitations and its category-based diversity measure is a simple proxy for novelty. The local JSON tracker and Python scheduler are appropriate for a reproducible course project but not a multi-node production platform. Next steps would be to replace the synthetic source with MovieLens or a domain dataset, compare against matrix factorization or a content/hybrid model, add temporal drift checks, and use a production feature store, MLflow and a managed orchestration service.', styles['BodySmall']))
story.append(Spacer(1,5))
story.append(Paragraph('<b>Artifacts:</b> source modules, raw/processed/features CSVs, quarantined error CSV, validation JSON, model pickle files, experiment records, evaluation CSV, run log, architecture diagram and this report are included in the repository.', styles['Tiny']))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica',7)
    canvas.setFillColor(colors.HexColor('#6b7785'))
    canvas.drawString(17*mm, 9*mm, 'Recommendation Pipeline | Assignment 2')
    canvas.drawRightString(A4[0]-17*mm, 9*mm, f'Page {doc.page}')
    canvas.restoreState()

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(pdf_path)
