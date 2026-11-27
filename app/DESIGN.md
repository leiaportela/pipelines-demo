---
name: Vertex AI pipeline dashboard
description: Three models trained by one Vertex AI pipeline, drawn as ruled data boxes and plain charts on an engineering sheet (light) or blueprint (dark).
colors:
  drawing-ground: "#f6f7f5"
  drawing-sheet: "#ffffff"
  ink-navy: "#16233a"
  ink-navy-2: "#3f4b5e"
  ink-navy-3: "#5c6879"
  hairline: "#d6dbe2"
  row-hover: "#edf0f3"
  row-selected: "#e3ecf8"
  shell-navy: "#16233a"
  shell-ink: "#f2f4f7"
  shell-ink-2: "#aab6c6"
  shell-rule: "#2c3b55"
  process-blue: "#2d6cc0"
  series-work: "#2d6cc0"
  series-verification: "#13968c"
  series-rework: "#d9482b"
  series-setup: "#8a63d2"
  good: "#1f7a4d"
  warn: "#9a6700"
  bad: "#c0392b"
  blueprint-ground: "#10263f"
  blueprint-sheet: "#13304f"
  chalk-ink: "#eaf1fa"
  chalk-ink-2: "#c3d0df"
  chalk-ink-3: "#93a7bd"
  blueprint-hairline: "#2a4a6e"
  blueprint-hover: "#183a5e"
  blueprint-selected: "#1d4670"
  blueprint-shell: "#0b1c30"
  blueprint-shell-ink-2: "#8fa3ba"
  blueprint-shell-rule: "#22405f"
  blueprint-accent: "#7fb0f0"
  blueprint-focus: "#9cc4ff"
  blueprint-series-work: "#4f8cdc"
  blueprint-series-verification: "#169e8c"
  blueprint-series-rework: "#e0583a"
  blueprint-series-setup: "#9677de"
  blueprint-good: "#5cc28e"
  blueprint-warn: "#e3b341"
  blueprint-bad: "#ff8a75"
typography:
  display:
    fontFamily: "Archivo Variable, Archivo, system-ui, sans-serif"
    fontSize: "23px"
    fontWeight: 680
    lineHeight: 1.45
    letterSpacing: "-0.012em"
  figure:
    fontFamily: "Archivo Variable, Archivo, system-ui, sans-serif"
    fontSize: "23px"
    fontWeight: 650
    lineHeight: 1.15
    letterSpacing: "-0.02em"
    fontFeature: "tnum"
  headline:
    fontFamily: "Archivo Variable, Archivo, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 650
    lineHeight: 1.45
    letterSpacing: "-0.012em"
  title:
    fontFamily: "Archivo Variable, Archivo, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 650
    lineHeight: 1.45
    letterSpacing: "-0.012em"
  body:
    fontFamily: "Archivo Variable, Archivo, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "tnum"
  data:
    fontFamily: "Archivo Variable, Archivo, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "tnum"
  label:
    fontFamily: "Archivo Variable, Archivo, system-ui, sans-serif"
    fontSize: "12.5px"
    fontWeight: 640
    lineHeight: 1.45
    fontVariation: "'wdth' 92"
  axis:
    fontFamily: "Archivo Variable, Archivo, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
rounded:
  hairline: "1px"
  sm: "2px"
  md: "3px"
  lg: "4px"
  dot: "50%"
spacing:
  tight: "4px"
  cell: "10px"
  gutter: "12px"
  panel: "14px"
  sheet: "24px"
  split: "28px"
components:
  data-box:
    backgroundColor: "{colors.drawing-sheet}"
    textColor: "{colors.ink-navy}"
    rounded: "{rounded.hairline}"
    padding: "8px 10px 9px"
  data-box-head:
    textColor: "{colors.ink-navy}"
    typography: "{typography.label}"
    padding: "6px 10px"
  train-box-head:
    backgroundColor: "{colors.row-selected}"
    textColor: "{colors.ink-navy}"
    typography: "{typography.label}"
    padding: "6px 10px"
  shell-tab:
    backgroundColor: "{colors.shell-navy}"
    textColor: "{colors.shell-ink-2}"
    padding: "0 12px"
    height: "52px"
  shell-tab-selected:
    backgroundColor: "{colors.shell-navy}"
    textColor: "{colors.shell-ink}"
  button:
    backgroundColor: "{colors.drawing-sheet}"
    textColor: "{colors.ink-navy-2}"
    rounded: "{rounded.md}"
    height: "28px"
    padding: "0 8px"
  button-hover:
    backgroundColor: "{colors.row-hover}"
  select:
    backgroundColor: "{colors.drawing-sheet}"
    textColor: "{colors.ink-navy}"
    rounded: "{rounded.md}"
    padding: "5px 8px"
  table-row-selected:
    backgroundColor: "{colors.row-selected}"
    textColor: "{colors.ink-navy-2}"
    padding: "7px 10px"
---

# Design System: Vertex AI pipeline dashboard

## Overview

**Creative North Star: "The Pipeline Drawing Sheet"**

The dashboard's world is a working engineering sheet. In the light "drawing" theme it is ink-navy hairlines and type on near-white paper; in the dark "blueprint" theme it is chalk ink on navy. It shows what each of three models predicts, how good it is, and how it was trained. The pipeline itself is drawn as a stage-lane-stage diagram of ruled boxes joined by 1px rules.

Density is a presenter's desk tool: 14px body, 12.5px secondary text, tabular figures everywhere, 1px rules doing the structural work. Colour is spent on data and on one blue accent. Chrome stays navy, white and grey, and both themes must hold on a projector.

Depth is flat. Only things that float (chart tooltips) carry a shadow. Totals sit in square-shouldered boxes, never in rounded KPI cards.

**Key Characteristics:**
- Square data boxes with a ruled header band, 1px ink border, 1px radius.
- One technical grotesk (Archivo Variable) with tabular figures, no second face.
- A dark navy shell bar in both themes; the sheet beneath swaps between drawing and blueprint.
- Charts read their colours and type from the live theme tokens, so they follow the toggle.
- Pipeline drawn as boxes on ruled connectors; skipped steps are dashed.

## Colors

An ink-on-paper neutral world with one process-blue accent; the series hues carry data.

### Primary
- **Process Blue** (`process-blue`; `blueprint-accent` in dark): links, focus ring, select focus border, selection, slider accent, the training-box border, bar fills, and the primary line in charts. The only interactive accent.

### Series (data only)
- **Series Blue** (`series-work`): primary measure in charts (forecast line, curves, bar tracks).
- **Series Teal** (`series-verification`): secondary or comparison marks (non-best tuning trials) and the pipeline lane swatches.
- **Series Vermilion** (`series-rework`): the single emphasised mark (the best tuning trial) and one pipeline lane swatch.
- **Series Violet** (`series-setup`): additional lane swatch in the pipeline drawing.
Each has a lighter-ground-tuned dark counterpart (`blueprint-series-*`), always read through the `--s-*` custom properties.

### Status
- **Good**, **Warn**, **Bad**: text status only (Succeeded, "Not the latest run", Failed, alert border). Bad also drives the danger tint behind missed cells and the alert box.

### Neutral
- **Drawing Ground / Sheet** (`drawing-ground`, `drawing-sheet`; blueprint equivalents in dark): page ground and the fill of every box, table and input.
- **Ink Navy 1-3** (`ink-navy`, `-2`, `-3`; chalk ink in dark): primary text, secondary text, muted text and dashed reference lines.
- **Hairline** (`hairline`): 1px row and section rules. Box borders and header rules use full Ink Navy.
- **Row Hover / Row Selected**: hover fills and the selected row, picked bar, hit cell and training-box header.
- **Shell Navy, Shell Ink, Shell Ink 2, Shell Rule**: the top bar in both themes.

### Named Rules
**The Tokens Only Rule.** Component and chart colour comes from the `--*` custom properties (charts through `chartBase`), never literals, so both themes stay correct.
**The Blue Is Data Rule.** Process blue marks both the interactive state and the main series; red, teal and violet mark only data, never text.

## Typography

**Display, Body and Label Font:** Archivo Variable (with Archivo, system-ui, sans-serif), loaded with the width axis.

**Character:** One technical grotesk. Hierarchy comes from weight (520 to 680), size and a narrowed 92% width on box header labels, never from a second face or tracked caps.

### Hierarchy
- **Display** (680, 23px, 1.45): page title.
- **Figure** (650, 23px, 1.15, -0.02em; 18px small variant; 17px step duration): the one big number in a data box.
- **Headline** (650, 16px): section heads; train-method line.
- **Title** (650, 14px): h3 and empty-state titles.
- **Body** (400, 14px, 1.45): all running text; section notes cap at 70ch.
- **Data** (400, 13px): tables, bar lists, fresh line.
- **Label** (640, 12.5px, width 92%): box and step header bands, matrix headings.
- **Axis** (12px): chart axes and tooltip text.

### Named Rules
**The Tabular Rule.** Every figure is set with tabular numerals; the body default carries it.

## Layout

A single sheet, max 1600px wide, 24px side padding (14px under 720px), under a sticky 52px shell bar. Pages are stacks of sections (18px top padding, 1px hairline rule between, 22px gap) each with a head row of a 16px title and a muted note. Box rows are grids of 3, 4 or 5 equal columns on a 12px gutter. Two-up content uses 28px-gap splits (1.4:1, 1:1, 2:1 for the forecast chart); the model trio and splits collapse to one column at 1100px. The pipeline drawing is a 200px / fluid / 200px grid with 40px gaps. Tables scroll inside a ruled frame capped at 70vh. Desktop 1280 to 1600px is the target; the build only degrades gracefully below 720px, it is not designed for mobile.

## Elevation & Depth

Flat by default. Hierarchy is drawn with 1px rules and tonal fills (Row Selected for the active item).

### Shadow Vocabulary
- **Float** (light `0 8px 24px rgba(22,35,58,0.14), 0 2px 6px rgba(22,35,58,0.08)`; blueprint `0 10px 30px rgba(0,0,0,0.45), 0 2px 8px rgba(0,0,0,0.3)`): chart tooltips only.
- **Focus ring** (`0 0 0 3px` process blue at 22%): selects on focus.

### Named Rules
**The Drawn Not Lifted Rule.** Containers are drawn with lines. A shadow means the layer floats above the sheet.

## Shapes

Square-shouldered. Data boxes, steps, swatches and bar tracks use 1px radius or none; skeleton, alert and chart tooltip 2px; buttons and selects 3px; shell icon button and segmented control 4px. The only round shape is the freshness dot. Connectors between pipeline stages are 1px ink rules; skipped steps switch to a dashed border and muted header.

## Components

### Data Boxes
Process boxes: 1px Ink border, 1px radius, Sheet fill. A header band (Label type, 6px 10px) ruled off by a second Ink line may carry an aside link on the right; the body holds a Figure value and key/value rows (12.5px Ink 2 label, Ink value at 600, right-aligned, optional 9px swatch). The "How it was trained" variant has a process-blue border and a Row Selected header band.

### Shell Bar
Sticky, 52px, Shell Navy in both themes. Brand mark (a stepped line) and name (650, 15px), Overview/Churn/CLV/Demand tabs (520, 14px, Shell Ink 2; hover and selected go to Shell Ink with a 2px underline), a run selector (select styled on the shell), and a 32px bordered icon button for the theme toggle. Transitions 160ms ease-out.

### Buttons and Links
- **Button:** 3px radius, 28px high, 1px Hairline border, Sheet fill, Ink 2 text; hover Row Hover fill and Ink 3 border (140ms); disabled 45% opacity.
- **Text link action:** underlined Ink 2 text with a Hairline underline that turns Process Blue on hover.

### Inputs
Selects: Sheet fill, 1px Hairline border, 3px radius, 5px 8px, 13px text; focus swaps the border to Process Blue with a 3px ring at 22%. The threshold slider uses the accent colour inside a ruled bar on Sheet.

### Tables
Sheet fill in a 1px Hairline frame, sticky header ruled with Ink (12px, Ink 3, right-aligned figures, first column left), 7px 10px cells, Hairline row rules, sortable headers as plain buttons. Row hover Row Hover; selected row Row Selected; rows are keyboard-focusable.

### Confusion Matrix and Bar Lists
The matrix is a fixed-layout ruled table with a Figure value per cell; hit cells fill Row Selected, miss cells the danger tint. Bar lists use a name, an 8px track on Row Hover filled with Series Work, and a right-aligned value; the pickable variant highlights the pressed row with Row Selected.

### Pipeline Drawing (signature)
Stage boxes for inputs and outputs flank three lanes (one per model, labelled with its method), each lane a row of step boxes with a duration, a 5px share bar in a series colour and joined by 1px rules. Skipped steps are dashed.

### Status, Fresh Line, States
Status pairs a 14px icon with text (Good, Bad, or Ink 3 for skipped); never colour alone. The fresh line sits above page content (13px, run id, status, start, data date, a Warn "Not the latest run" notice). Loading uses shimmering skeleton blocks; errors a Bad-bordered alert with a "Try again" button; missing models a dashed empty box.

### Charts
ECharts on canvas, built from `chartBase`: Archivo 12px text, Hairline split lines, Ink-ruled axis lines, sheet-filled tooltip with a 1px ink border, 2px radius and the Float shadow. Primary series are Series Work lines at 2 to 2.5px; reference lines are dashed 1px Ink 3 with a text label; the forecast interval is a Series Work area at 22% opacity. Each chart is wrapped with a role and a text label.

## Do's and Don'ts

### Do:
- **Do** frame every total as a square data box with a ruled header band (1px Ink border, 1px radius).
- **Do** take all colour from the `--*` properties and, in charts, from `chartBase`, so both themes follow the toggle.
- **Do** set figures in Archivo with tabular numerals and format them with the shared helpers (`format.js`).
- **Do** use Bad, Good and Warn for status text, always paired with an icon or words.
- **Do** keep depth to 1px rules and tonal fills.
- **Do** draw reference lines dashed in Ink 3 and label them in text.

### Don't:
- **Don't** build rounded KPI cards.
- **Don't** colour body text with series hues.
- **Don't** add shadows to containers that sit on the sheet.
- **Don't** introduce a second typeface.
- **Don't** rely on hover alone for a key number; the audience is on a projector.
