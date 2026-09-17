# Portfolio Assignment 1 — Deep Learning

Typst source for the Portfolio Assignment 1 report: image classification on
**BloodMNIST** using k-Nearest Neighbors (1.1), linear classifiers (1.2), and
assignment 1.3.

Course: *Deep Learning within the Health Technology Domain*, 5th semester, SDU.

The report is written here; the implementations live in the separate notebook
repo, [`Deep-Learning-Assignment-1`](https://github.com/Mads-Paaske/Deep-Learning-Assignment-1).

---

## Build

```bash
typst compile main.typ
```

Live preview while writing:

```bash
typst watch main.typ
```

Output lands in `main.pdf`, which is gitignored. Compile locally and share the
PDF manually; there is no CI build.

---

## Layout

| Path | What it holds |
|---|---|
| `main.typ` | Document setup, title page, table of contents, and the include order |
| `lib.typ` | Theme and components — all styling lives here, not in the chapters |
| `chapters/` | One file per report section |
| `figures/` | Exported plots from the notebooks (PNG/SVG) |

### Chapters, in document order

| File | Section |
|---|---|
| `introduction.typ` | Problem statement, scope, reading guide |
| `data-and-setup.typ` | Dataset, class imbalance, splits, preprocessing, metrics, tuning strategy |
| `knn.typ` | Assignment 1.1 — theory, implementation, sanity check, tuning, test results |
| `linear-classifiers.typ` | Assignment 1.2 — SVM and softmax loss, gradients, tuning, weight visualisation |
| `assignment-1-3.typ` | Assignment 1.3 — placeholder, headings mirror 1.1/1.2 |
| `discussion.typ` | Cross-model comparison, imbalance, limitations, improvements |
| `conclusion.typ` | Summary |
| `appendix.typ` | Reproducibility, extra figures, references |

The shared experimental setup (splits, preprocessing, metrics) is deliberately in
**one** chapter rather than repeated per assignment, so the three parts read as a
single report.

---

## Components

Import them with `#import "../lib.typ": *` at the top of a chapter.

| Component | Use |
|---|---|
| `#eq[...]` | Display equation with breathing room |
| `#note(title: [...])[...]` | Short aside |
| `#choice(title: [...])[...]` | A design decision *and its justification*, set off by hairlines |
| `#limitation(title: [...])[...]` | A caveat or limitation |
| `#codeblock(caption: [...], explain: [...])[...]` | Code excerpt with a required explanation |
| `#fig(caption: [...], reading: [...])[...]` | Figure with caption plus how to read it |
| `#restable(columns: ..., caption: [...])[...]` | Results table |
| `#todo[...]` | Grey inline placeholder — visible so it cannot slip into the hand-in |
| `#part[...]` | Part divider on its own page |

`codeblock` requires `explain` and `fig` takes `reading` on purpose: the grading
rubric awards points for code and figures that are *explained*, not merely
included.

---

## Grading rubric

Six criteria, 5 points each, **30 total** (`Rubrics Feedback Instruktion-1-1.pdf`
in the notebook repo). Where each is earned:

| Criterion | Primarily addressed in |
|---|---|
| Teoretisk beskrivelse af metode/modeller | The *Theory* section of each assignment chapter |
| Eksperimentelt design | `data-and-setup.typ` — splits, metrics, tuning; plus each `#choice` block |
| Præsentation og struktur | Document structure, headings, transitions |
| Visualiseringer | Every `#fig` — labelled axes and a stated reading |
| Diskussion af resultater | `discussion.typ` — implications, limitations, improvements |
| Inklusion og forklaring af eksempelkode | Every `#codeblock` and its `explain` |

Two things the rubric calls out that are easy to lose points on:

- **Justify design choices**, don't just state them. That is what `#choice` is for.
- **Discussion needs all three** of implications, limitations, and potential
  improvements — not just a summary of the numbers.

---

## Status

Drafted from the group's answers. Assignments 1.1 and 1.2 are written up with
real results; 1.3 is still a placeholder.

To find what is still unwritten:

```bash
grep -rn "todo\[" chapters/
```

### Results so far

| Model | Best config | Validation | Test |
|---|---|---|---|
| kNN | k=5, L1 | 23.6% | 17.6% |
| Linear, SVM | lr=1e-6, reg=1e3 | 75.8% | 70.6% |
| Linear, softmax | lr=1e-6, reg=1e2 | 71.8% | 68.8% |

### Known gaps

- **No figures yet.** The notebooks were run in Colab and saved without outputs,
  so every `#fig` is a placeholder. Re-run both and export plots to `figures/`.
- **No confusion matrices or balanced accuracy.** The assignment description asks
  for these explicitly, and the dataset is imbalanced, so plain accuracy is not
  enough on its own.
- **Class distribution is only given for the training split.** The assignment
  asks for all three.
- **Assignment 1.3** (neural network) has not been downloaded.
- **Group member names** are missing from the title page.
