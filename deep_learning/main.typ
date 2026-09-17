#import "lib.typ": *

#set document(
  title: "Portfolio Assignment 1 — Deep Learning",
  author: ("Mathias Phillip",), // add the other two group members here
)
#set page(
  paper: "a4",
  margin: (x: 2.2cm, y: 2.4cm),
  numbering: "1",
  header: context {
    let heads = query(selector(heading.where(level: 2)).before(here()))
    if heads.len() > 0 {
      set text(size: 8.5pt, fill: rgb("#64748b"))
      heads.last().body
      h(1fr)
      [Portfolio Assignment 1]
      line(length: 100%, stroke: 0.3pt + rgb("#cbd5e1"))
    }
  },
)
#set text(font: "New Computer Modern", size: 11pt, lang: "en", fill: ink)
#set par(justify: true, leading: 0.78em, spacing: 1.1em)
#set heading(numbering: "1.1")

// --- heading styles (monochrome) ---
#show heading.where(level: 1): it => {
  set text(size: 24pt, fill: ink, weight: "bold")
  v(10pt)
  block(it)
  v(4pt)
  line(length: 100%, stroke: 1pt + ink)
  v(14pt)
}
#show heading.where(level: 2): it => {
  pagebreak(weak: true)
  set text(size: 17pt, fill: ink, weight: "bold")
  block(it)
  v(8pt)
}
#show heading.where(level: 3): it => {
  set text(size: 12pt, fill: ink, weight: "bold")
  v(12pt)
  block(it)
  v(2pt)
}
#show heading.where(level: 4): it => {
  set text(size: 11pt, fill: ink, weight: "bold")
  v(10pt)
  block(it)
  v(1pt)
}
#show link: it => underline(text(fill: ink)[#it])
#show raw: set text(font: "DejaVu Sans Mono", size: 8.5pt)

// --- title page ---
#align(center + horizon)[
  #text(size: 34pt, weight: "bold", fill: ink)[Portfolio Assignment 1]
  #v(6pt)
  #text(size: 14pt)[Image classification on BloodMNIST]
  #v(2pt)
  #text(size: 11pt, fill: rgb("#64748b"))[
    k-Nearest Neighbors, linear classifiers, and neural networks
  ]
  #v(30pt)
  #text(size: 10pt, fill: rgb("#64748b"))[
    Deep Learning within the Health Technology Domain \
    5th semester, University of Southern Denmark
  ]
  #v(16pt)
  #text(size: 10pt)[Mathias Phillip #todo[+ two group members]]
  #v(2pt)
  #text(size: 10pt, fill: rgb("#64748b"))[#datetime.today().display("[day]. [month repr:long] [year]")]
]
#pagebreak()

// --- clickable table of contents ---
#outline(title: [Contents], depth: 3, indent: auto)
#pagebreak()

// ============================================================
#include "chapters/introduction.typ"
#include "chapters/data-and-setup.typ"

#part[Part I — Assignment 1.1: k-Nearest Neighbors]
#include "chapters/knn.typ"

#part[Part II — Assignment 1.2: Linear classifiers]
#include "chapters/linear-classifiers.typ"

#part[Part III — Assignment 1.3: Neural network]
#include "chapters/assignment-1-3.typ"

#part[Part IV — Discussion]
#include "chapters/discussion.typ"
#include "chapters/conclusion.typ"

#part[Appendix]
#include "chapters/appendix.typ"
