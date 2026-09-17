#import "lib.typ": *

// Group members, defined once and used for both the PDF metadata and the
// title block so the two cannot drift apart.
#let authors = (
  "Mathias Klintebjerg Phillip",
  "Mads Paaske Pedersen",
  "Magnus Koldby Mackenhauer",
)

#set document(
  title: "Portfolio Assignment 1: Deep Learning",
  author: authors,
)
#set page(
  paper: "a4",
  margin: (x: 1.9cm, y: 1.9cm),
  numbering: "1",
)
#set text(font: "New Computer Modern", size: 10pt, lang: "en", fill: ink)
#set par(justify: true, leading: 0.58em, spacing: 0.78em)
#set heading(numbering: "1.1")

// Headings stay in the text flow. No forced page breaks: the report is short
// enough to read straight through.
#show heading.where(level: 1): it => {
  v(11pt, weak: true)
  block(text(size: 14pt, weight: "bold", it))
  v(5pt, weak: true)
}
#show heading.where(level: 2): it => {
  v(8pt, weak: true)
  block(text(size: 11pt, weight: "bold", it))
  v(3pt, weak: true)
}
#show link: it => underline(text(fill: ink)[#it])
#show raw: set text(font: "DejaVu Sans Mono", size: 7.5pt)

// --- compact title block ---
#align(center)[
  #text(size: 19pt, weight: "bold")[Portfolio Assignment 1]
  #v(3pt)
  #text(size: 11pt)[Image classification on BloodMNIST]
  #v(3pt)
  #text(size: 9pt, fill: soft)[
    Deep Learning within the Health Technology Domain, 5th semester, SDU
  ]
  #v(3pt)
  #text(size: 9pt)[#authors.join(", ")]
  #v(2pt)
  #text(size: 9pt, fill: soft)[#datetime.today().display("[day]. [month repr:long] [year]")]
]
#v(10pt)
#line(length: 100%, stroke: 0.5pt + ink)

#outline(title: none, depth: 1, indent: auto)
#v(4pt)
#line(length: 100%, stroke: 0.5pt + hair)

#include "chapters/introduction.typ"
#include "chapters/data-and-setup.typ"
#include "chapters/knn.typ"
#include "chapters/linear-classifiers.typ"
#include "chapters/assignment-1-3.typ"
#include "chapters/discussion.typ"
#include "chapters/conclusion.typ"
