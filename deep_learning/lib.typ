// Theme and components for the Deep Learning portfolio report.
// Design: monochrome, spacious, easy to read.
// Group with whitespace and thin gray hairlines.

#let ink = rgb("#111111")
#let soft = rgb("#5b5b5b")
#let hair = rgb("#d8d8d8")

// --- Display equation with room around it. Use for every formula worth its own line.
#let eq(body) = block(width: 100%, above: 10pt, below: 10pt)[
  #align(center)[#body]
]

// --- Aside: a small heading followed by normal prose. No box, no rule.
#let note(title: [Note], body) = block(
  width: 100%, breakable: true, above: 10pt, below: 10pt,
)[
  #text(weight: "bold", size: 11pt)[#title]
  #v(3pt)
  #body
]

#let choice(title: [Design choice], body) = block(
  width: 100%, breakable: true, above: 10pt, below: 10pt,
)[
  #text(weight: "bold", size: 11pt)[#title]
  #v(3pt)
  #body
]

// --- A limitation or caveat.
#let limitation(title: [Limitation], body) = block(
  width: 100%, breakable: true, above: 10pt, below: 10pt,
)[
  #text(weight: "bold", size: 11pt)[#title]
  #v(3pt)
  #body
]

// --- Code excerpt with a caption and a required explanation.
#let codeblock(caption: [], explain: [], code) = block(
  width: 100%, breakable: true, above: 11pt, below: 11pt,
)[
  #if caption != [] {
    text(weight: "bold", size: 10pt)[#caption]
    v(5pt)
  }
  #block(
    width: 100%,
    inset: (x: 10pt, y: 9pt),
    stroke: 0.5pt + hair,
    radius: 2pt,
  )[#code]
  #if explain != [] {
    v(6pt)
    text(size: 10pt, fill: soft)[#explain]
  }
]

// --- Figure with a caption that carries the interpretation, not just a label.
#let fig(caption: [], reading: [], img) = block(
  width: 100%, breakable: false, above: 11pt, below: 11pt,
)[
  #align(center)[#img]
  #v(7pt)
  #text(size: 10pt)[#text(weight: "bold")[Figure. ] #caption]
  #if reading != [] {
    v(3pt)
    text(size: 10pt, fill: soft)[#reading]
  }
]

// --- Results table with a consistent look.
#let restable(columns: auto, caption: [], ..cells) = block(
  width: 100%, breakable: true, above: 11pt, below: 11pt,
)[
  #table(
    columns: columns,
    stroke: (x, y) => if y == 0 { (bottom: 0.5pt + ink) } else { none },
    inset: (x: 7pt, y: 4.5pt),
    align: left,
    ..cells,
  )
  #if caption != [] {
    v(5pt)
    text(size: 10pt)[#text(weight: "bold")[Table. ] #caption]
  }
]

// --- A bug or pitfall worth calling out.
#let trap(title: [Pitfall], body) = block(
  width: 100%, breakable: true, above: 10pt, below: 10pt,
)[
  #text(weight: "bold", size: 11pt)[#title]
  #v(3pt)
  #body
]

// --- Visible placeholder for anything still missing.
#let todo(body) = box(
  fill: rgb("#f0f0f0"),
  outset: (y: 2pt),
  inset: (x: 3pt),
)[#text(weight: "bold", size: 9pt)[TODO: #body]]

// --- Part divider on its own page.
#let part(name) = {
  pagebreak(weak: true)
  heading(level: 1, numbering: none, name)
}
