// Theme and components for the Deep Learning portfolio report.
// Design: monochrome, spacious, easy to read. No coloured fills.
// Group with whitespace and thin gray hairlines.

#let ink = rgb("#111111")
#let soft = rgb("#5b5b5b")
#let hair = rgb("#d8d8d8")

// --- Display equation with room around it. Use for every formula worth its own line.
#let eq(body) = block(width: 100%, above: 14pt, below: 14pt)[
  #align(center)[#body]
]

// --- Aside: a small heading followed by normal prose. No box, no rule.
#let note(title: [Note], body) = block(
  width: 100%, breakable: true, above: 14pt, below: 14pt,
)[
  #text(weight: "bold", size: 11pt)[#title]
  #v(3pt)
  #body
]

// --- A design decision with its justification. The rubric rewards justified
// choices explicitly ("Begrunder klart designvalg"), so make them visible.
#let choice(title: [Design choice], body) = block(
  width: 100%, breakable: true, above: 14pt, below: 14pt,
)[
  #line(length: 100%, stroke: 0.4pt + hair)
  #v(6pt)
  #text(weight: "bold", size: 11pt)[#title]
  #v(3pt)
  #body
  #v(6pt)
  #line(length: 100%, stroke: 0.4pt + hair)
]

// --- A limitation or caveat. Same plain styling; the heading carries the point.
#let limitation(title: [Limitation], body) = block(
  width: 100%, breakable: true, above: 14pt, below: 14pt,
)[
  #text(weight: "bold", size: 11pt)[#title]
  #v(3pt)
  #body
]

// --- Code excerpt with a caption explaining what it does and why.
// The rubric grades code inclusion on whether it is *explained* and tied to
// theory, so `explain` is a required argument, not an optional one.
#let codeblock(caption: [], explain: [], code) = block(
  width: 100%, breakable: true, above: 16pt, below: 16pt,
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
  width: 100%, breakable: false, above: 16pt, below: 16pt,
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
  width: 100%, breakable: true, above: 16pt, below: 16pt,
)[
  #table(
    columns: columns,
    stroke: none,
    inset: (x: 8pt, y: 6pt),
    ..cells,
  )
  #if caption != [] {
    v(5pt)
    text(size: 10pt)[#text(weight: "bold")[Table. ] #caption]
  }
]

// --- Placeholder for a number you have not filled in yet. Prints visibly so an
// unfilled result cannot slip into the hand-in unnoticed.
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
