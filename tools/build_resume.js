/**
 * Smart Resume — DOCX Builder
 * Reads resume JSON from stdin, writes formatted DOCX to stdout path arg.
 * Usage: node build_resume.js /path/output.docx < resume.json
 */

const {
  Document, Packer, Paragraph, TextRun, AlignmentType,
  LevelFormat, BorderStyle, WidthType, HeadingLevel,
  UnderlineType, TabStopType, TabStopPosition,
} = require("docx");
const fs   = require("fs");
const path = require("path");

const outPath = process.argv[2];
if (!outPath) { console.error("Usage: node build_resume.js <output.docx>"); process.exit(1); }

let raw = "";
process.stdin.on("data", d => raw += d);
process.stdin.on("end", () => {
  let data;
  try { data = JSON.parse(raw); }
  catch(e) { console.error("JSON parse error:", e.message); process.exit(1); }

  const { name, contact, objective, skills, experience, projects, education, certifications } = data;

  // ── Colours & fonts ───────────────────────────────────────────────────────
  const ACCENT  = "1A3C6E";   // dark navy
  const DIVIDER = "CCCCCC";
  const FONT    = "Calibri";
  const BODY_SZ = 20;         // 10pt in half-points
  const HDR_SZ  = 22;         // 11pt

  // ── Helpers ───────────────────────────────────────────────────────────────
  const hr = () => new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT } },
    spacing: { before: 60, after: 60 },
    children: [],
  });

  const sectionHead = (text) => new Paragraph({
    spacing: { before: 180, after: 40 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT } },
    children: [new TextRun({
      text: text.toUpperCase(),
      font: FONT, size: HDR_SZ + 2, bold: true, color: ACCENT,
    })],
  });

  const bodyText = (text, opts = {}) => new TextRun({
    text, font: FONT, size: BODY_SZ,
    bold:      opts.bold      || false,
    italics:   opts.italic    || false,
    color:     opts.color     || "000000",
  });

  const bullet = (text) => new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 20, after: 20 },
    children: [bodyText(text)],
  });

  // ── Name block ────────────────────────────────────────────────────────────
  const children = [];

  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 60 },
    children: [new TextRun({ text: name, font: FONT, size: 36, bold: true, color: ACCENT })],
  }));

  if (contact) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 40 },
      children: [bodyText(contact, { color: "444444" })],
    }));
  }

  children.push(hr());

  // ── Objective ─────────────────────────────────────────────────────────────
  if (objective && objective.trim()) {
    children.push(sectionHead("Objective"));
    children.push(new Paragraph({
      spacing: { before: 60, after: 60 },
      children: [bodyText(objective)],
    }));
  }

  // ── Technical skills ─────────────────────────────────────────────────────
  if (skills && Object.keys(skills).length > 0) {
    children.push(sectionHead("Technical Skills"));
    for (const [category, items] of Object.entries(skills)) {
      if (!items || !items.length) continue;
      children.push(new Paragraph({
        spacing: { before: 40, after: 20 },
        children: [
          new TextRun({ text: category + ": ", font: FONT, size: BODY_SZ, bold: true }),
          bodyText(Array.isArray(items) ? items.join(", ") : items),
        ],
      }));
    }
  }

  // ── Experience ────────────────────────────────────────────────────────────
  if (experience && experience.length > 0) {
    children.push(sectionHead("Experience"));
    for (const exp of experience) {
      // Role | Company line
      children.push(new Paragraph({
        spacing: { before: 100, after: 20 },
        children: [
          new TextRun({ text: exp.role || "", font: FONT, size: BODY_SZ, bold: true }),
          bodyText("  |  " + (exp.company || ""), { color: "333333" }),
          ...(exp.duration ? [bodyText("  |  " + exp.duration, { italic: true, color: "666666" })] : []),
        ],
      }));
      for (const resp of (exp.responsibilities || [])) {
        if (resp && resp.trim()) children.push(bullet(resp.trim()));
      }
    }
  }

  // ── Projects ─────────────────────────────────────────────────────────────
  if (projects && projects.length > 0) {
    children.push(sectionHead("Projects"));
    for (const proj of projects) {
      const techStr = proj.technologies && proj.technologies.length
        ? "  |  " + proj.technologies.join(", ")
        : "";
      children.push(new Paragraph({
        spacing: { before: 100, after: 20 },
        children: [
          new TextRun({ text: proj.title || "", font: FONT, size: BODY_SZ, bold: true }),
          bodyText(techStr, { italic: true, color: "555555" }),
        ],
      }));
      if (proj.description && proj.description.trim()) {
        children.push(bullet(proj.description.trim()));
      }
      for (const m of (proj.metrics || [])) {
        if (m && m.trim()) children.push(bullet(m.trim()));
      }
    }
  }

  // ── Education ────────────────────────────────────────────────────────────
  if (education && education.length > 0) {
    children.push(sectionHead("Education"));
    for (const edu of education) {
      const parts = [edu.degree, edu.institution].filter(Boolean).join("  |  ");
      const sub   = [edu.graduation_year, edu.gpa ? "GPA: " + edu.gpa : null].filter(Boolean).join("  |  ");
      children.push(new Paragraph({
        spacing: { before: 80, after: 10 },
        children: [new TextRun({ text: parts, font: FONT, size: BODY_SZ, bold: true })],
      }));
      if (sub) {
        children.push(new Paragraph({
          spacing: { before: 0, after: 40 },
          children: [bodyText(sub, { color: "555555" })],
        }));
      }
    }
  }

  // ── Certifications ───────────────────────────────────────────────────────
  if (certifications && certifications.length > 0) {
    children.push(sectionHead("Certifications"));
    for (const cert of certifications) {
      const line = [cert.name, cert.issuer, cert.year].filter(Boolean).join("  |  ");
      children.push(bullet(line));
    }
  }

  // ── Build document ───────────────────────────────────────────────────────
  const doc = new Document({
    numbering: {
      config: [{
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 360, hanging: 180 } } },
        }],
      }],
    },
    styles: {
      default: {
        document: { run: { font: FONT, size: BODY_SZ } },
      },
    },
    sections: [{
      properties: {
        page: {
          size:   { width: 12240, height: 15840 },
          margin: { top: 900, right: 1080, bottom: 900, left: 1080 },
        },
      },
      children,
    }],
  });

  Packer.toBuffer(doc).then(buf => {
    fs.writeFileSync(outPath, buf);
    console.log("OK:" + buf.length);
  }).catch(e => { console.error("Pack error:", e.message); process.exit(1); });
});