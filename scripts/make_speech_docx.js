// Builds the presentation speech as a Word file. Text in **double asterisks** is bold.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, LevelFormat,
  PositionalTab, PositionalTabAlignment, PositionalTabRelativeTo, PositionalTabLeader,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
} = require("docx");

const GREEN = "1B5E3A";
const MUTED = "5B6573";

const slides = [
  ["Slide 1: Title", "0:15", [
    "Good morning, everyone. We are Koushik, Santosh, Srinith and Dharani Pratap. Our project is **SupplyAI-RL: Explainable Supply Chain Optimization**. In short, we built a system that decides how much stock a shop should order and from which supplier, and it explains its decisions in plain English.",
  ]],
  ["Slide 2: Outline", "0:15", [
    "This is our plan. First the problem and background. Then what we built: the system, the agent, the data, and the language-model parts. Then the dashboard, with live screenshots. Then the results, and a conclusion.",
  ]],
  ["Slide 3: Introduction and Motivation", "1:00", [
    "A shop can make two mistakes every day. If it orders too much, money sits in the warehouse and seasonal goods get marked down. If it orders too little, the customer buys somewhere else and that sale is gone. On top of that, there is the supplier choice. One supplier is cheap but slow. One is fast but expensive. One ships only 80% of what you order.",
    "This is a sequential problem. Today’s order becomes next week’s stock, so today’s decision changes what you can do later.",
    "Textbook rules like (s,S) or EOQ look at each product alone and assume demand behaves nicely. In our data, daily demand varies about 216 times more than a standard Poisson model allows. So our hypothesis was: a reinforcement learning agent can learn a better joint ordering policy. But a manager will not act on a number they cannot understand, so we added a language model that explains each decision.",
  ]],
  ["Slide 4: Problem Definition", "1:00", [
    "Here is the problem exactly. We have 10 real products, three suppliers, and 180 simulated days. The first 14 days are warm-up and are not scored. All products share one warehouse of 20,000 units.",
    "Each day, for each product, the system picks one of 21 options: seven quantity levels times three suppliers. Across ten products, that is about 17 trillion possible combinations every day. The goal is to maximise profit.",
    "The three suppliers are in the table. We invented their terms, because no public dataset has supplier details. No supplier is best: the cheap one ties up a week of stock in transit, the fast one eats your margin, and the middle one is unreliable. All money is in rupees, converted from pounds at 105 rupees per pound.",
  ]],
  ["Slide 5: Literature Review", "0:50", [
    "We anchored our work on five papers. PPO is our learning algorithm. The beer-game paper shows deep RL can match standard policies in a simple serial chain. The Gijsbrechts paper finds deep RL competitive with tuned heuristics, but not clearly better. The roadmap paper says fair benchmarking and interpretability are open problems. And the masking paper shows that blocking invalid actions works even with huge action spaces. The pattern is clear: RL has been competitive, not clearly superior, and it is hard to explain.",
  ]],
  ["Slide 6: Gaps Identified", "0:50", [
    "We found four gaps, and each has an answer. Most work uses single products or simple chains, so we model ten products, three suppliers and one shared warehouse. Learned policies cannot explain themselves, so we added an LLM explainer. Stress tests are usually written by hand, so we let an LLM write overlapping crises, and we validate every one. And baselines are often left untuned, so a new method wins for free. We tuned all four classical methods first.",
  ]],
  ["Slide 7: Objectives and Status", "0:50", [
    "Our five objectives, and all five are done. We built a simulator from real data. We trained a reinforcement learning agent. We built an LLM that explains decisions, with 100% delivered accuracy. We built an LLM that generates crisis scenarios. And we compared the agent with traditional policies.",
    "I want to be honest about the fifth one. It says ‘compare’, not ‘win’. The comparison came out against the AI, and we report it that way. The claim we can defend is that RL reaches 90.8% of a strong tuned baseline, and we know why it falls short.",
  ]],
  ["Slide 8: System Architecture", "1:15", [
    "This is the whole system, left to right. Real sales data builds a simulator. In the simulator, two decision makers act: the tuned classical engine, and the reinforcement learning challenger. The language-model layer does two jobs: it explains decisions, and it writes crises for testing. Evaluation and the dashboard are on the right.",
    "The green bar at the bottom is the deployment idea: shadow mode. The engine decides. The challenger runs beside it and is scored, but it never places an order. If it consistently beats the engine on live data, only then would it be promoted.",
  ]],
  ["Slide 9: Methodology: How the Agent Learns", "1:00", [
    "Every day the agent sees the state of the shop, as 88 numbers, and picks an action for each product. It gets a reward, which is that day’s profit. The algorithm is MaskablePPO.",
    "Three choices mattered. Action masking blocks impossible choices before the agent decides, such as an offline supplier. Without it, the best score was ₹24.9 lakh. With it, ₹76.4 lakh, about three times better. Choosing quantity and supplier together as one action made training 2.8 times faster. And hyperparameter tuning mattered: we tried eight settings and trained the best two to two million steps.",
  ]],
  ["Slide 10: Dataset Description", "1:15", [
    "Our data is the UCI Online Retail II dataset: 1,067,371 sales records from a UK gift shop, December 2009 to December 2011. We cleaned out cancellations and returns, and removed Saturdays because the shop never traded then.",
    "We chose ten products on purpose, in three types: four steady, three moderate and three bursty. If all ten behaved the same, one simple rule would handle them and there would be nothing for an AI to learn. The agent never sees the raw records. We measure five things from them and the simulator reproduces those.",
    "**(Covers a removed slide.)** Faculty may ask whether 2009 to 2011 data is too old. We tested that: we split the two years in half. The weekday pattern matched strongly, with a correlation of 0.87, and November was the busiest month in both years. So the demand shape is stable, but we cannot claim anything about today’s market. We also tested a newer 2022 to 2024 dataset and rejected it, because it had no weekly or seasonal pattern, so it is synthetic.",
  ]],
  ["Slide 11: Classical Baselines and Fair Evaluation", "1:00", [
    "Fairness is the core of this project. We have four classical methods, each tuned. Tuning lifted the best one from ₹69.7 lakh to ₹89.2 lakh on the tuning seeds. If we had left them untuned, our AI would have looked better than it is.",
    "We also use three separate groups of random seeds. 100 to 111 tune the classical methods. 900 to 905 pick the best AI checkpoint. And 500 to 529 are used only for reporting, never to choose anything. Every method faces the same customers and the same delivery delays on each seed, so we compare the per-seed difference.",
  ]],
  ["Slide 12: LLM Explainer", "1:15", [
    "This is the explainer. The facts come from the simulator: stock, orders on the way, demand, supplier details. The LLM writes two or three sentences. Then a check compares every number in the text against those facts. If it finds an invented number, a fixed template is shown instead.",
    "The key rule: **the LLM is never asked why.** The agent’s real reason is inside a neural network. Anything the LLM wrote after ‘because’ would be a believable story, not the cause. So it only restates facts. We say so openly. We measured this on 30 decisions: the raw model invented a number in 5 of them, and the check caught all five.",
  ]],
  ["Slide 13: Crisis Generator and Curriculum Training", "1:15", [
    "The second LLM job is imagination. We asked it for a realistic crisis, and it wrote four events that overlap: a viral demand spike, a supplier cyberattack, port delays and a cost jump. The overlap is what really hurts, and single-event tests never reach it.",
    "We do not trust anything it writes. Every scenario is validated: the type must be known, supplier and product numbers must exist, values are clamped, and windows are moved inside the 180 days. We fed it six deliberately broken inputs, and none survived as written.",
    "Then we tried curriculum training: 16 crisis sets, split ten for training and six for testing. The agent was tested only on crises it never saw. That split is the whole method, because testing on training crises would only measure memorising.",
  ]],
  ["Slide 14: Dashboard 1/5, Episode View", "0:45", [
    "Now the dashboard. This is the episode view. On the left you choose a policy, a test seed, and whether to inject a crisis. The cards show profit, fill rate, ordering fees, stockout cost and overflow. Here the engine earns ₹87.8 lakh in 180 days. The charts show warehouse stock and cumulative profit over time, and the orders by supplier. The engine sends almost every order to the cheap supplier.",
  ]],
  ["Slide 15: Dashboard 2/5, Challenger and Crisis", "1:15", [
    "On the left, we switch to the RL challenger. Look at the supplier bars: it placed 602 orders across all three suppliers, against 323 for the engine.",
    "**(Covers a removed slide.)** This is why the agent loses. It spreads its orders and pays about double the ordering fees. And the reason it cannot fix this is that its 88 inputs contain no ordering history: no days since its last order, no total fees paid, no last supplier used. It pays double and cannot see it. That is our main future experiment.",
    "On the right, the engine with a crisis injected. The shaded band shows when the disruption is active.",
  ]],
  ["Slide 16: Dashboard 3/5, Before vs After", "1:00", [
    "This tab answers: what does the system actually improve? ‘Before’ is a shop ordering the same fixed amount every day. ‘After’ is our engine. Profit goes from ₹53.3 lakh to ₹87.8 lakh, up 65%. Fill rate goes from 92.3% to 97.9%. Stockout cost falls 79%.",
    "To be honest about the source: most of this gain comes from forecasting with safety stock, which is the engine, not from reinforcement learning. The RL challenger also beats the fixed order, but it trails the engine.",
  ]],
  ["Slide 17: Dashboard 4/5, Explain a Decision", "0:50", [
    "Here is the explainer, live. You pick a day and a product and press the button. For day 45 it writes a short explanation. The black badge shows who wrote it: a model name, or ‘deterministic template’ if our check rejected the LLM. You can open ‘the facts the model was given’ and verify every number. For example, 2,365 units, 14 days of cover and 169 units of daily demand all appear in the facts.",
  ]],
  ["Slide 18: Dashboard 5/5, Comparison and Robustness", "0:40", [
    "These are the last two tabs. The comparison tab shows the 30 held-out seeds, with the best classical method at ₹87.8 lakh and the best agent at ₹79.7 lakh. The robustness tab compares calm runs against six unseen crises. Both tabs read our saved result files, so the dashboard cannot disagree with the report.",
  ]],
  ["Slide 19: Results 1/4, Main Comparison", "1:15", [
    "This is the main result, on 30 unseen runs. The best classical method, forecast plus safety stock, earns ₹87.8 lakh. Our tuned agent earns ₹79.7 lakh, which is **90.8%** of that. The classical method wins by ₹8.1 lakh per run, and the paired t-test gives t = −6.45, so this is real, not chance. The 95% confidence interval is 5.5 to 10.6 lakh.",
    "The agent does beat two of the four classical methods: EOQ and (s,S). And look at the last two rows. The AI without action masking is far behind, and ‘no system’ at all earns just ₹53.3 lakh.",
  ]],
  ["Slide 20: Results 2/4, What Each Step Was Worth", "0:50", [
    "This shows where the improvement came from. Plain PPO scored ₹23.7 lakh on test. Adding action masking took it to ₹74.8 lakh, about three times better. Tuning and longer training took it to ₹79.7 lakh. That closed 38% of the remaining gap to the classical method. The curves on the right show the learning progress.",
  ]],
  ["Slide 21: Results 3/4, Robustness", "1:15", [
    "We tested every method on six crises it had never seen. Every method loses more than a third of its profit. The classical method drops 38.6%. Our tuned agent drops 37.0%, about the same. So the AI is not more adaptive than the tuned classical method.",
    "We hoped crisis training would help. It did not. The crisis-trained agent earns ₹38.9 lakh in a crisis against ₹38.8 lakh for the control, and it gave up ₹7.7 lakh of normal profit. Its smaller percentage drop is misleading, because it has less to lose. The result is not significant. The useful finding: plain tuning made the agent more robust than crisis training did.",
  ]],
  ["Slide 22: Results 4/4, Explainer Accuracy", "0:50", [
    "We scored the explainer on 30 real decisions. The raw LLM was 83% accurate: 5 of 30 explanations contained an invented number. They read fluently, and a buyer would not have noticed. After our check, delivered accuracy was 100%. So the check is essential. Without it, one explanation in six would carry a made-up figure. We report both numbers, because 100% alone would hide the model’s real behaviour.",
  ]],
  ["Slide 23: Conclusion and Future Work", "1:15", [
    "To conclude. We built a complete system: simulator, RL agent, four tuned baselines, an explainer, a crisis generator and a dashboard. The tuned agent reaches 90.8% of the best classical profit, but the classical method still wins, and we know why. Masking gave about three times, tuning closed 38% of the rest, and crisis training did not help. The explainer is accurate only because it is checked.",
    "**(Covers a removed slide.)** Our limits: the data is from 2009 to 2011, the supplier terms are invented, and the explainer was tested on 30 decisions. So we do not recommend deploying the agent today. The honest use is shadow mode.",
    "Next, we would give the agent ordering memory and re-run on the same 30 seeds. That is the experiment most likely to close the gap.",
  ]],
  ["Slide 24: References", "0:10", [
    "These are our references. The full paper is in the repository.",
  ]],
  ["Slide 25: Thank You", "0:10", [
    "Thank you. We are happy to take questions.",
  ]],
];

const questions = [
  ["Why does your RL lose?", "It spreads orders across suppliers and pays double the fees, and it cannot see its own ordering history."],
  ["Where is the accuracy?", "Profit optimisation has no accuracy. We report relative profit of 90.8%, fill rate of 95.1%, and explainer accuracy of 100% delivered, 83% raw."],
  ["Is the data too old?", "The demand shape is stable across both years (weekday correlation 0.87), but we claim nothing about today."],
  ["What is action masking?", "Blocking illegal choices before the agent decides."],
];

function runs(text, base = {}) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((part) =>
    part.startsWith("**")
      ? new TextRun({ ...base, text: part.slice(2, -2), bold: true })
      : new TextRun({ ...base, text: part }));
}

const body = (t) => new Paragraph({ spacing: { after: 120, line: 300 }, children: runs(t) });

const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun({ text: "Presentation Speech", color: GREEN, bold: true })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "SupplyAI-RL: Explainable Supply Chain Optimization", size: 28, bold: true })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "22AIE450 Reinforcement Learning — Final Project Review", color: MUTED })] }),
  new Paragraph({
    spacing: { after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: GREEN, space: 6 } },
    children: [new TextRun({ text: "Koushik (CB.SC.U4AIE23312)  •  Santosh (CB.SC.U4AIE23310)  •  Srinith (CB.SC.U4AIE23357)  •  Dharani Pratap (CB.SC.U4AIE23330)", color: MUTED, size: 20 })],
  }),
  new Paragraph({ spacing: { after: 120, line: 300 }, children: runs("Matched to the 25-slide deck. About 15 minutes at a normal pace; the time beside each slide is a rough guide. The slides on data age, error analysis and limitations were removed from the deck, so their key points are spoken on the slides marked **(Covers a removed slide)**.") }),
];

for (const [title, time, paras] of slides) {
  children.push(new Paragraph({
    heading: HeadingLevel.HEADING_2,
    keepNext: true,
    children: [
      new TextRun({ text: title }),
      new TextRun({ children: [new PositionalTab({ alignment: PositionalTabAlignment.RIGHT, relativeTo: PositionalTabRelativeTo.MARGIN, leader: PositionalTabLeader.NONE }), time], color: MUTED, size: 22, bold: false }),
    ],
  }));
  paras.forEach((p) => children.push(body(p)));
}

children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("Likely questions, with short answers")] }));
for (const [q, a] of questions) {
  children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 100, line: 300 }, children: [new TextRun({ text: q + " ", bold: true }), new TextRun({ text: a })] }));
}
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Suggested split for four speakers")] }));
const split = [
  ["Slides 1–7", "Introduction, problem, literature, gaps, objectives"],
  ["Slides 8–13", "Architecture, method, data, baselines, explainer, crisis generator"],
  ["Slides 14–18", "The dashboard demo"],
  ["Slides 19–25", "Results, conclusion, questions"],
];
const W = [2200, 7160];
const cell = (t, w, head) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  margins: { top: 80, bottom: 80, left: 120, right: 120 },
  shading: head ? { type: ShadingType.CLEAR, fill: GREEN, color: "auto" } : undefined,
  children: [new Paragraph({ children: [new TextRun({ text: t, bold: head, color: head ? "FFFFFF" : undefined })] })],
});
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: W,
  rows: [
    new TableRow({ tableHeader: true, children: [cell("Slides", W[0], true), cell("Covers", W[1], true)] }),
    ...split.map(([a, b]) => new TableRow({ children: [cell(a, W[0], false), cell(b, W[1], false)] })),
  ],
}));

const doc = new Document({
  creator: "SupplyAI-RL team",
  title: "SupplyAI-RL Presentation Speech",
  styles: {
    default: { document: { run: { font: "Calibri", size: 23 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 48, bold: true, font: "Calibri" }, paragraph: { spacing: { after: 120 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 32, bold: true, color: GREEN }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, color: GREEN }, paragraph: { spacing: { before: 260, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1273, right: 1273 } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = process.argv[2];
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
